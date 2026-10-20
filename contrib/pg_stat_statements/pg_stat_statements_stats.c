/*-------------------------------------------------------------------------
 *
 * pg_stat_statements_stats.c
 *	  pgstat integration for pg_stat_statements.
 *
 * Copyright (c) 2008-2026, PostgreSQL Global Development Group
 *
 * IDENTIFICATION
 *	  contrib/pg_stat_statements/pg_stat_statements_stats.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "common/int.h"
#include "pg_stat_statements_internal.h"
#include "mb/pg_wchar.h"
#include "storage/dsm_registry.h"
#include "storage/s_lock.h"
#include "utils/dsa.h"
#include "utils/memutils.h"

/* Read/write helpers for stats file serialization, as in pgstat.c */
#define pgss_write_chunk(fpout, ptr, len) (fwrite(ptr, len, 1, fpout) == 1)
#define pgss_write_chunk_s(fpout, ptr) pgss_write_chunk(fpout, ptr, sizeof(*ptr))
#define pgss_read_chunk(fpin, ptr, len) (fread(ptr, 1, len, fpin) == (len))
#define pgss_read_chunk_s(fpin, ptr) pgss_read_chunk(fpin, ptr, sizeof(*ptr))

/* Backend-local pending entry */
typedef struct PgStat_PgssPending
{
	TimestampTz minmax_since[PGSS_NUMKIND];
	Counters	counters;
} PgStat_PgssPending;

typedef struct pgssResetFilter
{
	Oid			userid;
	Oid			dbid;
	int64		queryid;
} pgssResetFilter;

typedef struct pgssRestoredQueryText
{
	dsa_pointer text;
	int			len;
} pgssRestoredQueryText;

/* Eviction victim selection */
typedef struct pgssEvictCandidate
{
	uint64		objid;
	uint32		generation;
	pgssHashKey key;
	uint64		usage;
	int			query_len;
	bool		has_query_text;
} pgssEvictCandidate;

typedef struct pgssEvictContext
{
	pgssEvictCandidate *candidates;
	int			target;
} pgssEvictContext;

int			pgss_max = 5000;	/* target number of statements to track */
int			pgss_query_text_memory = PGSS_QUERY_TEXT_MEMORY_DEFAULT;	/* in KB */
bool		pgss_save = true;	/* whether to save stats across shutdown */

pgssSharedState *pgss_shared = NULL;
dshash_table *pgss_hash = NULL;

static dsa_area *pgss_query_text_dsa = NULL;
static pgssRestoredQueryText *pgss_restored_query_texts = NULL;
static int	pgss_num_restored_query_texts = 0;
static int	pgss_restored_query_text_capacity = 0;
static bool pgss_have_restored_entries = false;

static void pgss_init_shmem(void *ptr, void *arg);
static void pgss_attach_shmem_cb(void);
static uint64 pgss_query_text_memory_limit_bytes(void);
static bool pgss_reserve_query_text_space(int query_len);
static bool pgss_remember_restored_query_text(dsa_pointer text, int len);
static uint64 pgss_decay_usage(PgStatShared_Pgss *shared);
static void pgss_flush_kind(Counters *shared, Counters *pending,
							pgssStoreKind kind,
							TimestampTz minmax_stats_since,
							TimestampTz pending_minmax_since);
static PgStat_FlushResult pgss_flush_pending_cb(PgStat_EntryRef *entry_ref,
												bool nowait,
												bool xact_boundary);
static bool pgss_to_serialized_data(const PgStat_HashKey *key,
									const PgStatShared_Common *header,
									FILE *statfile);
static bool pgss_from_serialized_data(const PgStat_HashKey *key,
									  PgStatShared_Common *header,
									  FILE *statfile);
static void pgss_stats_file_finish(PgStat_StatsFileOp status);
static void pgss_accumulate_pending(PgStat_PgssPending *pending,
									pgssStoreKind kind,
									double total_time, uint64 rows,
									const BufferUsage *bufusage,
									const WalUsage *walusage,
									const JitInstrumentation *jitusage,
									int parallel_workers_to_launch,
									int parallel_workers_launched,
									PlannedStmtOrigin planOrigin);
static bool pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data);
static bool pgss_entry_matches_filter(PgStatShared_Pgss *shared,
									  pgssResetFilter *filter);
static bool pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data);
static bool pgss_drop_evicted_entry(PgStatShared_HashEntry *p, Datum match_data);
static void pgss_clear_query_text_locked(PgStatShared_Pgss *shared);
static int	pgss_evict_candidate_objid_cmp(const void *a, const void *b);
static int	pgss_evict_candidate_usage_cmp(const void *a, const void *b);
static const pgssEvictCandidate *pgss_find_evict_candidate(pgssEvictContext *context,
														   uint64 objid);

static const PgStat_KindInfo pgss_kind_info = {
	.name = "pg_stat_statements",
	.fixed_amount = false,
	.write_to_file = true,
	.accessed_across_databases = true,
	.own_hash = true,
	.shared_size = sizeof(PgStatShared_Pgss),
	.shared_data_off = offsetof(PgStatShared_Pgss, entry.counters),
	.shared_data_len = sizeof(Counters),
	.pending_size = sizeof(PgStat_PgssPending),
	.init_backend_cb = pgss_attach_shmem_cb,
	.flush_pending_cb = pgss_flush_pending_cb,
	.to_serialized_data = pgss_to_serialized_data,
	.from_serialized_data = pgss_from_serialized_data,
	.finish = pgss_stats_file_finish,
};

void
pgss_register_kind(void)
{
	pgstat_register_kind(PGSTAT_KIND_PGSS, &pgss_kind_info);
}

static void
pgss_init_shmem(void *ptr, void *arg)
{
	pgssSharedState *state = (pgssSharedState *) ptr;

	pg_atomic_init_u64(&state->entry_dealloc, 0);
	pg_atomic_init_u64(&state->entry_count, 0);
	pg_atomic_init_u64(&state->qtext_count, 0);
	pg_atomic_init_u64(&state->qtext_total_len, 0);
	pg_atomic_init_u64(&state->stats_reset, (uint64) GetCurrentTimestamp());
	pg_atomic_init_u64(&state->cur_median_usage, PGSS_USAGE_MEDIAN_INIT);
	LWLockInitialize(&state->entry_evict_lock,
					 LWLockNewTrancheId("pg_stat_statements entry eviction"));
}

/*
 * init_backend_cb: attach to the shared state.  Runs in every process that
 * uses the cumulative statistics system, so pgss_shared and pgss_query_text_dsa
 * are set wherever a stats callback or hook can run.
 */
static void
pgss_attach_shmem_cb(void)
{
	bool		found;

	if (pgss_shared == NULL)
		pgss_shared = GetNamedDSMSegment("pg_stat_statements_state",
										 sizeof(pgssSharedState),
										 pgss_init_shmem,
										 &found, NULL);

	if (pgss_query_text_dsa == NULL)
		pgss_query_text_dsa = GetNamedDSA("pg_stat_statements_qtext", &found);

	if (pgss_hash == NULL)
		pgss_hash = pgStatLocal.kind_hash[PGSTAT_KIND_PGSS];
}

void
pgss_require_ready(void)
{
	if (pgss_hash == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via shared_preload_libraries")));
}

PgStatShared_Pgss *
pgss_shared_from_hash_entry(PgStatShared_HashEntry *p)
{
	return (PgStatShared_Pgss *)
		dsa_get_address(pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS], p->body);
}

char *
pgss_query_text_address(dsa_pointer query_text)
{
	return dsa_get_address(pgss_query_text_dsa, query_text);
}

uint64
pgss_entry_dsa_size(void)
{
	dsa_area   *dsa = pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS];

	if (dsa == NULL)
		return 0;

	return (uint64) dsa_get_total_size(dsa);
}

uint64
pgss_query_text_dsa_size(void)
{
	if (pgss_query_text_dsa == NULL)
		return 0;

	return (uint64) dsa_get_total_size(pgss_query_text_dsa);
}

void
pgss_mark_entry_live(PgStatShared_Pgss *shared)
{
	if (pg_atomic_exchange_u32(&shared->entry_live, 1) == 0)
		pg_atomic_fetch_add_u64(&pgss_shared->entry_count, 1);
}

static bool
pgss_mark_entry_dead(PgStatShared_Pgss *shared)
{
	if (pg_atomic_exchange_u32(&shared->entry_live, 0) == 0)
		return false;

	pg_atomic_fetch_sub_u64(&pgss_shared->entry_count, 1);
	return true;
}

uint64
pgss_evict_threshold(void)
{
	return Max((uint64) pgss_max + 1,
			   (uint64) pgss_max * PGSS_EVICT_THRESHOLD_PERCENT / 100);
}

static uint64
pgss_query_text_memory_limit_bytes(void)
{
	return (uint64) pgss_query_text_memory * 1024;
}

bool
pgss_admission_pressure(uint64 threshold, int needed_query_len)
{
	uint64		limit = pgss_query_text_memory_limit_bytes();
	uint64		qtext_total_len;

	qtext_total_len = pg_atomic_read_u64(&pgss_shared->qtext_total_len);

	return pg_atomic_read_u64(&pgss_shared->qtext_count) >= threshold ||
		pg_atomic_read_u64(&pgss_shared->entry_count) >= threshold ||
		qtext_total_len >= limit ||
		(needed_query_len > 0 &&
		 ((uint64) needed_query_len > limit ||
		  (uint64) needed_query_len > limit - qtext_total_len));
}

static bool
pgss_reserve_query_text_space(int query_len)
{
	uint64		limit = pgss_query_text_memory_limit_bytes();
	uint64		old_total;
	uint64		new_total;

	if (query_len < 0 || !AllocSizeIsValid((Size) query_len + 1))
		return false;

	if ((uint64) query_len > limit)
		return false;

	old_total = pg_atomic_read_u64(&pgss_shared->qtext_total_len);
	for (;;)
	{
		if (old_total > limit ||
			(uint64) query_len > limit - old_total)
			return false;

		new_total = old_total + (uint64) query_len;
		if (pg_atomic_compare_exchange_u64(&pgss_shared->qtext_total_len,
										   &old_total, new_total))
			return true;
	}
}

/* Store query text in DSA, returning InvalidDsaPointer on allocation failure. */
dsa_pointer
pgss_store_query_text_dsa(const char *query, int query_len)
{
	dsa_pointer dp = InvalidDsaPointer;
	char	   *dst;
	Size		query_size;

	if (!pgss_reserve_query_text_space(query_len))
		return InvalidDsaPointer;

	query_size = (Size) query_len + 1;
	dp = dsa_allocate_extended(pgss_query_text_dsa, query_size, DSA_ALLOC_NO_OOM);

	if (!DsaPointerIsValid(dp))
	{
		pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
								(uint64) query_len);
		return InvalidDsaPointer;
	}

	dst = dsa_get_address(pgss_query_text_dsa, dp);
	memcpy(dst, query, query_len);
	dst[query_len] = '\0';
	pg_atomic_fetch_add_u64(&pgss_shared->qtext_count, 1);

	return dp;
}

void
pgss_free_dsa_query_text(dsa_pointer query_text, int query_len)
{
	if (!DsaPointerIsValid(query_text))
		return;

	Assert(query_len >= 0);

	pg_atomic_fetch_sub_u64(&pgss_shared->qtext_count, 1);
	pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
							(uint64) query_len);

	dsa_free(pgss_query_text_dsa, query_text);
}

static void
pgss_clear_query_text_locked(PgStatShared_Pgss *shared)
{
	Assert(LWLockHeldByMeInMode(&shared->header.lock, LW_EXCLUSIVE));

	pgss_free_dsa_query_text(shared->entry.query_text, shared->entry.query_len);
	shared->entry.query_text = InvalidDsaPointer;
	shared->entry.query_len = 0;
}

/*
 * Attach a stored query text to an entry, replacing any existing one.
 * The text is released instead if the entry died in the meantime.
 */
void
pgss_attach_query_text(PgStatShared_Pgss *shpgss, const pgssHashKey *key,
					   dsa_pointer query_text, int query_len, int encoding)
{
	Assert(DsaPointerIsValid(query_text) && query_len >= 0);

	LWLockAcquire(&shpgss->header.lock, LW_EXCLUSIVE);

	if (pg_atomic_read_u32(&shpgss->entry_live) == 0 ||
		memcmp(&shpgss->entry.key, key, sizeof(pgssHashKey)) != 0)
		pgss_free_dsa_query_text(query_text, query_len);
	else
	{
		pgss_clear_query_text_locked(shpgss);
		shpgss->entry.query_text = query_text;
		shpgss->entry.query_len = query_len;
		shpgss->entry.encoding = encoding;
	}

	LWLockRelease(&shpgss->header.lock);
}

/*
 * Remember a query text restored from the stats file, so that it can be
 * released if core later discards the file.
 */
static bool
pgss_remember_restored_query_text(dsa_pointer text, int len)
{
	if (!DsaPointerIsValid(text))
		return true;

	if (pgss_num_restored_query_texts >= pgss_restored_query_text_capacity)
	{
		int			newcapacity;
		Size		newsize;

		newcapacity = Max(pgss_restored_query_text_capacity * 2, 1024);
		if (!AllocSizeIsValid((Size) newcapacity * sizeof(pgssRestoredQueryText)))
			return false;
		newsize = (Size) newcapacity * sizeof(pgssRestoredQueryText);
		if (pgss_restored_query_texts == NULL)
			pgss_restored_query_texts =
				MemoryContextAllocExtended(TopMemoryContext, newsize,
										   MCXT_ALLOC_NO_OOM);
		else
		{
			pgssRestoredQueryText *newtexts;

			newtexts = repalloc_extended(pgss_restored_query_texts, newsize,
										 MCXT_ALLOC_NO_OOM);
			if (newtexts == NULL)
				return false;
			pgss_restored_query_texts = newtexts;
		}

		if (pgss_restored_query_texts == NULL)
			return false;

		pgss_restored_query_text_capacity = newcapacity;
	}

	pgss_restored_query_texts[pgss_num_restored_query_texts].text = text;

	pgss_restored_query_texts[pgss_num_restored_query_texts].len = len;
	pgss_num_restored_query_texts++;

	return true;
}

/*
 * Apply the eviction-pass decay factor to an entry's usage, and return the
 * new value.  Backends executing the statement can update usage or clear
 * sticky concurrently, so read both inside the compare-and-exchange loop.
 */
static uint64
pgss_decay_usage(PgStatShared_Pgss *shared)
{
	uint64		old_usage;
	uint64		new_usage;
	uint64		retain_percent;

	old_usage = pg_atomic_read_u64(&shared->usage);
	do
	{
		if (pg_atomic_read_u32(&shared->sticky) != PGSS_STICKY_NONE)
			retain_percent = PGSS_STICKY_RETAIN_PERCENT;
		else
			retain_percent = PGSS_USAGE_RETAIN_PERCENT;

		/* Split the multiplication so that it cannot overflow. */
		new_usage = old_usage / 100 * retain_percent +
			old_usage % 100 * retain_percent / 100;
	} while (!pg_atomic_compare_exchange_u64(&shared->usage,
											 &old_usage, new_usage));

	return new_usage;
}

/*
 * Merge one kind's pending timing counters into the shared entry.
 *
 * Pending batches that started before the last min/max reset still contribute
 * to calls, totals, mean and variance, but not to min/max.
 */
static void
pgss_flush_kind(Counters *shared, Counters *pending, pgssStoreKind kind,
				TimestampTz minmax_stats_since,
				TimestampTz pending_minmax_since)
{
	int64		n_a,
				n_b;
	double		delta;
	bool		merge_minmax;

	n_a = shared->calls[kind];
	n_b = pending->calls[kind];

	Assert(n_b > 0);
	Assert(pending_minmax_since != 0);

	merge_minmax = pending_minmax_since >= minmax_stats_since;

	shared->calls[kind] += n_b;
	shared->total_time[kind] += pending->total_time[kind];

	if (merge_minmax)
	{
		if (n_a == 0 ||
			(shared->min_time[kind] == 0 && shared->max_time[kind] == 0))
		{
			shared->min_time[kind] = pending->min_time[kind];
			shared->max_time[kind] = pending->max_time[kind];
		}
		else
		{
			shared->min_time[kind] =
				Min(shared->min_time[kind], pending->min_time[kind]);
			shared->max_time[kind] =
				Max(shared->max_time[kind], pending->max_time[kind]);
		}
	}

	if (n_a == 0)
	{
		shared->mean_time[kind] = pending->mean_time[kind];
		shared->sum_var_time[kind] = pending->sum_var_time[kind];
	}
	else
	{
		/*
		 * Chan's parallel variance algorithm: combine two sets of (count,
		 * mean, sum_of_squared_deviations). See
		 * <http://www.johndcook.com/blog/standard_deviation/>
		 */
		delta = pending->mean_time[kind] - shared->mean_time[kind];
		shared->sum_var_time[kind] +=
			pending->sum_var_time[kind] +
			delta * delta * (double) n_a * (double) n_b / (double) (n_a + n_b);
		shared->mean_time[kind] += delta * (double) n_b / (double) (n_a + n_b);
	}
}

static PgStat_FlushResult
pgss_flush_pending_cb(PgStat_EntryRef *entry_ref, bool nowait,
					  bool xact_boundary)
{
	PgStat_PgssPending *pending;
	PgStatShared_Pgss *shared;

	pending = (PgStat_PgssPending *) entry_ref->pending;
	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (pgss_entry_ref_is_stale(entry_ref))
	{
		memset(pending, 0, sizeof(*pending));
		return PGSTAT_FLUSH_DONE;
	}

	/* Pending entry was flushed and retained; nothing new. */
	if (pending->counters.calls[PGSS_PLAN] == 0 &&
		pending->counters.calls[PGSS_EXEC] == 0)
		return PGSTAT_FLUSH_DONE;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return PGSTAT_FLUSH_LOCK_CONFLICT;

	for (int kind = 0; kind < PGSS_NUMKIND; kind++)
	{
		if (pending->counters.calls[kind] > 0)
			pgss_flush_kind(&shared->entry.counters, &pending->counters,
							(pgssStoreKind) kind,
							shared->entry.minmax_stats_since,
							pending->minmax_since[kind]);
	}

#define PGSS_ACCUM_COUNTER(item)		\
	(shared)->entry.counters.item += (pending)->counters.item

	PGSS_ACCUM_COUNTER(rows);
	PGSS_ACCUM_COUNTER(shared_blks_hit);
	PGSS_ACCUM_COUNTER(shared_blks_read);
	PGSS_ACCUM_COUNTER(shared_blks_dirtied);
	PGSS_ACCUM_COUNTER(shared_blks_written);
	PGSS_ACCUM_COUNTER(local_blks_hit);
	PGSS_ACCUM_COUNTER(local_blks_read);
	PGSS_ACCUM_COUNTER(local_blks_dirtied);
	PGSS_ACCUM_COUNTER(local_blks_written);
	PGSS_ACCUM_COUNTER(temp_blks_read);
	PGSS_ACCUM_COUNTER(temp_blks_written);
	PGSS_ACCUM_COUNTER(shared_blk_read_time);
	PGSS_ACCUM_COUNTER(shared_blk_write_time);
	PGSS_ACCUM_COUNTER(local_blk_read_time);
	PGSS_ACCUM_COUNTER(local_blk_write_time);
	PGSS_ACCUM_COUNTER(temp_blk_read_time);
	PGSS_ACCUM_COUNTER(temp_blk_write_time);
	PGSS_ACCUM_COUNTER(wal_records);
	PGSS_ACCUM_COUNTER(wal_fpi);
	PGSS_ACCUM_COUNTER(wal_bytes);
	PGSS_ACCUM_COUNTER(wal_buffers_full);
	PGSS_ACCUM_COUNTER(jit_functions);
	PGSS_ACCUM_COUNTER(jit_generation_time);
	PGSS_ACCUM_COUNTER(jit_inlining_count);
	PGSS_ACCUM_COUNTER(jit_inlining_time);
	PGSS_ACCUM_COUNTER(jit_optimization_count);
	PGSS_ACCUM_COUNTER(jit_optimization_time);
	PGSS_ACCUM_COUNTER(jit_emission_count);
	PGSS_ACCUM_COUNTER(jit_emission_time);
	PGSS_ACCUM_COUNTER(jit_deform_count);
	PGSS_ACCUM_COUNTER(jit_deform_time);
	PGSS_ACCUM_COUNTER(parallel_workers_to_launch);
	PGSS_ACCUM_COUNTER(parallel_workers_launched);
	PGSS_ACCUM_COUNTER(generic_plan_calls);
	PGSS_ACCUM_COUNTER(custom_plan_calls);
#undef PGSS_ACCUM_COUNTER

	pgstat_unlock_entry(entry_ref);

	memset(pending, 0, sizeof(*pending));

	return PGSTAT_FLUSH_DONE;
}

/*
 * Serialize metadata and query text alongside the counters, which core has
 * already written.  Core writes every entry, so when pg_stat_statements.save
 * is off, or the entry is already dead, we write only a marker telling the
 * reader to drop it.
 */
static bool
pgss_to_serialized_data(const PgStat_HashKey *key,
						const PgStatShared_Common *header,
						FILE *statfile)
{
	PgStatShared_Pgss *shpgss = (PgStatShared_Pgss *) header;
	bool		save_entry;
	uint64		usage;
	uint64		entry_dealloc;
	uint64		stats_reset;
	int			qtext_len;
	bool		ok;

	if (!pgss_save)
	{
		save_entry = false;
		return pgss_write_chunk_s(statfile, &save_entry);
	}

	LWLockAcquire(&shpgss->header.lock, LW_SHARED);

	save_entry = pg_atomic_read_u32(&shpgss->entry_live) != 0 &&
		(shpgss->entry.counters.calls[PGSS_PLAN] > 0 ||
		 shpgss->entry.counters.calls[PGSS_EXEC] > 0);
	if (!pgss_write_chunk_s(statfile, &save_entry))
	{
		LWLockRelease(&shpgss->header.lock);
		return false;
	}
	if (!save_entry)
	{
		LWLockRelease(&shpgss->header.lock);
		return true;
	}

	usage = pg_atomic_read_u64(&shpgss->usage);
	entry_dealloc = pg_atomic_read_u64(&pgss_shared->entry_dealloc);
	stats_reset = pg_atomic_read_u64(&pgss_shared->stats_reset);
	qtext_len = DsaPointerIsValid(shpgss->entry.query_text) ? shpgss->entry.query_len : -1;

	ok = pgss_write_chunk_s(statfile, &shpgss->entry.key) &&
		pgss_write_chunk_s(statfile, &shpgss->entry.encoding) &&
		pgss_write_chunk_s(statfile, &shpgss->entry.stats_since) &&
		pgss_write_chunk_s(statfile, &shpgss->entry.minmax_stats_since) &&
		pgss_write_chunk_s(statfile, &entry_dealloc) &&
		pgss_write_chunk_s(statfile, &stats_reset) &&
		pgss_write_chunk_s(statfile, &usage) &&
		pgss_write_chunk_s(statfile, &qtext_len);

	if (ok && qtext_len >= 0)
		ok = pgss_write_chunk(statfile,
							  dsa_get_address(pgss_query_text_dsa,
											  shpgss->entry.query_text),
							  qtext_len + 1);

	LWLockRelease(&shpgss->header.lock);

	return ok;
}

/*
 * Restore what pgss_to_serialized_data() wrote.  Returning false makes core
 * discard the whole stats file, so treat anything implausible as corruption.
 */
static bool
pgss_from_serialized_data(const PgStat_HashKey *key,
						  PgStatShared_Common *header,
						  FILE *statfile)
{
	PgStatShared_Pgss *shpgss = (PgStatShared_Pgss *) header;
	bool		saved_entry;
	dsa_pointer dp = InvalidDsaPointer;
	int			qtext_len;
	uint64		entry_dealloc;
	uint64		stats_reset;
	uint64		usage;

	if (!pgss_read_chunk_s(statfile, &saved_entry))
		return false;

	if (!saved_entry)
	{
		pgstat_drop_entry(PGSTAT_KIND_PGSS, key->dboid, key->objid, false);
		return true;
	}

	pgss_have_restored_entries = true;

	if (!pgss_read_chunk_s(statfile, &shpgss->entry.key) ||
		!pgss_read_chunk_s(statfile, &shpgss->entry.encoding) ||
		!pgss_read_chunk_s(statfile, &shpgss->entry.stats_since) ||
		!pgss_read_chunk_s(statfile, &shpgss->entry.minmax_stats_since) ||
		!pgss_read_chunk_s(statfile, &entry_dealloc) ||
		!pgss_read_chunk_s(statfile, &stats_reset) ||
		!pgss_read_chunk_s(statfile, &usage) ||
		!pgss_read_chunk_s(statfile, &qtext_len))
		return false;

	/* Sanity-check what we can; the rest is opaque. */
	if (!PG_VALID_BE_ENCODING(shpgss->entry.encoding) ||
		(shpgss->entry.key.toplevel != false && shpgss->entry.key.toplevel != true) ||
		qtext_len < -1 || qtext_len >= MaxAllocSize)
		return false;

	if (qtext_len >= 0)
	{
		char	   *qtext;

		qtext = palloc_extended((Size) qtext_len + 1, MCXT_ALLOC_NO_OOM);
		if (qtext == NULL)
		{
			if (fseek(statfile, qtext_len + 1, SEEK_CUR) != 0)
				return false;
		}
		else
		{
			if (!pgss_read_chunk(statfile, qtext, qtext_len + 1))
			{
				pfree(qtext);
				return false;
			}
			dp = pgss_store_query_text_dsa(qtext, qtext_len);
			pfree(qtext);
		}
	}

	/* Nobody else can see the entry yet, so plain initialization is fine. */
	pg_atomic_init_u32(&shpgss->sticky,
					   (shpgss->entry.counters.calls[PGSS_PLAN] +
						shpgss->entry.counters.calls[PGSS_EXEC]) == 0 ?
					   PGSS_STICKY_PENDING : PGSS_STICKY_NONE);
	pg_atomic_init_u64(&shpgss->usage, usage);
	pg_atomic_init_u32(&shpgss->entry_live, 0);
	if (DsaPointerIsValid(dp))
	{
		if (pgss_remember_restored_query_text(dp, qtext_len))
		{
			shpgss->entry.query_text = dp;
			shpgss->entry.query_len = qtext_len;
		}
		else
			pgss_free_dsa_query_text(dp, qtext_len);
	}
	pgss_mark_entry_live(shpgss);
	pg_atomic_write_u64(&pgss_shared->entry_dealloc, entry_dealloc);
	pg_atomic_write_u64(&pgss_shared->stats_reset, stats_reset);

	return true;
}

static void
pgss_stats_file_finish(PgStat_StatsFileOp status)
{
	if (status == STATS_DISCARD && pgss_shared != NULL)
	{
		for (int i = 0; i < pgss_num_restored_query_texts; i++)
			pgss_free_dsa_query_text(pgss_restored_query_texts[i].text,
									 pgss_restored_query_texts[i].len);

		pg_atomic_write_u64(&pgss_shared->entry_dealloc, 0);
		pg_atomic_write_u64(&pgss_shared->entry_count, 0);
		pg_atomic_write_u64(&pgss_shared->stats_reset,
							(uint64) GetCurrentTimestamp());
		pg_atomic_write_u64(&pgss_shared->cur_median_usage,
							PGSS_USAGE_MEDIAN_INIT);
	}

	if (pgss_restored_query_texts != NULL)
		pfree(pgss_restored_query_texts);
	pgss_restored_query_texts = NULL;
	pgss_num_restored_query_texts = 0;
	pgss_restored_query_text_capacity = 0;
	pgss_have_restored_entries = false;
}

static void
pgss_accumulate_pending(PgStat_PgssPending *pending, pgssStoreKind kind,
						double total_time, uint64 rows,
						const BufferUsage *bufusage,
						const WalUsage *walusage,
						const JitInstrumentation *jitusage,
						int parallel_workers_to_launch,
						int parallel_workers_launched,
						PlannedStmtOrigin planOrigin)
{
	Assert(kind == PGSS_PLAN || kind == PGSS_EXEC);

	/*
	 * This timestamp decides whether the batch's min/max values survive a
	 * concurrent min/max-only reset.  We set it once per pending batch, so a
	 * batch that spans such a reset may omit post-reset min/max observations
	 * until a later batch establishes new values.  Avoiding that would
	 * require consulting shared reset state while accumulating every
	 * statement.
	 */
	if (pending->counters.calls[kind] == 0)
		pending->minmax_since[kind] = GetCurrentTimestamp();

	pending->counters.calls[kind]++;
	pending->counters.total_time[kind] += total_time;

	if (pending->counters.calls[kind] == 1)
	{
		pending->counters.min_time[kind] = total_time;
		pending->counters.max_time[kind] = total_time;
		pending->counters.mean_time[kind] = total_time;
	}
	else
	{
		/*
		 * Welford's online algorithm for accumulating mean and sum of squared
		 * deviations. See <http://www.johndcook.com/blog/standard_deviation/>
		 */
		double		old_mean = pending->counters.mean_time[kind];

		pending->counters.mean_time[kind] +=
			(total_time - old_mean) / pending->counters.calls[kind];
		pending->counters.sum_var_time[kind] +=
			(total_time - old_mean) * (total_time - pending->counters.mean_time[kind]);

		if (pending->counters.min_time[kind] > total_time)
			pending->counters.min_time[kind] = total_time;
		if (pending->counters.max_time[kind] < total_time)
			pending->counters.max_time[kind] = total_time;
	}

	pending->counters.rows += rows;

#define PGSS_ACCUM_BUFUSAGE(item)		\
	pending->counters.item += bufusage->item
#define PGSS_ACCUM_BUFUSAGE_TIME(item)	\
	pending->counters.item += INSTR_TIME_GET_MILLISEC(bufusage->item)

	PGSS_ACCUM_BUFUSAGE(shared_blks_hit);
	PGSS_ACCUM_BUFUSAGE(shared_blks_read);
	PGSS_ACCUM_BUFUSAGE(shared_blks_dirtied);
	PGSS_ACCUM_BUFUSAGE(shared_blks_written);
	PGSS_ACCUM_BUFUSAGE(local_blks_hit);
	PGSS_ACCUM_BUFUSAGE(local_blks_read);
	PGSS_ACCUM_BUFUSAGE(local_blks_dirtied);
	PGSS_ACCUM_BUFUSAGE(local_blks_written);
	PGSS_ACCUM_BUFUSAGE(temp_blks_read);
	PGSS_ACCUM_BUFUSAGE(temp_blks_written);
	PGSS_ACCUM_BUFUSAGE_TIME(shared_blk_read_time);
	PGSS_ACCUM_BUFUSAGE_TIME(shared_blk_write_time);
	PGSS_ACCUM_BUFUSAGE_TIME(local_blk_read_time);
	PGSS_ACCUM_BUFUSAGE_TIME(local_blk_write_time);
	PGSS_ACCUM_BUFUSAGE_TIME(temp_blk_read_time);
	PGSS_ACCUM_BUFUSAGE_TIME(temp_blk_write_time);
#undef PGSS_ACCUM_BUFUSAGE_TIME
#undef PGSS_ACCUM_BUFUSAGE

	pending->counters.wal_records += walusage->wal_records;
	pending->counters.wal_fpi += walusage->wal_fpi;
	pending->counters.wal_bytes += walusage->wal_bytes;
	pending->counters.wal_buffers_full += walusage->wal_buffers_full;

	if (jitusage)
	{
		pending->counters.jit_functions += jitusage->created_functions;
		pending->counters.jit_generation_time += INSTR_TIME_GET_MILLISEC(jitusage->generation_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->deform_counter))
			pending->counters.jit_deform_count++;
		pending->counters.jit_deform_time += INSTR_TIME_GET_MILLISEC(jitusage->deform_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter))
			pending->counters.jit_inlining_count++;
		pending->counters.jit_inlining_time += INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter))
			pending->counters.jit_optimization_count++;
		pending->counters.jit_optimization_time += INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->emission_counter))
			pending->counters.jit_emission_count++;
		pending->counters.jit_emission_time += INSTR_TIME_GET_MILLISEC(jitusage->emission_counter);
	}

	pending->counters.parallel_workers_to_launch += parallel_workers_to_launch;
	pending->counters.parallel_workers_launched += parallel_workers_launched;

	if (planOrigin == PLAN_STMT_CACHE_GENERIC)
		pending->counters.generic_plan_calls++;
	else if (planOrigin == PLAN_STMT_CACHE_CUSTOM)
		pending->counters.custom_plan_calls++;
}

void
pgss_entry_accumulate(PgStat_EntryRef *entry_ref, pgssStoreKind kind,
					  double total_time, uint64 rows,
					  const BufferUsage *bufusage,
					  const WalUsage *walusage,
					  const JitInstrumentation *jitusage,
					  int parallel_workers_to_launch,
					  int parallel_workers_launched,
					  PlannedStmtOrigin planOrigin)
{
	PgStatShared_Pgss *shared;

	pgstat_prep_pending_from_entry_ref(entry_ref);

	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;
	pgss_accumulate_pending((PgStat_PgssPending *) entry_ref->pending,
							kind, total_time, rows, bufusage,
							walusage, jitusage,
							parallel_workers_to_launch,
							parallel_workers_launched, planOrigin);

	/* "Unstick" entry if it was previously sticky. */
	for (;;)
	{
		uint32		sticky;

		sticky = pg_atomic_read_u32(&shared->sticky);
		if (sticky == PGSS_STICKY_NONE)
			break;

		if (sticky == PGSS_STICKY_PENDING)
		{
			uint32		expected = PGSS_STICKY_PENDING;

			if (pg_atomic_compare_exchange_u32(&shared->sticky, &expected,
											   PGSS_STICKY_UNSTICKING))
			{
				pg_atomic_write_u64(&shared->usage, PGSS_USAGE_ONE);
				pg_atomic_write_u32(&shared->sticky, PGSS_STICKY_NONE);
				break;
			}
		}
		else
		{
			SpinDelayStatus delayStatus;

			init_local_spin_delay(&delayStatus);
			while (pg_atomic_read_u32(&shared->sticky) ==
				   PGSS_STICKY_UNSTICKING)
				perform_spin_delay(&delayStatus);
			finish_spin_delay(&delayStatus);
		}
	}

	pg_atomic_fetch_add_u64(&shared->usage, PGSS_USAGE_ONE);
}

/*
 * Does the entry match a pgssResetFilter?  Zero fields match anything.
 */
static bool
pgss_entry_matches_filter(PgStatShared_Pgss *shared, pgssResetFilter *filter)
{
	return (!filter->userid || shared->entry.key.userid == filter->userid) &&
		(!filter->dbid || shared->entry.key.dbid == filter->dbid) &&
		(!filter->queryid || shared->entry.key.queryid == filter->queryid);
}

static bool
pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssResetFilter *filter = (pgssResetFilter *) DatumGetPointer(match_data);
	PgStatShared_Pgss *shared;
	bool		matches;

	shared = pgss_shared_from_hash_entry(p);

	if (pg_atomic_read_u32(&shared->entry_live) == 0)
		return false;

	LWLockAcquire(&shared->header.lock, LW_SHARED);
	matches = pg_atomic_read_u32(&shared->entry_live) != 0 &&
		pgss_entry_matches_filter(shared, filter);
	LWLockRelease(&shared->header.lock);

	return matches;
}

static bool
pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssResetFilter *filter = (pgssResetFilter *) DatumGetPointer(match_data);
	PgStatShared_Pgss *shared = pgss_shared_from_hash_entry(p);
	bool		dropped = false;

	LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
	if (pg_atomic_read_u32(&shared->entry_live) != 0 &&
		pgss_entry_matches_filter(shared, filter))
	{
		dropped = pgss_mark_entry_dead(shared);
		if (dropped)
			pgss_clear_query_text_locked(shared);
	}
	LWLockRelease(&shared->header.lock);

	return dropped;
}

/*
 * Reset entries corresponding to parameters passed.
 */
TimestampTz
pgss_entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only)
{
	TimestampTz stats_reset = GetCurrentTimestamp();
	pgssResetFilter filter = {userid, dbid, queryid};

	/*
	 * A min/max-only reset is partial, so it updates matching entries in
	 * place rather than dropping them.  Flush our own pending stats first so
	 * they are covered by the reset.
	 */
	if (minmax_only)
	{
		dshash_seq_status hstat;
		PgStatShared_HashEntry *p;

		pgstat_report_stat(true);

		dshash_seq_init(&hstat, pgss_hash, false);
		while ((p = dshash_seq_next(&hstat)) != NULL)
		{
			PgStatShared_Pgss *shared;

			if (p->dropped)
				continue;
			if (!pgss_match_entry(p, PointerGetDatum(&filter)))
				continue;

			shared = pgss_shared_from_hash_entry(p);

			LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
			if (pg_atomic_read_u32(&shared->entry_live) != 0 &&
				pgss_entry_matches_filter(shared, &filter))
			{
				shared->entry.minmax_stats_since = stats_reset;
				for (int kind = 0; kind < PGSS_NUMKIND; kind++)
				{
					shared->entry.counters.min_time[kind] = 0;
					shared->entry.counters.max_time[kind] = 0;
				}
			}
			LWLockRelease(&shared->header.lock);
		}
		dshash_seq_term(&hstat);

		return stats_reset;
	}

	pgstat_drop_matching_entries_of_kind(PGSTAT_KIND_PGSS,
										 pgss_drop_matching_entry,
										 PointerGetDatum(&filter));

	/* Reset global statistics too when all entries were targeted */
	if (!userid && !dbid && !queryid)
	{
		pg_atomic_write_u64(&pgss_shared->entry_dealloc, 0);
		pg_atomic_write_u64(&pgss_shared->stats_reset, (uint64) stats_reset);
	}

	return stats_reset;
}

static int
pgss_evict_candidate_objid_cmp(const void *a, const void *b)
{
	const pgssEvictCandidate *left = (const pgssEvictCandidate *) a;
	const pgssEvictCandidate *right = (const pgssEvictCandidate *) b;

	if (left->objid < right->objid)
		return -1;
	if (left->objid > right->objid)
		return 1;
	return 0;
}

static int
pgss_evict_candidate_usage_cmp(const void *a, const void *b)
{
	const pgssEvictCandidate *left = (const pgssEvictCandidate *) a;
	const pgssEvictCandidate *right = (const pgssEvictCandidate *) b;

	/* Text-less entries have no displayable value, so evict them first. */
	if (left->has_query_text != right->has_query_text)
		return left->has_query_text ? 1 : -1;
	if (left->usage < right->usage)
		return -1;
	if (left->usage > right->usage)
		return 1;
	return pgss_evict_candidate_objid_cmp(a, b);
}

static const pgssEvictCandidate *
pgss_find_evict_candidate(pgssEvictContext *context, uint64 objid)
{
	pgssEvictCandidate key;

	key.objid = objid;

	return bsearch(&key, context->candidates, context->target,
				   sizeof(pgssEvictCandidate),
				   pgss_evict_candidate_objid_cmp);
}

static bool
pgss_drop_evicted_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssEvictContext *context = (pgssEvictContext *) DatumGetPointer(match_data);
	const pgssEvictCandidate *candidate;
	PgStatShared_Pgss *shared;
	bool		dropped = false;

	candidate = pgss_find_evict_candidate(context, p->key.objid);
	if (candidate == NULL)
		return false;

	if (pg_atomic_read_u32(&p->generation) != candidate->generation)
		return false;

	shared = pgss_shared_from_hash_entry(p);
	LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
	if (pg_atomic_read_u32(&p->generation) == candidate->generation &&
		pg_atomic_read_u32(&shared->entry_live) != 0 &&
		memcmp(&shared->entry.key, &candidate->key, sizeof(pgssHashKey)) == 0)
	{
		dropped = pgss_mark_entry_dead(shared);
		if (dropped)
			pgss_clear_query_text_locked(shared);
	}
	LWLockRelease(&shared->header.lock);

	return dropped;
}

/*
 * Decay the usage of all live entries, then drop the least-used ones.
 *
 * The number of victims is what it takes to bring retained query texts and
 * live entries back to their targets, capped at PGSS_EVICT_PASS_PERCENT of
 * the live entries, with a small floor, so that one backend never has to pay
 * for a huge pass (e.g. after pgss_max was lowered); further misses will
 * finish the job.
 */
void
pgss_evict_for_entry(int needed_query_len)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *p;
	pgssEvictCandidate *candidates = NULL;
	pgssEvictContext context;
	uint64		entry_victims;
	uint64		freed_query_text = 0;
	uint64		live_entry_count;
	uint64		max_candidates;
	uint64		qtext_bytes_to_free = 0;
	uint64		qtext_bytes_needed = 0;
	uint64		qtext_count;
	uint64		qtext_victims;
	uint64		qtext_memory;
	uint64		qtext_total_len;
	uint64		threshold;
	uint64		victim_count;
	bool		query_text_fits = true;
	int			byte_victims = 0;
	int			first_text;
	int			capacity;
	int			target;
	int			ncandidates = 0;

	/* Let one backend handle victim selection while others keep going. */
	if (!LWLockConditionalAcquire(&pgss_shared->entry_evict_lock,
								  LW_EXCLUSIVE))
		return;

	threshold = pgss_evict_threshold();
	qtext_memory = pgss_query_text_memory_limit_bytes();
	if (needed_query_len > 0)
		qtext_bytes_needed = (uint64) needed_query_len;
	query_text_fits = qtext_bytes_needed <= qtext_memory;
	qtext_count = pg_atomic_read_u64(&pgss_shared->qtext_count);
	qtext_total_len = pg_atomic_read_u64(&pgss_shared->qtext_total_len);
	live_entry_count = pg_atomic_read_u64(&pgss_shared->entry_count);
	if (qtext_count < threshold && live_entry_count < threshold &&
		qtext_total_len < qtext_memory &&
		(!query_text_fits ||
		 qtext_bytes_needed <= qtext_memory - qtext_total_len))
		goto done;
	if (live_entry_count == 0)
		goto done;

	max_candidates = (uint64) Min(MaxAllocSize / sizeof(pgssEvictCandidate),
								  INT_MAX);
	if (live_entry_count >= max_candidates)
		capacity = (int) max_candidates;
	else
		capacity = (int) Min(live_entry_count + PGSS_ENTRY_SCAN_HEADROOM,
							 max_candidates);
	candidates = palloc_extended((Size) capacity * sizeof(pgssEvictCandidate),
								 MCXT_ALLOC_NO_OOM);
	if (!candidates)
		goto done;

	dshash_seq_init(&hstat, pgss_hash, false);
	while ((p = dshash_seq_next(&hstat)) != NULL)
	{
		PgStatShared_Pgss *shared;

		if (p->dropped)
			continue;

		shared = pgss_shared_from_hash_entry(p);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
			continue;

		/* Entries can be admitted while we scan */
		if (ncandidates >= capacity)
		{
			pgssEvictCandidate *new_candidates;
			int			new_capacity;

			if ((uint64) capacity >= max_candidates)
				break;

			new_capacity = (int) Min((uint64) capacity * 2,
									 max_candidates);
			new_candidates =
				repalloc_extended(candidates,
								  (Size) new_capacity * sizeof(pgssEvictCandidate),
								  MCXT_ALLOC_NO_OOM);
			if (!new_candidates)
				break;

			candidates = new_candidates;
			capacity = new_capacity;
		}

		LWLockAcquire(&shared->header.lock, LW_SHARED);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
		{
			LWLockRelease(&shared->header.lock);
			continue;
		}

		candidates[ncandidates].objid = p->key.objid;
		candidates[ncandidates].generation =
			pg_atomic_read_u32(&p->generation);
		candidates[ncandidates].key = shared->entry.key;
		candidates[ncandidates].has_query_text =
			DsaPointerIsValid(shared->entry.query_text);
		candidates[ncandidates].query_len =
			candidates[ncandidates].has_query_text ? shared->entry.query_len : 0;
		LWLockRelease(&shared->header.lock);

		candidates[ncandidates].usage =
			candidates[ncandidates].has_query_text ?
			pgss_decay_usage(shared) : 0;
		ncandidates++;
	}
	dshash_seq_term(&hstat);

	if (ncandidates == 0)
		goto done;

	/* Sort into increasing order by usage, and record the median */
	qsort(candidates, ncandidates, sizeof(pgssEvictCandidate),
		  pgss_evict_candidate_usage_cmp);
	for (first_text = 0; first_text < ncandidates; first_text++)
		if (candidates[first_text].has_query_text)
			break;
	pg_atomic_write_u64(&pgss_shared->cur_median_usage,
						first_text < ncandidates ?
						candidates[first_text +
								   (ncandidates - first_text) / 2].usage :
						PGSS_USAGE_MEDIAN_INIT);

	entry_victims = live_entry_count > (uint64) pgss_max ?
		live_entry_count - (uint64) pgss_max : 0;
	qtext_victims = qtext_count > (uint64) pgss_max ?
		qtext_count - (uint64) pgss_max : 0;
	victim_count = Max(qtext_victims, entry_victims);

	if (query_text_fits &&
		(qtext_total_len >= qtext_memory ||
		 qtext_bytes_needed > qtext_memory - qtext_total_len))
	{
		if (qtext_total_len >= qtext_memory)
			qtext_bytes_to_free =
				qtext_total_len - qtext_memory + qtext_bytes_needed;
		else
			qtext_bytes_to_free =
				qtext_bytes_needed - (qtext_memory - qtext_total_len);
		for (byte_victims = 0; byte_victims < ncandidates; byte_victims++)
		{
			if (candidates[byte_victims].has_query_text &&
				candidates[byte_victims].query_len > 0)
				freed_query_text += candidates[byte_victims].query_len;
			if (freed_query_text >= qtext_bytes_to_free)
			{
				byte_victims++;
				break;
			}
		}
		victim_count = Max(victim_count, (uint64) byte_victims);
	}

	target = (int) Min(victim_count, (uint64) ncandidates);
	target = Min(target, Max(10, ncandidates * PGSS_EVICT_PASS_PERCENT / 100));
	if (qtext_bytes_to_free > 0)
		target = Max(target, Min(byte_victims, ncandidates));
	if (target == 0)
		goto done;

	/* Sort the victims by objid for pgss_find_evict_candidate() */
	qsort(candidates, target, sizeof(pgssEvictCandidate),
		  pgss_evict_candidate_objid_cmp);

	context.candidates = candidates;
	context.target = target;
	pgstat_drop_matching_entries_of_kind(PGSTAT_KIND_PGSS,
										 pgss_drop_evicted_entry,
										 PointerGetDatum(&context));

	pg_atomic_fetch_add_u64(&pgss_shared->entry_dealloc, 1);

	/* Give freed query text memory back to the OS when possible */
	dsa_trim(pgss_query_text_dsa);

done:
	if (candidates)
		pfree(candidates);
	LWLockRelease(&pgss_shared->entry_evict_lock);
}
