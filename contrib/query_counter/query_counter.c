/*
 * query_counter.c
 */
#include <math.h>

#include "postgres.h"

#include "access/htup_details.h"
#include "common/hashfn.h"
#include "funcapi.h"
#include "jit/jit.h"
#include "parser/analyze.h"
#include "storage/dsm_registry.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/pgstat_internal.h"

PG_MODULE_MAGIC_EXT(
					.name = "query_counter",
					.version = PG_VERSION);

#define PGSTAT_KIND_PGSS 28

/* GUCs */
static int	pgss_max = 5000;	/* max # statements to track */

#define USAGE_EXEC(duration) (1.0)
#define USAGE_DECREASE_FACTOR (0.99)	/* decreased every entry_dealloc */
#define STICKY_DECREASE_FACTOR (0.50)	/* factor for sticky entries */
#define USAGE_DEALLOC_PERCENT 5 /* free this % of entries at once */
#define IS_STICKY(c) (c == 0)

#define PGSTAT_PGSS_IDX(query_id, user_id, toplevel) hash_combine64(toplevel, hash_combine64(query_id, user_id))

typedef enum pgssStoreKind
{
	PGSS_INVALID = -1,

	/*
	 * PGSS_PLAN and PGSS_EXEC must be respectively 0 and 1 as they're used to
	 * reference the underlying values in the arrays in the Counters struct,
	 * and this order is required in query_counter_internal().
	 */
	PGSS_PLAN = 0,
	PGSS_EXEC_START,
	PGSS_EXEC_END,
} pgssStoreKind;

#define PGSS_NUMKIND (PGSS_EXEC_END + 1)

/* Previous hooks */
static post_parse_analyze_hook_type prev_post_parse_analyze_hook = NULL;
static ExecutorStart_hook_type prev_ExecutorStart = NULL;
static ExecutorRun_hook_type prev_ExecutorRun = NULL;
static ExecutorFinish_hook_type prev_ExecutorFinish = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd = NULL;

typedef struct PgssMatchData
{
	Oid			dbid;
	uint64		objid;
}			PgssMatchData;

/* Shared memory structures */
typedef struct pgssSharedState
{
	void	   *raw_dsa_area;
	dshash_table_handle hash_handle;
	LWLock	   *lock;
} pgssSharedState;

typedef struct pgssHashKey
{
	int64		queryId;
	Oid			dbid;
	Oid			userid;
	bool		toplevel;
	uint64		objid;
} pgssHashKey;

/* Structures for statistics of pgss */
typedef struct Counters
{
	PgStat_Counter calls[PGSS_NUMKIND]; /* # of times planned/executed */
	double		total_time[PGSS_NUMKIND];	/* total planning/execution time,
											 * in msec */
	double		min_time[PGSS_NUMKIND]; /* minimum planning/execution time in
										 * msec since min/max reset */
	double		max_time[PGSS_NUMKIND]; /* maximum planning/execution time in
										 * msec since min/max reset */
	double		mean_time[PGSS_NUMKIND];	/* mean planning/execution time in
											 * msec */
	double		sum_var_time[PGSS_NUMKIND]; /* sum of variances in
											 * planning/execution time in msec */
	PgStat_Counter rows;		/* total # of retrieved or affected rows */
	PgStat_Counter shared_blks_hit; /* # of shared buffer hits */
	PgStat_Counter shared_blks_read;	/* # of shared disk blocks read */
	PgStat_Counter shared_blks_dirtied; /* # of shared disk blocks dirtied */
	PgStat_Counter shared_blks_written; /* # of shared disk blocks written */
	PgStat_Counter local_blks_hit;	/* # of local buffer hits */
	PgStat_Counter local_blks_read; /* # of local disk blocks read */
	PgStat_Counter local_blks_dirtied;	/* # of local disk blocks dirtied */
	PgStat_Counter local_blks_written;	/* # of local disk blocks written */
	PgStat_Counter temp_blks_read;	/* # of temp blocks read */
	PgStat_Counter temp_blks_written;	/* # of temp blocks written */
	double		shared_blk_read_time;	/* time spent reading shared blocks,
										 * in msec */
	double		shared_blk_write_time;	/* time spent writing shared blocks,
										 * in msec */
	double		local_blk_read_time;	/* time spent reading local blocks, in
										 * msec */
	double		local_blk_write_time;	/* time spent writing local blocks, in
										 * msec */
	double		temp_blk_read_time; /* time spent reading temp blocks, in msec */
	double		temp_blk_write_time;	/* time spent writing temp blocks, in
										 * msec */
	double		usage;			/* usage factor */
	PgStat_Counter wal_records; /* # of WAL records generated */
	PgStat_Counter wal_fpi;		/* # of WAL full page images generated */
	uint64		wal_bytes;		/* total amount of WAL generated in bytes */
	PgStat_Counter wal_buffers_full;	/* # of times the WAL buffers became
										 * full */
	PgStat_Counter jit_functions;	/* total number of JIT functions emitted */
	double		jit_generation_time;	/* total time to generate jit code */
	PgStat_Counter jit_inlining_count;	/* number of times inlining time has
										 * been > 0 */
	double		jit_deform_time;	/* total time to deform tuples in jit code */
	PgStat_Counter jit_deform_count;	/* number of times deform time has
										 * been > 0 */

	double		jit_inlining_time;	/* total time to inline jit code */
	PgStat_Counter jit_optimization_count;	/* number of times optimization
											 * time has been > 0 */
	double		jit_optimization_time;	/* total time to optimize jit code */
	PgStat_Counter jit_emission_count;	/* number of times emission time has
										 * been > 0 */
	double		jit_emission_time;	/* total time to emit jit code */
	PgStat_Counter parallel_workers_to_launch;	/* # of parallel workers
												 * planned to be launched */
	PgStat_Counter parallel_workers_launched;	/* # of parallel workers
												 * actually launched */
	PgStat_Counter generic_plan_calls;	/* number of calls using a generic
										 * plan */
	PgStat_Counter custom_plan_calls;	/* number of calls using a custom plan */
} Counters;

typedef struct PgStatShared_pgss
{
	PgStatShared_Common header;
	Counters	stats;
	int64		queryId;
	dsa_pointer query_text;
}			PgStatShared_pgss;

static bool pgss_flush_cb(PgStat_EntryRef *entry_ref, bool nowait);
void		pgss_serialize_extra(const PgStat_HashKey *key,
								 const PgStatShared_Common *header,
								 FILE *fd);
bool		pgss_deserialize_extra(PgStat_HashKey *key,
								   const PgStatShared_Common *header,
								   FILE *fd);

static const PgStat_KindInfo pgss_stats = {
	.name = "query_counter",
	.fixed_amount = false,
	.write_to_file = true,
	.accessed_across_databases = true,
	.shared_size = sizeof(PgStatShared_pgss),
	.shared_data_off = offsetof(PgStatShared_pgss, stats),
	.shared_data_len = sizeof(((PgStatShared_pgss *) 0)->stats) + sizeof(int64) + sizeof(dsa_pointer),
	.pending_size = sizeof(Counters),
	.flush_pending_cb = pgss_flush_cb,
	.track_entry_count = true,
	.to_serialized_extra = pgss_serialize_extra,
	.from_serialized_extra = pgss_deserialize_extra,
};

/* Forward declarations */
static void pgss_post_parse_analyze(ParseState *pstate, Query *query, JumbleState *jstate);
static void pgss_ExecutorStart(QueryDesc *queryDesc, int eflags);
static void pgss_ExecutorRun(QueryDesc *queryDesc,
							 ScanDirection direction,
							 uint64 count);
static void pgss_ExecutorFinish(QueryDesc *queryDesc);
static void pgss_ExecutorEnd(QueryDesc *queryDesc);
static void pgss_store(const char *query, int64 queryId,
					   int query_location, int query_len,
					   pgssStoreKind kind,
					   double total_time, uint64 rows,
					   const BufferUsage *bufusage,
					   const WalUsage *walusage,
					   const struct JitInstrumentation *jitusage,
					   JumbleState *jstate,
					   int parallel_workers_to_launch,
					   int parallel_workers_launched,
					   PlannedStmtOrigin planOrigin);

/* static void pgss_gc(void); */
static bool pgss_flush_cb(PgStat_EntryRef *entry_ref, bool nowait);

/*---- Local variables ----*/

/* Current nesting depth of planner/ExecutorRun/ProcessUtility calls */
static int	nesting_level = 0;

#define pgss_enabled(level) \
	((level) == 0)

static dsa_area *pgss_query_dsa = NULL;
static void
pgss_attach_shmem(void)
{
	bool		found;

	if (pgss_query_dsa == NULL)
		pgss_query_dsa = GetNamedDSA("pgss query dsa", &found);
}

/* -------------------------------
 * Module load/init
 * ------------------------------- */
void
_PG_init(void)
{
	prev_post_parse_analyze_hook = post_parse_analyze_hook;
	post_parse_analyze_hook = pgss_post_parse_analyze;
	prev_ExecutorStart = ExecutorStart_hook;
	ExecutorStart_hook = pgss_ExecutorStart;
	prev_ExecutorRun = ExecutorRun_hook;
	ExecutorRun_hook = pgss_ExecutorRun;
	prev_ExecutorFinish = ExecutorFinish_hook;
	ExecutorFinish_hook = pgss_ExecutorFinish;
	prev_ExecutorEnd = ExecutorEnd_hook;
	ExecutorEnd_hook = pgss_ExecutorEnd;

	/*
	 * Inform the postmaster that we want to enable query_id calculation if
	 * compute_query_id is set to auto.
	 */
	EnableQueryId();

	/*
	 * Define (or redefine) custom GUC variables.
	 */
	DefineCustomIntVariable("pgss.max",
							"Sets the maximum number of statements tracked by pgss.",
							NULL,
							&pgss_max,
							5000,
							5,
							INT_MAX / 2,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
							NULL);

	/* register our custom stats kind */
	pgstat_register_kind(PGSTAT_KIND_PGSS, &pgss_stats);
}

/* -------------------------------
 * Core functional hooks
 * ------------------------------- */
static void
pgss_post_parse_analyze(ParseState *pstate, Query *query, JumbleState *jstate)
{
	if (prev_post_parse_analyze_hook)
		prev_post_parse_analyze_hook(pstate, query, jstate);

	if (jstate && jstate->clocations_count > 0)
		pgss_store(pstate->p_sourcetext,
				   query->queryId,
				   query->stmt_location,
				   query->stmt_len,
				   PGSS_INVALID,
				   0,
				   0,
				   NULL,
				   NULL,
				   NULL,
				   jstate,
				   0,
				   0,
				   PLAN_STMT_UNKNOWN);
}

static void
pgss_ExecutorStart(QueryDesc *queryDesc, int eflags)
{
	if (prev_ExecutorStart)
		prev_ExecutorStart(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	/*
	 * If query has queryId zero, don't track it.  This prevents double
	 * counting of optimizable statements that are directly contained in
	 * utility statements.
	 */
	if (pgss_enabled(nesting_level) && queryDesc->plannedstmt->queryId != INT64CONST(0))
	{
		/*
		 * Set up to track total elapsed time in ExecutorRun.  Make sure the
		 * space is allocated in the per-query context so it will go away at
		 * ExecutorEnd.
		 */
		if (queryDesc->totaltime == NULL)
		{
			MemoryContext oldcxt;

			oldcxt = MemoryContextSwitchTo(queryDesc->estate->es_query_cxt);
			queryDesc->totaltime = InstrAlloc(1, INSTRUMENT_ALL, false);
			MemoryContextSwitchTo(oldcxt);
		}
	}
}

static void
pgss_ExecutorRun(QueryDesc *queryDesc,
				 ScanDirection direction,
				 uint64 count)
{
	nesting_level++;
	PG_TRY();
	{
		if (prev_ExecutorRun)
			prev_ExecutorRun(queryDesc, direction, count);
		else
			standard_ExecutorRun(queryDesc, direction, count);
	}
	PG_FINALLY();
	{
		nesting_level--;
	}
	PG_END_TRY();
}

static void
pgss_ExecutorFinish(QueryDesc *queryDesc)
{
	nesting_level++;
	PG_TRY();
	{
		if (prev_ExecutorFinish)
			prev_ExecutorFinish(queryDesc);
		else
			standard_ExecutorFinish(queryDesc);
	}
	PG_FINALLY();
	{
		nesting_level--;
	}
	PG_END_TRY();
}

static void
pgss_ExecutorEnd(QueryDesc *queryDesc)
{
	int64		queryId = queryDesc->plannedstmt->queryId;

	if (queryId != INT64CONST(0) && queryDesc->totaltime &&
		pgss_enabled(nesting_level))
	{
		/*
		 * Make sure stats accumulation is done.  (Note: it's okay if several
		 * levels of hook all do this.)
		 */
		InstrEndLoop(queryDesc->totaltime);

		pgss_store(queryDesc->sourceText,
				   queryId,
				   queryDesc->plannedstmt->stmt_location,
				   queryDesc->plannedstmt->stmt_len,
				   PGSS_EXEC_END,
				   queryDesc->totaltime->total * 1000.0,	/* convert to msec */
				   queryDesc->estate->es_total_processed,
				   &queryDesc->totaltime->bufusage,
				   &queryDesc->totaltime->walusage,
				   queryDesc->estate->es_jit ? &queryDesc->estate->es_jit->instr : NULL,
				   NULL,
				   queryDesc->estate->es_parallel_workers_to_launch,
				   queryDesc->estate->es_parallel_workers_launched,
				   queryDesc->plannedstmt->planOrigin);
	}

	if (prev_ExecutorEnd)
		prev_ExecutorEnd(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

static void
pgss_store(const char *query, int64 queryId,
		   int query_location, int query_len,
		   pgssStoreKind kind,
		   double total_time, uint64 rows,
		   const BufferUsage *bufusage,
		   const WalUsage *walusage,
		   const struct JitInstrumentation *jitusage,
		   JumbleState *jstate,
		   int parallel_workers_to_launch,
		   int parallel_workers_launched,
		   PlannedStmtOrigin planOrigin)
{
	pgssHashKey key;

	if (!query)
		return;

	memset(&key, 0, sizeof(pgssHashKey));

	key.queryId = queryId;
	key.userid = GetUserId();
	key.dbid = MyDatabaseId;
	key.toplevel = (nesting_level == 0);
	key.objid = PGSTAT_PGSS_IDX(key.queryId, key.userid, key.toplevel);

	if (!jstate)
	{
		PgStat_EntryRef *entry_ref;
		Counters   *pending;
		bool		created_entry;

		Assert(kind == PGSS_PLAN || kind == PGSS_EXEC_START || kind == PGSS_EXEC_END);

		entry_ref = pgstat_prep_pending_entry(PGSTAT_KIND_PGSS,
											  key.dbid,
											  key.objid,
											  &created_entry);
		if (entry_ref == NULL)
			return;

		if (created_entry)
		{
			size_t		qlen;
			char	   *shared_query = NULL;
			dsa_pointer dp = InvalidDsaPointer;
			PgStatShared_pgss *shent = (PgStatShared_pgss *) entry_ref->shared_stats;

			if (!query)
				goto update_counters;

			/* Attach shared memory segment for pgss */
			pgss_attach_shmem();

			/* Lock the entry before modifying shared stats */
			pgstat_lock_entry(entry_ref, true);

			shent->queryId = queryId;

			/* Compute query length safely */
			qlen = strlen(query) + 1;

			/* Allocate memory in the DSA segment */
			if (pgss_query_dsa)
				dp = dsa_allocate(pgss_query_dsa, qlen);

			if (DsaPointerIsValid(dp))
			{
				shared_query = dsa_get_address(pgss_query_dsa, dp);

				/* Copy query safely, ensuring null-termination */
				memcpy(shared_query, query, qlen);
				shared_query[qlen - 1] = '\0';

				shent->query_text = dp;
			}

			/* Unlock the entry */
			pgstat_unlock_entry(entry_ref);
		}
update_counters:
		pending = (Counters *) entry_ref->pending;

		/* Increment the appropriate counter */
		pending->calls[kind] += 1;
		pending->calls[kind] += total_time;

		if (pending->calls[kind] == 1)
		{
			pending->min_time[kind] = total_time;
			pending->max_time[kind] = total_time;
			pending->mean_time[kind] = total_time;
		}
		else
		{
			/*
			 * Welford's method for accurately computing variance. See
			 * <http://www.johndcook.com/blog/standard_deviation/>
			 */
			double		old_mean = pending->mean_time[kind];

			pending->mean_time[kind] +=
				(total_time - old_mean) / pending->calls[kind];
			pending->sum_var_time[kind] +=
				(total_time - old_mean) * (total_time - pending->mean_time[kind]);

			/*
			 * Calculate min and max time. min = 0 and max = 0 means that the
			 * min/max statistics were reset
			 */
			if (pending->min_time[kind] == 0 && pending->max_time[kind] == 0)
			{
				pending->min_time[kind] = total_time;
				pending->max_time[kind] = total_time;
			}
			else
			{
				if (pending->min_time[kind] > total_time)
					pending->min_time[kind] = total_time;
				if (pending->max_time[kind] < total_time)
					pending->max_time[kind] = total_time;
			}
		}
		pending->rows += rows;
		pending->shared_blks_hit += bufusage->shared_blks_hit;
		pending->shared_blks_read += bufusage->shared_blks_read;
		pending->shared_blks_dirtied += bufusage->shared_blks_dirtied;
		pending->shared_blks_written += bufusage->shared_blks_written;
		pending->local_blks_hit += bufusage->local_blks_hit;
		pending->local_blks_read += bufusage->local_blks_read;
		pending->local_blks_dirtied += bufusage->local_blks_dirtied;
		pending->local_blks_written += bufusage->local_blks_written;
		pending->temp_blks_read += bufusage->temp_blks_read;
		pending->temp_blks_written += bufusage->temp_blks_written;
		pending->shared_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_read_time);
		pending->shared_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_write_time);
		pending->local_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_read_time);
		pending->local_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_write_time);
		pending->temp_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_read_time);
		pending->temp_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_write_time);
		pending->usage += USAGE_EXEC(total_time);
		pending->wal_records += walusage->wal_records;
		pending->wal_fpi += walusage->wal_fpi;
		pending->wal_bytes += walusage->wal_bytes;
		pending->wal_buffers_full += walusage->wal_buffers_full;
		if (jitusage)
		{
			pending->jit_functions += jitusage->created_functions;
			pending->jit_generation_time += INSTR_TIME_GET_MILLISEC(jitusage->generation_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->deform_counter))
				pending->jit_deform_count++;
			pending->jit_deform_time += INSTR_TIME_GET_MILLISEC(jitusage->deform_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter))
				pending->jit_inlining_count++;
			pending->jit_inlining_time += INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter))
				pending->jit_optimization_count++;
			pending->jit_optimization_time += INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter);

			if (INSTR_TIME_GET_MILLISEC(jitusage->emission_counter))
				pending->jit_emission_count++;
			pending->jit_emission_time += INSTR_TIME_GET_MILLISEC(jitusage->emission_counter);
		}

		/* parallel worker counters */
		pending->parallel_workers_to_launch += parallel_workers_to_launch;
		pending->parallel_workers_launched += parallel_workers_launched;

		/* plan cache counters */
		if (planOrigin == PLAN_STMT_CACHE_GENERIC)
			pending->generic_plan_calls++;
		else if (planOrigin == PLAN_STMT_CACHE_CUSTOM)
			pending->custom_plan_calls++;
	}
}

/*
 * Merge per-backend statistics (localent) into global statistics (shent)
 * for a specific statement kind.
 */
static void
merge_kind_stats(int kind,
				 PgStatShared_pgss * shent,
				 Counters *localent)
{
	int64		shared_calls = shent->stats.calls[kind];	/* global count before
															 * merge */
	int64		local_calls = localent->calls[kind];	/* backend count to
														 * merge */
	int64		n_total = shared_calls + local_calls;
	double		old_mean;
	double		delta;

	/* Nothing to merge */
	if (local_calls == 0)
		return;

	/* Update cumulative totals */
	shent->stats.calls[kind] += local_calls;
	shent->stats.total_time[kind] += localent->total_time[kind];

	/*
	 * Delta between local and global mean. Used to adjust mean and variance
	 * when merging two sets. Ensures accurate, numerically stable results
	 * even if means differ.
	 */
	old_mean = shent->stats.mean_time[kind];
	delta = localent->mean_time[kind] - old_mean;

	/* Update global mean using weighted delta (floating-point division!) */
	shent->stats.mean_time[kind] = old_mean + delta * ((double) local_calls / (double) n_total);

	/* Update sum of squared deviations (variance accumulator) */
	shent->stats.sum_var_time[kind] += localent->sum_var_time[kind] + delta * delta * shared_calls * local_calls / (double) n_total;

	/* Update global min/max */
	if (shared_calls == 0)
	{
		shent->stats.min_time[kind] = localent->min_time[kind];
		shent->stats.max_time[kind] = localent->max_time[kind];
	}
	else
	{
		shent->stats.min_time[kind] = Min(shent->stats.min_time[kind],
										  localent->min_time[kind]);
		shent->stats.max_time[kind] = Max(shent->stats.max_time[kind],
										  localent->max_time[kind]);
	}
}

static bool
pgss_flush_cb(PgStat_EntryRef *entry_ref, bool nowait)
{
	Counters   *localent;
	PgStatShared_pgss *shent;

	localent = (Counters *) entry_ref->pending;
	shent = (PgStatShared_pgss *) entry_ref->shared_stats;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return false;

	for (int kind = 0; kind < PGSS_NUMKIND; kind++)
		merge_kind_stats(kind, shent, localent);

	shent->stats.generic_plan_calls += localent->generic_plan_calls;
	shent->stats.custom_plan_calls += localent->custom_plan_calls;

	pgstat_unlock_entry(entry_ref);

	return true;
}

void
pgss_serialize_extra(const PgStat_HashKey *key,
					 const PgStatShared_Common *header,
					 FILE *fd)
{
	PgStatShared_pgss *entry = (PgStatShared_pgss *) header;

	pgss_attach_shmem();

	if (DsaPointerIsValid(entry->query_text))
	{
		dsa_pointer dp = entry->query_text;
		char	   *shared_query = dsa_get_address(pgss_query_dsa, dp);
		size_t		qlen = strlen(shared_query) + 1;

		shared_query[qlen - 1] = '\0';

		pgstat_write_chunk(fd, (void *) key, sizeof(PgStat_HashKey));	/* use parameter pointer
														 * directly */
		pgstat_write_chunk(fd, (void *) shared_query, qlen);
	}

	entry->query_text = InvalidDsaPointer;
}

bool
pgss_deserialize_extra(PgStat_HashKey *key,
					   const PgStatShared_Common *header,
					   FILE *fd)
{
	PgStatShared_pgss *entry;
	size_t		bufcap;
	size_t		len;
	char	   *buffer;
	int			c;
	dsa_pointer dp;
	char	   *dest;

	pgss_attach_shmem();

	entry = (PgStatShared_pgss *) header;

	/* Read the key */
	if (!pgstat_read_chunk(fd, (void *) key, sizeof(PgStat_HashKey)))
	{
		if (feof(fd))
			return true;		/* no more entries */
		elog(WARNING, "failed to read key from %s", "PGSS_DUMP_FILE");
		return false;
	}

	/* Read null-terminated query string */
	bufcap = 128;
	len = 0;
	buffer = palloc(bufcap);

	while ((c = fgetc(fd)) != EOF)
	{
		if (len + 1 >= bufcap)
		{
			bufcap *= 2;
			buffer = repalloc(buffer, bufcap);
		}
		buffer[len++] = (char) c;
		if (c == '\0')
			break;
	}

	if (c == EOF)
	{
		elog(WARNING, "unterminated query string in %s", "PGSS_DUMP_FILE");
		pfree(buffer);
		return false;
	}

	/* Allocate DSA memory */
	dp = dsa_allocate(pgss_query_dsa, len);
	dest = dsa_get_address(pgss_query_dsa, dp);
	memcpy(dest, buffer, len);
	pfree(buffer);

	/* Assign to entry */
	entry->query_text = dp;

	/* Keep fd open for next call! Do NOT fclose(fd) here */

	return true;				/* successfully read one entry */
}

/* -------------------------------
 * SQL-callable functions
 * ------------------------------- */
PG_FUNCTION_INFO_V1(query_counter);
Datum
query_counter(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	dshash_seq_status *hstat;
	PgStatShared_HashEntry *ps;
	PgStatShared_Common *shstats;
	Counters	entry;
	Datum		values[9];
	bool		nulls[9] = {false};
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();

		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(9);
		TupleDescInitEntry(tupdesc, 1, "userid", OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, 2, "dbid", OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, 3, "toplevel", BOOLOID, -1, 0);
		TupleDescInitEntry(tupdesc, 4, "queryid", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 5, "query_text", TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, 6, "calls_started", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 7, "calls_completed", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 8, "generic_plan_calls", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, 9, "custom_plan_calls", INT8OID, -1, 0);

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		hstat = MemoryContextAlloc(funcctx->multi_call_memory_ctx,
								   sizeof(dshash_seq_status));
		dshash_seq_init(hstat, pgStatLocal.shared_hash, false);
		funcctx->user_fctx = hstat;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	hstat = (dshash_seq_status *) funcctx->user_fctx;

	if (hstat == NULL)
		SRF_RETURN_DONE(funcctx);

	/* Find next PGSS entry */
	while ((ps = dshash_seq_next(hstat)) != NULL)
	{
		if (ps->key.kind == PGSTAT_KIND_PGSS || ps->dropped)
			break;
	}

	if (ps == NULL)
	{
		dshash_seq_term(hstat);
		funcctx->user_fctx = NULL;
		SRF_RETURN_DONE(funcctx);
	}

	shstats = (PgStatShared_Common *) dsa_get_address(pgStatLocal.dsa, ps->body);
	entry = ((PgStatShared_pgss *) shstats)->stats;

	/* At this point, ps is a PGSS entry */
	nulls[0] = true;
	values[1] = ObjectIdGetDatum(ps->key.dboid);
	nulls[2] = true;
	values[3] = Int64GetDatum(((PgStatShared_pgss *) shstats)->queryId);
	{
		dsa_pointer dp = ((PgStatShared_pgss *) shstats)->query_text;

		pgss_attach_shmem();

		if (DsaPointerIsValid(dp))
			values[4] = CStringGetTextDatum(dsa_get_address(pgss_query_dsa, dp));
		else
			nulls[4] = true;
	}
	values[5] = Int64GetDatum(entry.calls[PGSS_EXEC_START]);
	values[6] = Int64GetDatum(entry.calls[PGSS_EXEC_END]);
	values[7] = Int64GetDatum(entry.generic_plan_calls);
	values[8] = Int64GetDatum(entry.custom_plan_calls);

	tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
	SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
}
