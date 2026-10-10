/*-------------------------------------------------------------------------
 *
 * pg_stat_statements_internal.h
 *	  Shared declarations for pg_stat_statements implementation files.
 *
 * Copyright (c) 2008-2026, PostgreSQL Global Development Group
 *
 * IDENTIFICATION
 *	  contrib/pg_stat_statements/pg_stat_statements_internal.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef PG_STAT_STATEMENTS_INTERNAL_H
#define PG_STAT_STATEMENTS_INTERNAL_H

#include "executor/instrument.h"
#include "jit/jit.h"
#include "lib/dshash.h"
#include "nodes/plannodes.h"
#include "nodes/queryjumble.h"
#include "pgstat.h"
#include "storage/lwlock.h"
#include "utils/pgstat_internal.h"
#include "utils/timestamp.h"

/* Custom pgstat kind ID */
#define PGSTAT_KIND_PGSS	25

/*
 * Entries are stored under InvalidOid so that core's per-database processing
 * (pg_stat_reset(), DROP DATABASE) leaves them alone; the real database OID
 * is part of pgssHashKey, which determines the object ID.
 *
 * XXX DROP DATABASE should arguably remove matching entries.
 */
#define PGSS_STATS_DBOID	InvalidOid

/*
 * Eviction ranks entries by usage, matching pg_stat_statements' historical
 * policy.  Usage grows by one fixed-point unit for each planning or execution
 * store, and is decayed when eviction scans entries.
 *
 * Entries created by the parse hook are sticky until the statement completes:
 * they start at the current median usage, but decay faster if no counter is
 * ever stored.
 */
#define PGSS_USAGE_ONE			(UINT64CONST(1) << 16)
#define PGSS_USAGE_MEDIAN_INIT		(10 * PGSS_USAGE_ONE)
#define PGSS_USAGE_RETAIN_PERCENT	99
#define PGSS_STICKY_RETAIN_PERCENT	50

typedef enum pgssStickyState
{
	PGSS_STICKY_NONE = 0,
	PGSS_STICKY_PENDING,
	PGSS_STICKY_UNSTICKING,
}			pgssStickyState;

/*
 * Eviction starts once entries exceed pgss_max by this much, and never evicts
 * more than this fraction of the live entries in one pass.
 */
#define PGSS_EVICT_THRESHOLD_PERCENT	110
#define PGSS_EVICT_PASS_PERCENT		10

/*
 * Extra slots for arrays sized from the live-entry gauge.  Entries can be
 * admitted while scans run, so this avoids immediate repalloc for small drift.
 */
#define PGSS_ENTRY_SCAN_HEADROOM	16

typedef enum pgssStoreKind
{
	PGSS_INVALID = -1,

	/*
	 * PGSS_PLAN and PGSS_EXEC must be respectively 0 and 1 as they're used to
	 * reference the underlying values in the arrays in the Counters struct,
	 * and this order is required in pg_stat_statements_internal().
	 */
	PGSS_PLAN = 0,
	PGSS_EXEC,
} pgssStoreKind;

#define PGSS_NUMKIND (PGSS_EXEC + 1)

/*
 * Hashtable key that defines the identity of a hashtable entry.  We separate
 * queries by user and by database even if they are otherwise identical.
 *
 * The whole struct is hashed to compute the pgstat object ID, so callers
 * must zero the padding bytes before hashing.
 */
typedef struct pgssHashKey
{
	Oid			userid;			/* user OID */
	Oid			dbid;			/* database OID */
	int64		queryid;		/* query identifier */
	bool		toplevel;		/* query executed at top level */
} pgssHashKey;

/*
 * The actual stats counters kept within pgssEntry.
 */
typedef struct Counters
{
	int64		calls[PGSS_NUMKIND];	/* # of times planned/executed */
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
	int64		rows;			/* total # of retrieved or affected rows */
	int64		shared_blks_hit;	/* # of shared buffer hits */
	int64		shared_blks_read;	/* # of shared disk blocks read */
	int64		shared_blks_dirtied;	/* # of shared disk blocks dirtied */
	int64		shared_blks_written;	/* # of shared disk blocks written */
	int64		local_blks_hit; /* # of local buffer hits */
	int64		local_blks_read;	/* # of local disk blocks read */
	int64		local_blks_dirtied; /* # of local disk blocks dirtied */
	int64		local_blks_written; /* # of local disk blocks written */
	int64		temp_blks_read; /* # of temp blocks read */
	int64		temp_blks_written;	/* # of temp blocks written */
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
	int64		wal_records;	/* # of WAL records generated */
	int64		wal_fpi;		/* # of WAL full page images generated */
	uint64		wal_bytes;		/* total amount of WAL generated in bytes */
	int64		wal_buffers_full;	/* # of times the WAL buffers became full */
	int64		jit_functions;	/* total number of JIT functions emitted */
	double		jit_generation_time;	/* total time to generate jit code */
	int64		jit_inlining_count; /* number of times inlining time has been
									 * > 0 */
	double		jit_deform_time;	/* total time to deform tuples in jit code */
	int64		jit_deform_count;	/* number of times deform time has been >
									 * 0 */

	double		jit_inlining_time;	/* total time to inline jit code */
	int64		jit_optimization_count; /* number of times optimization time
										 * has been > 0 */
	double		jit_optimization_time;	/* total time to optimize jit code */
	int64		jit_emission_count; /* number of times emission time has been
									 * > 0 */
	double		jit_emission_time;	/* total time to emit jit code */
	int64		parallel_workers_to_launch; /* # of parallel workers planned
											 * to be launched */
	int64		parallel_workers_launched;	/* # of parallel workers actually
											 * launched */
	int64		generic_plan_calls; /* number of calls using a generic plan */
	int64		custom_plan_calls;	/* number of calls using a custom plan */
} Counters;

/*
 * Statistics per statement
 *
 * Note: in event of a failure in garbage collection of the query text file,
 * we reset query_offset to zero and query_len to -1.  This will be seen as
 * an invalid state by pgss_fetch_query_text().
 */
typedef struct pgssEntry
{
	pgssHashKey key;			/* hash key of entry */
	Counters	counters;		/* the statistics for this query */
	Size		query_offset;	/* query text offset in external file */
	int			query_len;		/* # of valid bytes in query string, or -1 */
	int			encoding;		/* query text encoding */
	TimestampTz stats_since;	/* timestamp of entry allocation */
	TimestampTz minmax_stats_since; /* timestamp of last min/max values reset */
} pgssEntry;

/*
 * pgstat shared entry for pg_stat_statements.
 */
typedef struct PgStatShared_Pgss
{
	PgStatShared_Common header;
	pgssEntry	entry;
	pg_atomic_uint32 entry_live;	/* entry is visible to PGSS */
	pg_atomic_uint32 sticky;	/* no planning or execution recorded yet */
	pg_atomic_uint64 usage;		/* eviction score, fixed-point */
} PgStatShared_Pgss;

StaticAssertDecl(offsetof(PgStatShared_Pgss, usage) % 8 == 0,
				 "PgStatShared_Pgss.usage must be 8-byte aligned");

/*
 * Global shared state.
 *
 * Kept in a named DSM segment: a stats kind is either fixed- or
 * variable-numbered, and PGSTAT_KIND_PGSS is the latter.
 */
typedef struct pgssSharedState
{
	pg_atomic_uint64 entry_dealloc; /* # of entry eviction passes */
	pg_atomic_uint64 entry_count;	/* # of live entries */
	pg_atomic_uint64 stats_reset;	/* timestamp with all stats reset */
	pg_atomic_uint64 cur_median_usage;	/* current median usage of entries */
	pg_atomic_uint64 qtext_extent;	/* current extent of query file */
	Size		mean_query_len; /* current mean entry text length */
	LWLock		entry_evict_lock;	/* serializes entry eviction */
	LWLock		qtext_lock;		/* protects external query text file */
} pgssSharedState;

/* Local copy of an entry, for pg_stat_statements_internal() */
typedef struct pgssViewRow
{
	pgssHashKey key;
	TimestampTz stats_since;
	TimestampTz minmax_stats_since;
	int			encoding;
	int			query_len;
	char	   *query;			/* NULL if not shown */
	Counters	counters;
} pgssViewRow;

extern int	pgss_max;
extern bool pgss_save;
extern pgssSharedState *pgss_shared;
extern dshash_table *pgss_hash;

extern void pgss_register_kind(void);
extern void pgss_require_ready(void);
extern PgStatShared_Pgss *pgss_shared_from_hash_entry(PgStatShared_HashEntry *p);
extern void pgss_mark_entry_live(PgStatShared_Pgss *shared);
extern uint64 pgss_entry_dsa_size(void);
extern uint64 pgss_evict_threshold(void);
extern bool pgss_admission_pressure(uint64 threshold);
extern void pgss_evict_for_entry(void);
extern TimestampTz pgss_entry_reset(Oid userid, Oid dbid, int64 queryid,
									bool minmax_only);
extern void pgss_entry_accumulate(PgStat_EntryRef *entry_ref,
								  pgssStoreKind kind,
								  double total_time, uint64 rows,
								  const BufferUsage *bufusage,
								  const WalUsage *walusage,
								  const JitInstrumentation *jitusage,
								  int parallel_workers_to_launch,
								  int parallel_workers_launched,
								  PlannedStmtOrigin planOrigin);
extern bool pgss_store_query_text_file(const char *query, int query_len,
									   Size *query_offset);
extern bool pgss_store_query_text_file_locked(const char *query, int query_len,
											  Size *query_offset);
extern char *pgss_load_query_text_file(Size *buffer_size);
extern char *pgss_fetch_query_text(Size query_offset, int query_len,
								   char *buffer, Size buffer_size);
extern bool pgss_need_query_text_gc(void);
extern void pgss_gc_query_texts(void);
extern void pgss_reset_query_text_file(void);

static inline bool
pgss_entry_ref_is_dropped(PgStat_EntryRef *entry_ref)
{
	Assert(entry_ref->shared_entry != NULL);
	Assert(entry_ref->shared_stats != NULL);

	return entry_ref->shared_entry->dropped ||
		pg_atomic_read_u32(&entry_ref->shared_entry->generation) !=
		entry_ref->generation;
}

static inline bool
pgss_entry_ref_is_stale(PgStat_EntryRef *entry_ref)
{
	PgStatShared_Pgss *shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	Assert(entry_ref->shared_entry != NULL);
	Assert(entry_ref->shared_stats != NULL);

	return pgss_entry_ref_is_dropped(entry_ref) ||
		pg_atomic_read_u32(&shared->entry_live) == 0;
}

#endif							/* PG_STAT_STATEMENTS_INTERNAL_H */
