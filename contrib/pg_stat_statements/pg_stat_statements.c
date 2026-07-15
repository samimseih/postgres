/*-------------------------------------------------------------------------
 *
 * pg_stat_statements.c
 *		Track statement planning and execution times as well as resource
 *		usage across a whole database cluster.
 *
 * Execution costs are totaled for each distinct source query, and kept in
 * a custom pgstat kind entry.  The number of distinct queries tracked is
 * bounded by pg_stat_statements.max.
 *
 * Starting in Postgres 9.2, this module normalized query entries.  As of
 * Postgres 14, the normalization is done by the core if compute_query_id
 * is enabled, or optionally by third-party modules.
 *
 * To facilitate presenting entries to users, we create "representative" query
 * strings in which constants are replaced with parameter symbols ($n), to
 * make it clearer what a normalized entry can represent. The query text is
 * stored in a dedicated DSA area with the pointer tracked by the pgstat entry.
 *
 * Notes about locking: All locking is provided by the core pgstat infrastructure.
 *
 * Notes about eviction: When the number of tracked statements exceeds the
 * configured limit, the module evicts the least-recently-used entries based on a
 * sampling mechanism.
 *
 * Copyright (c) 2008-2026, PostgreSQL Global Development Group
 *
 * IDENTIFICATION
 *	  contrib/pg_stat_statements/pg_stat_statements.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "access/parallel.h"
#include "catalog/pg_authid.h"
#include "common/hashfn.h"
#include "executor/instrument.h"
#include "funcapi.h"
#include "jit/jit.h"
#include "lib/dshash.h"
#include "mb/pg_wchar.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "nodes/queryjumble.h"
#include "optimizer/planner.h"
#include "parser/analyze.h"
#include "storage/dsm_registry.h"
#include "tcop/utility.h"
#include "access/xact.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/dsa.h"
#include "utils/guc.h"
#include "utils/numeric.h"
#include "utils/pgstat_internal.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"

PG_MODULE_MAGIC_EXT(
					.name = "pg_stat_statements",
					.version = PG_VERSION
);

/* Custom pgstat kind ID */
#define PGSTAT_KIND_PGSS	25

/* Local read/write helpers for stats serialization */
#define write_chunk(fpout, ptr, len) (fwrite(ptr, len, 1, fpout) == 1)
#define write_chunk_s(fpout, ptr) write_chunk(fpout, ptr, sizeof(*ptr))
#define read_chunk(fpin, ptr, len) (fread(ptr, 1, len, fpin) == (len))
#define read_chunk_s(fpin, ptr) read_chunk(fpin, ptr, sizeof(*ptr))

/*
 * Extension version number, for supporting older extension versions' objects
 */
typedef enum pgssVersion
{
	PGSS_V1_0 = 0,
	PGSS_V1_1,
	PGSS_V1_2,
	PGSS_V1_3,
	PGSS_V1_8,
	PGSS_V1_9,
	PGSS_V1_10,
	PGSS_V1_11,
	PGSS_V1_12,
	PGSS_V1_13,
} pgssVersion;

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
 * Hashtable key that defines the identity of a tracked statement.
 * We separate queries by user and by database even if they are otherwise
 * identical.
 */
typedef struct pgssHashKey
{
	Oid			userid;			/* user OID */
	Oid			dbid;			/* database OID */
	int64		queryid;		/* query identifier */
	bool		toplevel;		/* query executed at top level */
} pgssHashKey;

/*
 * The actual stats counters kept within the custom pgstat kind.
 */
typedef struct pgssCounters
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
} pgssCounters;

/*
 * Statistics per statement
 */
typedef struct PgStatShared_Pgss
{
	PgStatShared_Common header;
	pgssHashKey key;
	dsa_pointer query_text;		/* DSA pointer to query text */
	int			query_len;		/* # of valid bytes in query string, or -1 */
	int			encoding;		/* query text encoding */
	TimestampTz stats_since;	/* timestamp of entry allocation */
	TimestampTz minmax_stats_since; /* timestamp of last min/max values reset */
	pg_atomic_uint64 last_access;	/* statement start timestamp of last
									 * access, for LRU eviction */
	pgssCounters counters;
} PgStatShared_Pgss;

/*
 * Eviction ring: holds the coldest candidates found during each sampling
 * pass.  Sized as a percentage of pgss_max, capped to avoid excessive
 * shared memory use.  The array in pgssSharedState is allocated at the
 * maximum size; the actual number of candidates collected per pass is
 * computed at runtime from pgss_max.
 */
#define PGSS_SAMPLE_RING_PCT	10	/* % of pgss_max to keep as candidates */
#define PGSS_SAMPLE_RING_MAX	1024	/* upper bound on ring size */
#define PGSS_SAMPLE_INTERVAL_MS	1000	/* minimum ms between sample passes */
#define PGSS_ENTRY_DSA_SIZE(n)	((size_t) (n) * sizeof(PgStatShared_Pgss) * 2)
#define PGSS_QTEXT_DSA_SIZE(kb)	((size_t) (kb) * 1024)

typedef struct pgssEvictionCandidate
{
	Oid			dbid;
	uint64		objid;
	int64		last_access;
} pgssEvictionCandidate;

/*
 * Global shared state
 */
typedef struct pgssSharedState
{
	pg_atomic_uint64 dealloc;	/* total # of entries evicted */
	pg_atomic_uint64 stats_reset;	/* timestamp with all stats reset */

	/* Sample-based eviction state */
	pg_atomic_uint64 last_sample;	/* timestamp of last sample pass */
	pg_atomic_uint32 ring_write;	/* next write position (sampler) */
	pg_atomic_uint32 ring_read; /* next read position (inline evict) */
	pgssEvictionCandidate ring[PGSS_SAMPLE_RING_MAX];
} pgssSharedState;

/* Backend-local pending entry */
/* Links to shared memory state */
static pgssSharedState *pgss_shared = NULL;
static dsa_area *pgss_qtext_dsa = NULL;
static dshash_table *pgss_hash = NULL;

/* Backend-local pending entry */
typedef struct PgStat_PgssPending
{
	pgssHashKey key;
	pgssCounters counters;
} PgStat_PgssPending;

typedef struct pgssResetFilter
{
	Oid			userid;
	Oid			dbid;
	int64		queryid;
} pgssResetFilter;

/*---- Local variables ----*/

/* Current nesting depth of planner/ExecutorRun/ProcessUtility calls */
static int	nesting_level = 0;

/* Saved hook values */
static post_parse_analyze_hook_type prev_post_parse_analyze_hook = NULL;
static planner_hook_type prev_planner_hook = NULL;
static ExecutorStart_hook_type prev_ExecutorStart = NULL;
static ExecutorRun_hook_type prev_ExecutorRun = NULL;
static ExecutorFinish_hook_type prev_ExecutorFinish = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd = NULL;
static ProcessUtility_hook_type prev_ProcessUtility = NULL;

/*---- GUC variables ----*/

typedef enum
{
	PGSS_TRACK_NONE,			/* track no statements */
	PGSS_TRACK_TOP,				/* only top level statements */
	PGSS_TRACK_ALL,				/* all statements, including nested ones */
}			PGSSTrackLevel;

static const struct config_enum_entry track_options[] =
{
	{"none", PGSS_TRACK_NONE, false},
	{"top", PGSS_TRACK_TOP, false},
	{"all", PGSS_TRACK_ALL, false},
	{NULL, 0, false}
};

static int	pgss_max = 5000;	/* max # statements to track */
static int	pgss_track = PGSS_TRACK_TOP;	/* tracking level */
static bool pgss_track_utility = true;	/* whether to track utility commands */
static bool pgss_track_planning = false;	/* whether to track planning
											 * duration */
static bool pgss_save = true;	/* whether to save stats across shutdown */
static int	pgss_query_text_memory = 4096;	/* in KB XXX: Should default be
											 * lower? */

#define pgss_enabled(level) \
	(!IsParallelWorker() && \
	(pgss_track == PGSS_TRACK_ALL || \
	(pgss_track == PGSS_TRACK_TOP && (level) == 0)))

/*---- Function declarations ----*/

PG_FUNCTION_INFO_V1(pg_stat_statements_reset);
PG_FUNCTION_INFO_V1(pg_stat_statements_reset_1_7);
PG_FUNCTION_INFO_V1(pg_stat_statements_reset_1_11);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_2);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_3);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_8);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_9);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_10);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_11);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_12);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_13);
PG_FUNCTION_INFO_V1(pg_stat_statements_1_14);
PG_FUNCTION_INFO_V1(pg_stat_statements);
PG_FUNCTION_INFO_V1(pg_stat_statements_info);

static void pgss_post_parse_analyze(ParseState *pstate, Query *query,
									const JumbleState *jstate);
static PlannedStmt *pgss_planner(Query *parse,
								 const char *query_string,
								 int cursorOptions,
								 ParamListInfo boundParams,
								 ExplainState *es);
static void pgss_ExecutorStart(QueryDesc *queryDesc, int eflags);
static void pgss_ExecutorRun(QueryDesc *queryDesc,
							 ScanDirection direction,
							 uint64 count);
static void pgss_ExecutorFinish(QueryDesc *queryDesc);
static void pgss_ExecutorEnd(QueryDesc *queryDesc);
static void pgss_ProcessUtility(PlannedStmt *pstmt, const char *queryString,
								bool readOnlyTree,
								ProcessUtilityContext context, ParamListInfo params,
								QueryEnvironment *queryEnv,
								DestReceiver *dest, QueryCompletion *qc);
static void pgss_store(const char *query, int64 queryId,
					   int query_location, int query_len,
					   pgssStoreKind kind,
					   double total_time, uint64 rows,
					   const BufferUsage *bufusage,
					   const WalUsage *walusage,
					   const struct JitInstrumentation *jitusage,
					   const JumbleState *jstate,
					   int parallel_workers_to_launch,
					   int parallel_workers_launched,
					   PlannedStmtOrigin planOrigin);
static void pg_stat_statements_internal(FunctionCallInfo fcinfo,
										pgssVersion api_version,
										bool showtext);
static void qtext_store(PgStatShared_Pgss *entry, const char *query, int query_len,
						int encoding);
static TimestampTz entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only);
static char *generate_normalized_query(const JumbleState *jstate,
									   const char *query,
									   int query_loc, int *query_len_p);
static void pgss_assign_query_text_memory(int newval, void *extra);
static void pgss_assign_max(int newval, void *extra);
static void pgss_reset_timestamp_cb(PgStatShared_Common *header, TimestampTz ts);
static bool pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data);
static bool pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data);
static void pgss_eviction_sample(void);
static void pgss_evict_one(void);
static inline uint64 pgss_hash_key(pgssHashKey *key);
static bool pgss_flush_pending_cb(PgStat_EntryRef *entry_ref, bool nowait);
static void pgss_attach_shmem_cb(void);
static bool pgss_to_serialized_data(const PgStat_HashKey *key,
									const PgStatShared_Common *header,
									FILE *statfile);
static bool pgss_from_serialized_data(const PgStat_HashKey *key,
									  PgStatShared_Common *header,
									  FILE *statfile);

/*
 * Custom pgstat kind definition
 */
static const PgStat_KindInfo pgss_kind_info = {
	.name = "pg_stat_statements",
	.fixed_amount = false,
	.write_to_file = true,
	.track_entry_count = true,
	.accessed_across_databases = true,
	.own_hash = true,
	.shared_size = sizeof(PgStatShared_Pgss),
	.shared_data_off = offsetof(PgStatShared_Pgss, counters),
	.shared_data_len = sizeof(pgssCounters),
	.pending_size = sizeof(PgStat_PgssPending),
	.init_backend_cb = pgss_attach_shmem_cb,
	.flush_pending_cb = pgss_flush_pending_cb,
	.reset_timestamp_cb = pgss_reset_timestamp_cb,
	.to_serialized_data = pgss_to_serialized_data,
	.from_serialized_data = pgss_from_serialized_data,
};

static inline uint64
pgss_hash_key(pgssHashKey *key)
{
	return hash_bytes_extended((const unsigned char *) key,
							   sizeof(pgssHashKey), 0);
}

static void
pgss_init_shmem(void *ptr, void *arg)
{
	pgssSharedState *state = (pgssSharedState *) ptr;

	pg_atomic_init_u64(&state->dealloc, 0);
	pg_atomic_init_u64(&state->stats_reset, (uint64) GetCurrentTimestamp());
	pg_atomic_init_u64(&state->last_sample, 0);
	pg_atomic_init_u32(&state->ring_write, 0);
	pg_atomic_init_u32(&state->ring_read, 0);
}

static void
pgss_assign_query_text_memory(int newval, void *extra)
{
	if (pgss_qtext_dsa)
		dsa_set_size_limit(pgss_qtext_dsa, PGSS_QTEXT_DSA_SIZE(newval));
}

static void
pgss_assign_max(int newval, void *extra)
{
	dsa_area   *kind_dsa = pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS);

	if (kind_dsa)
		dsa_set_size_limit(kind_dsa, PGSS_ENTRY_DSA_SIZE(newval));
}

static void
pgss_attach_shmem_cb(void)
{
	bool		found;

	if (pgss_shared != NULL)
		return;
	if (!pgstat_get_kind_info(PGSTAT_KIND_PGSS))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via shared_preload_libraries")));

	pgss_shared = GetNamedDSMSegment("pg_stat_statements_state",
									 sizeof(pgssSharedState),
									 pgss_init_shmem,
									 &found, NULL);

	if (pgss_qtext_dsa == NULL)
	{
		pgss_qtext_dsa = GetNamedDSA("pg_stat_statements_qtext", &found);
		dsa_set_size_limit(pgss_qtext_dsa, PGSS_QTEXT_DSA_SIZE(pgss_query_text_memory));
	}

	if (pgss_hash == NULL)
	{
		dsa_area   *kind_dsa = pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS);

		pgss_hash = pgstat_get_hash_for_kind(PGSTAT_KIND_PGSS);
		dsa_set_size_limit(kind_dsa, PGSS_ENTRY_DSA_SIZE(pgss_max));
	}
}

/*
 * Module load callback
 */
void
_PG_init(void)
{
	/*
	 * In order to register our custom pgstat kind, we have to be loaded via
	 * shared_preload_libraries.  If not, fall out without hooking into any of
	 * the main system.  (We don't throw error here because it seems useful to
	 * allow the pg_stat_statements functions to be created even when the
	 * module isn't active.  The functions must protect themselves against
	 * being called then, however.)
	 */
	if (!process_shared_preload_libraries_in_progress)
		return;

	/*
	 * Inform the postmaster that we want to enable query_id calculation if
	 * compute_query_id is set to auto.
	 */
	EnableQueryId();

	/* Register custom pgstat kind */
	pgstat_register_kind(PGSTAT_KIND_PGSS, &pgss_kind_info);

	/*
	 * Define (or redefine) custom GUC variables.
	 */
	DefineCustomIntVariable("pg_stat_statements.max",
							"Sets the maximum number of statements tracked by pg_stat_statements.",
							NULL,
							&pgss_max,
							5000,
							100,
							INT_MAX / 2,
							PGC_SIGHUP,
							0,
							NULL,
							pgss_assign_max,
							NULL);

	DefineCustomEnumVariable("pg_stat_statements.track",
							 "Selects which statements are tracked by pg_stat_statements.",
							 NULL,
							 &pgss_track,
							 PGSS_TRACK_TOP,
							 track_options,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("pg_stat_statements.track_utility",
							 "Selects whether utility commands are tracked by pg_stat_statements.",
							 NULL,
							 &pgss_track_utility,
							 true,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("pg_stat_statements.track_planning",
							 "Selects whether planning duration is tracked by pg_stat_statements.",
							 NULL,
							 &pgss_track_planning,
							 false,
							 PGC_SUSET,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomBoolVariable("pg_stat_statements.save",
							 "Save pg_stat_statements statistics across server shutdowns.",
							 NULL,
							 &pgss_save,
							 true,
							 PGC_SIGHUP,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomIntVariable("pg_stat_statements.query_text_memory",
							"Sets the memory limit for query text storage.",
							NULL,
							&pgss_query_text_memory,
							4096,
							256,
							MAX_KILOBYTES,
							PGC_SIGHUP,
							GUC_UNIT_KB,
							NULL,
							pgss_assign_query_text_memory,
							NULL);

	MarkGUCPrefixReserved("pg_stat_statements");

	/*
	 * Install hooks.
	 */
	prev_post_parse_analyze_hook = post_parse_analyze_hook;
	post_parse_analyze_hook = pgss_post_parse_analyze;
	prev_planner_hook = planner_hook;
	planner_hook = pgss_planner;
	prev_ExecutorStart = ExecutorStart_hook;
	ExecutorStart_hook = pgss_ExecutorStart;
	prev_ExecutorRun = ExecutorRun_hook;
	ExecutorRun_hook = pgss_ExecutorRun;
	prev_ExecutorFinish = ExecutorFinish_hook;
	ExecutorFinish_hook = pgss_ExecutorFinish;
	prev_ExecutorEnd = ExecutorEnd_hook;
	ExecutorEnd_hook = pgss_ExecutorEnd;
	prev_ProcessUtility = ProcessUtility_hook;
	ProcessUtility_hook = pgss_ProcessUtility;
}

/*
 * Merge the pending per store kind counters.
 */
static void
pgss_flush_kind(pgssCounters *shared, pgssCounters *pending, pgssStoreKind kind)
{
	int64		n_a,
				n_b;
	double		delta;

	n_a = shared->calls[kind];
	n_b = pending->calls[kind];

	shared->calls[kind] += n_b;
	shared->total_time[kind] += pending->total_time[kind];

	if (n_a == 0)
	{
		shared->min_time[kind] = pending->min_time[kind];
		shared->max_time[kind] = pending->max_time[kind];
		shared->mean_time[kind] = pending->mean_time[kind];
		shared->sum_var_time[kind] = pending->sum_var_time[kind];
	}
	else
	{
		if (pending->min_time[kind] < shared->min_time[kind])
			shared->min_time[kind] = pending->min_time[kind];
		if (pending->max_time[kind] > shared->max_time[kind])
			shared->max_time[kind] = pending->max_time[kind];

		/*
		 * Chan's parallel variance algorithm: combine two sets of (count,
		 * mean, sum_of_squared_deviations). See
		 * <http://www.johndcook.com/blog/standard_deviation/>
		 */
		delta = pending->mean_time[kind] - shared->mean_time[kind];
		shared->sum_var_time[kind] +=
			pending->sum_var_time[kind] +
			delta * delta * (double) n_a * (double) n_b / (double) (n_a + n_b);
		shared->mean_time[kind] =
			shared->total_time[kind] / shared->calls[kind];
	}
}

/*
 * Callback function to flush pending statistics for a given entry.
 */
static bool
pgss_flush_pending_cb(PgStat_EntryRef *entry_ref, bool nowait)
{
	PgStat_PgssPending *pending;
	PgStatShared_Pgss *shared;

	pending = (PgStat_PgssPending *) entry_ref->pending;
	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return false;

	shared->key = pending->key;

	pgss_flush_kind(&shared->counters, &pending->counters, PGSS_EXEC);

	if (pgss_track_planning && pending->counters.calls[PGSS_PLAN] > 0)
		pgss_flush_kind(&shared->counters, &pending->counters, PGSS_PLAN);

	shared->counters.rows += pending->counters.rows;
	shared->counters.shared_blks_hit += pending->counters.shared_blks_hit;
	shared->counters.shared_blks_read += pending->counters.shared_blks_read;
	shared->counters.shared_blks_dirtied += pending->counters.shared_blks_dirtied;
	shared->counters.shared_blks_written += pending->counters.shared_blks_written;
	shared->counters.local_blks_hit += pending->counters.local_blks_hit;
	shared->counters.local_blks_read += pending->counters.local_blks_read;
	shared->counters.local_blks_dirtied += pending->counters.local_blks_dirtied;
	shared->counters.local_blks_written += pending->counters.local_blks_written;
	shared->counters.temp_blks_read += pending->counters.temp_blks_read;
	shared->counters.temp_blks_written += pending->counters.temp_blks_written;
	shared->counters.shared_blk_read_time += pending->counters.shared_blk_read_time;
	shared->counters.shared_blk_write_time += pending->counters.shared_blk_write_time;
	shared->counters.local_blk_read_time += pending->counters.local_blk_read_time;
	shared->counters.local_blk_write_time += pending->counters.local_blk_write_time;
	shared->counters.temp_blk_read_time += pending->counters.temp_blk_read_time;
	shared->counters.temp_blk_write_time += pending->counters.temp_blk_write_time;
	shared->counters.wal_records += pending->counters.wal_records;
	shared->counters.wal_fpi += pending->counters.wal_fpi;
	shared->counters.wal_bytes += pending->counters.wal_bytes;
	shared->counters.wal_buffers_full += pending->counters.wal_buffers_full;
	shared->counters.jit_functions += pending->counters.jit_functions;
	shared->counters.jit_generation_time += pending->counters.jit_generation_time;
	shared->counters.jit_inlining_count += pending->counters.jit_inlining_count;
	shared->counters.jit_inlining_time += pending->counters.jit_inlining_time;
	shared->counters.jit_optimization_count += pending->counters.jit_optimization_count;
	shared->counters.jit_optimization_time += pending->counters.jit_optimization_time;
	shared->counters.jit_emission_count += pending->counters.jit_emission_count;
	shared->counters.jit_emission_time += pending->counters.jit_emission_time;
	shared->counters.jit_deform_count += pending->counters.jit_deform_count;
	shared->counters.jit_deform_time += pending->counters.jit_deform_time;
	shared->counters.parallel_workers_to_launch += pending->counters.parallel_workers_to_launch;
	shared->counters.parallel_workers_launched += pending->counters.parallel_workers_launched;
	shared->counters.generic_plan_calls += pending->counters.generic_plan_calls;
	shared->counters.custom_plan_calls += pending->counters.custom_plan_calls;

	pgstat_unlock_entry(entry_ref);

	/*
	 * Opportunistic time to trigger an eviction sample.
	 */
	pgss_eviction_sample();

	return true;
}

/*
 * Serialize entry metadata and query text alongside each pgstat entry.
 * On restart, from_serialized_data reconstructs both.
 */
static bool
pgss_to_serialized_data(const PgStat_HashKey *key,
						const PgStatShared_Common *header,
						FILE *statfile)
{
	PgStatShared_Pgss *shpgss = (PgStatShared_Pgss *) header;
	bool		found = pgss_save;
	char	   *qtext = NULL;
	int			qtext_len = 0;

	if (!write_chunk_s(statfile, &found))
		return false;

	if (!pgss_save)
		return true;

	if (!write_chunk_s(statfile, &shpgss->encoding))
		return false;
	if (!write_chunk_s(statfile, &shpgss->stats_since))
		return false;
	if (!write_chunk_s(statfile, &shpgss->minmax_stats_since))
		return false;

	/* Write query text */
	if (DsaPointerIsValid(shpgss->query_text) && shpgss->query_len >= 0)
		qtext = dsa_get_address(pgss_qtext_dsa, shpgss->query_text);

	if (qtext)
	{
		qtext_len = shpgss->query_len;
		if (!write_chunk_s(statfile, &qtext_len))
			return false;
		if (!write_chunk(statfile, qtext, qtext_len + 1))
			return false;
	}
	else
	{
		qtext_len = -1;
		if (!write_chunk_s(statfile, &qtext_len))
			return false;
	}

	return true;
}

/*
 * On startup, restores metadata fields and query text.
 */
static bool
pgss_from_serialized_data(const PgStat_HashKey *key,
						  PgStatShared_Common *header,
						  FILE *statfile)
{
	PgStatShared_Pgss *shpgss = (PgStatShared_Pgss *) header;
	bool		had_entry;
	int			qtext_len;

	if (!read_chunk_s(statfile, &had_entry))
		return false;

	if (!had_entry)
	{
		pgstat_drop_entry(PGSTAT_KIND_PGSS, key->dboid, key->objid, false);
		return true;
	}

	if (!read_chunk_s(statfile, &shpgss->encoding))
		return false;
	if (!read_chunk_s(statfile, &shpgss->stats_since))
		return false;
	if (!read_chunk_s(statfile, &shpgss->minmax_stats_since))
		return false;
	if (!read_chunk_s(statfile, &qtext_len))
		return false;

	if (qtext_len >= 0)
	{
		char	   *qtext;
		dsa_pointer dp;

		qtext = palloc(qtext_len + 1);
		if (!read_chunk(statfile, qtext, qtext_len + 1))
		{
			pfree(qtext);
			return false;
		}

		dp = dsa_allocate_extended(pgss_qtext_dsa, qtext_len + 1, DSA_ALLOC_NO_OOM);
		if (DsaPointerIsValid(dp))
		{
			memcpy(dsa_get_address(pgss_qtext_dsa, dp), qtext, qtext_len + 1);
			shpgss->query_text = dp;
			shpgss->query_len = qtext_len;
		}
		else
		{
			shpgss->query_text = InvalidDsaPointer;
			shpgss->query_len = -1;
		}

		pfree(qtext);
	}
	else
	{
		shpgss->query_text = InvalidDsaPointer;
		shpgss->query_len = -1;
	}


	return true;
}

/*
 * Scans all entries in shared mode, identifies the weakest entries by
 * last access time, and fills the eviction ring buffer.  Throttled to
 * once per PGSS_SAMPLE_INTERVAL_MS milliseconds.
 */
static void
pgss_eviction_sample(void)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *p;
	TimestampTz now;
	int			ncandidates = 0;
	int64		max_last_access = 0;
	int			max_idx = 0;
	int			ring_target = Min(pgss_max * PGSS_SAMPLE_RING_PCT / 100,
								  PGSS_SAMPLE_RING_MAX);
	pgssEvictionCandidate candidates[PGSS_SAMPLE_RING_MAX];
	int			i;

	if (IsTransactionOrTransactionBlock())
		return;

	now = GetCurrentTransactionStopTimestamp();
	if (pg_atomic_read_u64(&pgss_shared->last_sample) != 0 &&
		!TimestampDifferenceExceeds((TimestampTz) pg_atomic_read_u64(&pgss_shared->last_sample),
									now, PGSS_SAMPLE_INTERVAL_MS))
	{
		return;
	}

	/*
	 * Find the ring_target entries with the oldest last_access timestamps.
	 * These become eviction candidates.
	 */
	dshash_seq_init(&hstat, pgss_hash, false);
	while ((p = dshash_seq_next(&hstat)) != NULL)
	{
		PgStatShared_Pgss *shared;
		int64		last_access;
		int			idx;

		if (p->dropped)
			continue;

		shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), p->body);
		last_access = (int64) pg_atomic_read_u64(&shared->last_access);

		if (ncandidates < ring_target)
			idx = ncandidates++;
		else if (last_access < max_last_access)
			idx = max_idx;		/* older than our warmest candidate, replace
								 * it */
		else
			continue;

		candidates[idx].dbid = p->key.dboid;
		candidates[idx].objid = p->key.objid;
		candidates[idx].last_access = last_access;

		/*
		 * We replaced the warmest candidate at this position. Now, we need to
		 * rescan to find the new warmest from the entire set.
		 */
		if (idx == max_idx)
		{
			max_last_access = candidates[0].last_access;
			max_idx = 0;
			for (i = 1; i < ncandidates; i++)
			{
				if (candidates[i].last_access > max_last_access)
				{
					max_last_access = candidates[i].last_access;
					max_idx = i;
				}
			}
		}
		else if (last_access > max_last_access)
		{
			/* This candidate is warmer than the current max; update directly */
			max_last_access = last_access;
			max_idx = idx;
		}
	}
	dshash_seq_term(&hstat);

	if (ncandidates == 0)
	{
		pg_atomic_write_u64(&pgss_shared->last_sample, (uint64) now);
		return;
	}

	/* Write candidates into ring */
	for (i = 0; i < ncandidates; i++)
	{
		pgss_shared->ring[i].dbid = candidates[i].dbid;
		pgss_shared->ring[i].objid = candidates[i].objid;
		pgss_shared->ring[i].last_access = candidates[i].last_access;
	}

	pg_atomic_write_u32(&pgss_shared->ring_read, 0);
	pg_atomic_write_u32(&pgss_shared->ring_write, (uint32) ncandidates);
	pg_atomic_write_u64(&pgss_shared->last_sample, (uint64) now);
}

/*
 * Try to evict a single candidate from the ring buffer.  Pops a victim
 * from the ring, validates it is still cold, and drops it.  Returns true
 * Does nothing if the ring is empty or the candidate is no longer eligible.
 */
static void
pgss_evict_one(void)
{
	uint32		read_pos;
	uint32		write_pos;
	Oid			dbid;
	uint64		objid;
	int64		expected_last_access;
	PgStatShared_HashEntry *victim;
	PgStatShared_Pgss *shared;
	dsa_pointer text_ptr;
	PgStat_HashKey victim_key;

	/* Atomically pop from ring */
	read_pos = pg_atomic_read_u32(&pgss_shared->ring_read);
	write_pos = pg_atomic_read_u32(&pgss_shared->ring_write);

	if (read_pos >= write_pos)
		return;					/* ring empty */

	if (!pg_atomic_compare_exchange_u32(&pgss_shared->ring_read,
										&read_pos, read_pos + 1))
		return;					/* another backend grabbed it */

	dbid = pgss_shared->ring[read_pos % PGSS_SAMPLE_RING_MAX].dbid;
	objid = pgss_shared->ring[read_pos % PGSS_SAMPLE_RING_MAX].objid;
	expected_last_access = pgss_shared->ring[read_pos % PGSS_SAMPLE_RING_MAX].last_access;

	/* Look up victim with shared lock for freshness check */
	memset(&victim_key, 0, sizeof(victim_key));
	victim_key.kind = PGSTAT_KIND_PGSS;
	victim_key.dboid = dbid;
	victim_key.objid = objid;
	victim = dshash_find(pgss_hash, &victim_key, false);

	if (!victim || victim->dropped)
	{
		if (victim)
			dshash_release_lock(pgss_hash, victim);
		return;
	}

	shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), victim->body);

	/* Freshness check: skip if entry was accessed since sampling */
	if ((int64) pg_atomic_read_u64(&shared->last_access) > expected_last_access)
	{
		dshash_release_lock(pgss_hash, victim);
		return;
	}

	/* Grab query text pointer before dropping */
	text_ptr = shared->query_text;
	shared->query_text = InvalidDsaPointer;

	/* Release shared lock before taking exclusive in pgstat_drop_entry */
	dshash_release_lock(pgss_hash, victim);

	/* Drop the entry (takes exclusive lock internally) */
	if (!pgstat_drop_entry(PGSTAT_KIND_PGSS, dbid, objid, true))
		pgstat_request_entry_refs_gc();

	if (DsaPointerIsValid(text_ptr))
		dsa_free(pgss_qtext_dsa, text_ptr);

	pg_atomic_fetch_add_u64(&pgss_shared->dealloc, 1);
	pgstat_request_entry_refs_gc();
}

/*
 * Post-parse-analysis hook: mark query with a queryId
 */
static void
pgss_post_parse_analyze(ParseState *pstate, Query *query, const JumbleState *jstate)
{
	if (prev_post_parse_analyze_hook)
		prev_post_parse_analyze_hook(pstate, query, jstate);

	/* Safety check... */
	if (!pgss_hash || !pgss_enabled(nesting_level))
		return;

	/*
	 * If it's EXECUTE, clear the queryId so that stats will accumulate for
	 * the underlying PREPARE.  But don't do this if we're not tracking
	 * utility statements, to avoid messing up another extension that might be
	 * tracking them.
	 */
	if (query->utilityStmt)
	{
		if (pgss_track_utility && IsA(query->utilityStmt, ExecuteStmt))
		{
			query->queryId = INT64CONST(0);
			return;
		}
	}

	/*
	 * If query jumbling were able to identify any ignorable constants, we
	 * immediately create a hash table entry for the query, so that we can
	 * record the normalized form of the query string.  If there were no such
	 * constants, the normalized string would be the same as the query text
	 * anyway, so there's no need for an early entry.
	 */
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

/*
 * Planner hook: forward to regular planner, but measure planning time
 * if needed.
 */
static PlannedStmt *
pgss_planner(Query *parse,
			 const char *query_string,
			 int cursorOptions,
			 ParamListInfo boundParams,
			 ExplainState *es)
{
	PlannedStmt *result;

	/*
	 * We can't process the query if no query_string is provided, as
	 * pgss_store needs it.  We also ignore query without queryid, as it would
	 * be treated as a utility statement, which may not be the case.
	 */
	if (pgss_enabled(nesting_level)
		&& pgss_track_planning && query_string
		&& parse->queryId != INT64CONST(0))
	{
		instr_time	start;
		instr_time	duration;
		BufferUsage bufusage_start,
					bufusage;
		WalUsage	walusage_start,
					walusage;

		/* We need to track buffer usage as the planner can access them. */
		bufusage_start = pgBufferUsage;

		/*
		 * Similarly the planner could write some WAL records in some cases
		 * (e.g. setting a hint bit with those being WAL-logged)
		 */
		walusage_start = pgWalUsage;
		INSTR_TIME_SET_CURRENT(start);

		nesting_level++;
		PG_TRY();
		{
			if (prev_planner_hook)
				result = prev_planner_hook(parse, query_string, cursorOptions,
										   boundParams, es);
			else
				result = standard_planner(parse, query_string, cursorOptions,
										  boundParams, es);
		}
		PG_FINALLY();
		{
			nesting_level--;
		}
		PG_END_TRY();

		INSTR_TIME_SET_CURRENT(duration);
		INSTR_TIME_SUBTRACT(duration, start);

		/* calc differences of buffer counters. */
		memset(&bufusage, 0, sizeof(BufferUsage));
		BufferUsageAccumDiff(&bufusage, &pgBufferUsage, &bufusage_start);

		/* calc differences of WAL counters. */
		memset(&walusage, 0, sizeof(WalUsage));
		WalUsageAccumDiff(&walusage, &pgWalUsage, &walusage_start);

		pgss_store(query_string,
				   parse->queryId,
				   parse->stmt_location,
				   parse->stmt_len,
				   PGSS_PLAN,
				   INSTR_TIME_GET_MILLISEC(duration),
				   0,
				   &bufusage,
				   &walusage,
				   NULL,
				   NULL,
				   0,
				   0,
				   result->planOrigin);
	}
	else
	{
		/*
		 * Even though we're not tracking plan time for this statement, we
		 * must still increment the nesting level, to ensure that functions
		 * evaluated during planning are not seen as top-level calls.
		 */
		nesting_level++;
		PG_TRY();
		{
			if (prev_planner_hook)
				result = prev_planner_hook(parse, query_string, cursorOptions,
										   boundParams, es);
			else
				result = standard_planner(parse, query_string, cursorOptions,
										  boundParams, es);
		}
		PG_FINALLY();
		{
			nesting_level--;
		}
		PG_END_TRY();
	}

	return result;
}

/*
 * ExecutorStart hook: start up tracking if needed
 */
static void
pgss_ExecutorStart(QueryDesc *queryDesc, int eflags)
{
	/*
	 * If query has queryId zero, don't track it.  This prevents double
	 * counting of optimizable statements that are directly contained in
	 * utility statements.
	 */
	if (pgss_enabled(nesting_level) && queryDesc->plannedstmt->queryId != INT64CONST(0))
	{
		/* Request all summary instrumentation, i.e. timing, buffers and WAL */
		queryDesc->query_instr_options |= INSTRUMENT_ALL;
	}

	if (prev_ExecutorStart)
		prev_ExecutorStart(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);
}

/*
 * ExecutorRun hook: all we need do is track nesting depth
 */
static void
pgss_ExecutorRun(QueryDesc *queryDesc, ScanDirection direction, uint64 count)
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

/*
 * ExecutorFinish hook: all we need do is track nesting depth
 */
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

/*
 * ExecutorEnd hook: store results if needed
 */
static void
pgss_ExecutorEnd(QueryDesc *queryDesc)
{
	int64		queryId = queryDesc->plannedstmt->queryId;

	if (queryId != INT64CONST(0) && queryDesc->query_instr &&
		pgss_enabled(nesting_level))
	{
		pgss_store(queryDesc->sourceText,
				   queryId,
				   queryDesc->plannedstmt->stmt_location,
				   queryDesc->plannedstmt->stmt_len,
				   PGSS_EXEC,
				   INSTR_TIME_GET_MILLISEC(queryDesc->query_instr->total),
				   queryDesc->estate->es_total_processed,
				   &queryDesc->query_instr->bufusage,
				   &queryDesc->query_instr->walusage,
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

/*
 * ProcessUtility hook
 */
static void
pgss_ProcessUtility(PlannedStmt *pstmt, const char *queryString,
					bool readOnlyTree,
					ProcessUtilityContext context,
					ParamListInfo params, QueryEnvironment *queryEnv,
					DestReceiver *dest, QueryCompletion *qc)
{
	Node	   *parsetree = pstmt->utilityStmt;
	int64		saved_queryId = pstmt->queryId;
	int			saved_stmt_location = pstmt->stmt_location;
	int			saved_stmt_len = pstmt->stmt_len;
	PlannedStmtOrigin saved_planOrigin = pstmt->planOrigin;
	bool		enabled = pgss_track_utility && pgss_enabled(nesting_level);

	/*
	 * Force utility statements to get queryId zero.  We do this even in cases
	 * where the statement contains an optimizable statement for which a
	 * queryId could be derived (such as EXPLAIN or DECLARE CURSOR).  For such
	 * cases, runtime control will first go through ProcessUtility and then
	 * the executor, and we don't want the executor hooks to do anything,
	 * since we are already measuring the statement's costs at the utility
	 * level.
	 *
	 * Note that this is only done if pg_stat_statements is enabled and
	 * configured to track utility statements, in the unlikely possibility
	 * that user configured another extension to handle utility statements
	 * only.
	 */
	if (enabled)
		pstmt->queryId = INT64CONST(0);

	/*
	 * If it's an EXECUTE statement, we don't track it and don't increment the
	 * nesting level.  This allows the cycles to be charged to the underlying
	 * PREPARE instead (by the Executor hooks), which is much more useful.
	 *
	 * We also don't track execution of PREPARE.  If we did, we would get one
	 * hash table entry for the PREPARE (with hash calculated from the query
	 * string), and then a different one with the same query string (but hash
	 * calculated from the query tree) would be used to accumulate costs of
	 * ensuing EXECUTEs.  This would be confusing.  Since PREPARE doesn't
	 * actually run the planner (only parse+rewrite), its costs are generally
	 * pretty negligible and it seems okay to just ignore it.
	 */
	if (enabled &&
		!IsA(parsetree, ExecuteStmt) &&
		!IsA(parsetree, PrepareStmt))
	{
		instr_time	start;
		instr_time	duration;
		uint64		rows;
		BufferUsage bufusage_start,
					bufusage;
		WalUsage	walusage_start,
					walusage;

		bufusage_start = pgBufferUsage;
		walusage_start = pgWalUsage;
		INSTR_TIME_SET_CURRENT(start);

		nesting_level++;
		PG_TRY();
		{
			if (prev_ProcessUtility)
				prev_ProcessUtility(pstmt, queryString, readOnlyTree,
									context, params, queryEnv,
									dest, qc);
			else
				standard_ProcessUtility(pstmt, queryString, readOnlyTree,
										context, params, queryEnv,
										dest, qc);
		}
		PG_FINALLY();
		{
			nesting_level--;
		}
		PG_END_TRY();

		/*
		 * CAUTION: do not access the *pstmt data structure again below here.
		 * If it was a ROLLBACK or similar, that data structure may have been
		 * freed.  We must copy everything we still need into local variables,
		 * which we did above.
		 *
		 * For the same reason, we can't risk restoring pstmt->queryId to its
		 * former value, which'd otherwise be a good idea.
		 */
		pstmt = NULL;

		INSTR_TIME_SET_CURRENT(duration);
		INSTR_TIME_SUBTRACT(duration, start);

		/*
		 * Track the total number of rows retrieved or affected by the utility
		 * statements of COPY, FETCH, CREATE TABLE AS, CREATE MATERIALIZED
		 * VIEW, REFRESH MATERIALIZED VIEW and SELECT INTO.
		 */
		rows = (qc && (qc->commandTag == CMDTAG_COPY ||
					   qc->commandTag == CMDTAG_FETCH ||
					   qc->commandTag == CMDTAG_SELECT ||
					   qc->commandTag == CMDTAG_REFRESH_MATERIALIZED_VIEW)) ?
			qc->nprocessed : 0;

		/* calc differences of buffer counters. */
		memset(&bufusage, 0, sizeof(BufferUsage));
		BufferUsageAccumDiff(&bufusage, &pgBufferUsage, &bufusage_start);

		/* calc differences of WAL counters. */
		memset(&walusage, 0, sizeof(WalUsage));
		WalUsageAccumDiff(&walusage, &pgWalUsage, &walusage_start);

		pgss_store(queryString,
				   saved_queryId,
				   saved_stmt_location,
				   saved_stmt_len,
				   PGSS_EXEC,
				   INSTR_TIME_GET_MILLISEC(duration),
				   rows,
				   &bufusage,
				   &walusage,
				   NULL,
				   NULL,
				   0,
				   0,
				   saved_planOrigin);
	}
	else
	{
		/*
		 * Even though we're not tracking execution time for this statement,
		 * we must still increment the nesting level, to ensure that functions
		 * evaluated within it are not seen as top-level calls.  But don't do
		 * so for EXECUTE; that way, when control reaches pgss_planner or
		 * pgss_ExecutorStart, we will treat the costs as top-level if
		 * appropriate.  Likewise, don't bump for PREPARE, so that parse
		 * analysis will treat the statement as top-level if appropriate.
		 *
		 * To be absolutely certain we don't mess up the nesting level,
		 * evaluate the bump_level condition just once.
		 */
		bool		bump_level =
			!IsA(parsetree, ExecuteStmt) &&
			!IsA(parsetree, PrepareStmt);

		if (bump_level)
			nesting_level++;
		PG_TRY();
		{
			if (prev_ProcessUtility)
				prev_ProcessUtility(pstmt, queryString, readOnlyTree,
									context, params, queryEnv,
									dest, qc);
			else
				standard_ProcessUtility(pstmt, queryString, readOnlyTree,
										context, params, queryEnv,
										dest, qc);
		}
		PG_FINALLY();
		{
			if (bump_level)
				nesting_level--;
		}
		PG_END_TRY();
	}
}

/*
 * Store some statistics for a statement.
 *
 * If jstate is not NULL then we're trying to create an entry for which
 * we have no statistics as yet; we just want to record the normalized
 * query string.  total_time, rows, bufusage and walusage are ignored in this
 * case.
 *
 * If kind is PGSS_PLAN or PGSS_EXEC, its value is used as the array position
 * for the arrays in the Counters field.
 */
static void
pgss_store(const char *query, int64 queryId,
		   int query_location, int query_len,
		   pgssStoreKind kind,
		   double total_time, uint64 rows,
		   const BufferUsage *bufusage,
		   const WalUsage *walusage,
		   const struct JitInstrumentation *jitusage,
		   const JumbleState *jstate,
		   int parallel_workers_to_launch,
		   int parallel_workers_launched,
		   PlannedStmtOrigin planOrigin)
{
	pgssHashKey key;

	uint64		objid;
	PgStat_EntryRef *entry_ref;
	PgStat_PgssPending *pending;
	PgStatShared_Pgss *shared;
	bool		created_entry = false;

	Assert(query != NULL);

	if (queryId == INT64CONST(0))
		return;

	memset(&key, 0, sizeof(pgssHashKey));
	key.userid = GetUserId();
	key.dbid = MyDatabaseId;
	key.queryid = queryId;
	key.toplevel = (nesting_level == 0);

	objid = pgss_hash_key(&key);

	/* Fast path: look up existing entry without creating */
	entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, key.dbid, objid,
									 false, NULL);
	if (entry_ref == NULL)
	{
		entry_ref = pgstat_get_entry_ref_extended(PGSTAT_KIND_PGSS, key.dbid,
												  objid, true, &created_entry,
												  PGSTAT_ENTRY_REF_NO_OOM);
		if (entry_ref == NULL)
		{
			pgss_evict_one();

			entry_ref = pgstat_get_entry_ref_extended(PGSTAT_KIND_PGSS, key.dbid,
													  objid, true, &created_entry,
													  PGSTAT_ENTRY_REF_NO_OOM);
			if (entry_ref == NULL)
			{
				pg_atomic_fetch_add_u64(&pgss_shared->dealloc, 1);
				return;
			}
		}
	}

	pgstat_prep_pending(entry_ref);

	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (created_entry || !DsaPointerIsValid(shared->query_text))
	{
		char	   *norm_query = NULL;
		int			encoding = GetDatabaseEncoding();

		pgstat_lock_entry(entry_ref, false);

		if (created_entry)
		{
			shared->key = key;
			shared->stats_since = GetCurrentTimestamp();
			pg_atomic_init_u64(&shared->last_access, (uint64) GetCurrentStatementStartTimestamp());
			shared->minmax_stats_since = shared->stats_since;
			shared->encoding = encoding;
		}

		if (!DsaPointerIsValid(shared->query_text))
		{
			query = CleanQuerytext(query, &query_location, &query_len);
			if (jstate && jstate->clocations_count > 0)
				norm_query = generate_normalized_query(jstate, query,
													   query_location,
													   &query_len);

			qtext_store(shared, norm_query ? norm_query : query,
						query_len, encoding);
		}
		pgstat_unlock_entry(entry_ref);

		if (norm_query)
			pfree(norm_query);
	}

	if (!jstate)
	{
		Assert(kind == PGSS_PLAN || kind == PGSS_EXEC);

		pending = (PgStat_PgssPending *) entry_ref->pending;
		pending->key = key;

		pending->counters.calls[kind]++;
		pg_atomic_write_u64(&shared->last_access, (uint64) GetCurrentStatementStartTimestamp());
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
			 * Welford's online algorithm for accumulating mean and sum of
			 * squared deviations. See
			 * <http://www.johndcook.com/blog/standard_deviation/>
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

		if (bufusage)
		{
			pending->counters.shared_blks_hit += bufusage->shared_blks_hit;
			pending->counters.shared_blks_read += bufusage->shared_blks_read;
			pending->counters.shared_blks_dirtied += bufusage->shared_blks_dirtied;
			pending->counters.shared_blks_written += bufusage->shared_blks_written;
			pending->counters.local_blks_hit += bufusage->local_blks_hit;
			pending->counters.local_blks_read += bufusage->local_blks_read;
			pending->counters.local_blks_dirtied += bufusage->local_blks_dirtied;
			pending->counters.local_blks_written += bufusage->local_blks_written;
			pending->counters.temp_blks_read += bufusage->temp_blks_read;
			pending->counters.temp_blks_written += bufusage->temp_blks_written;
			pending->counters.shared_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_read_time);
			pending->counters.shared_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->shared_blk_write_time);
			pending->counters.local_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_read_time);
			pending->counters.local_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->local_blk_write_time);
			pending->counters.temp_blk_read_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_read_time);
			pending->counters.temp_blk_write_time += INSTR_TIME_GET_MILLISEC(bufusage->temp_blk_write_time);
		}

		if (walusage)
		{
			pending->counters.wal_records += walusage->wal_records;
			pending->counters.wal_fpi += walusage->wal_fpi;
			pending->counters.wal_bytes += walusage->wal_bytes;
			pending->counters.wal_buffers_full += walusage->wal_buffers_full;
		}

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
}

/*
 * Reset statement statistics corresponding to userid, dbid, and queryid.
 */
Datum
pg_stat_statements_reset_1_7(PG_FUNCTION_ARGS)
{
	Oid			userid;
	Oid			dbid;
	int64		queryid;

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = PG_GETARG_INT64(2);

	entry_reset(userid, dbid, queryid, false);

	PG_RETURN_VOID();
}

Datum
pg_stat_statements_reset_1_11(PG_FUNCTION_ARGS)
{
	Oid			userid;
	Oid			dbid;
	int64		queryid;
	bool		minmax_only;

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = PG_GETARG_INT64(2);
	minmax_only = PG_GETARG_BOOL(3);

	PG_RETURN_TIMESTAMPTZ(entry_reset(userid, dbid, queryid, minmax_only));
}

/*
 * Reset statement statistics.
 */
Datum
pg_stat_statements_reset(PG_FUNCTION_ARGS)
{
	entry_reset(0, 0, 0, false);

	PG_RETURN_VOID();
}

/* Number of output arguments (columns) for various API versions */
#define PG_STAT_STATEMENTS_COLS_V1_0	14
#define PG_STAT_STATEMENTS_COLS_V1_1	18
#define PG_STAT_STATEMENTS_COLS_V1_2	19
#define PG_STAT_STATEMENTS_COLS_V1_3	23
#define PG_STAT_STATEMENTS_COLS_V1_8	32
#define PG_STAT_STATEMENTS_COLS_V1_9	33
#define PG_STAT_STATEMENTS_COLS_V1_10	43
#define PG_STAT_STATEMENTS_COLS_V1_11	49
#define PG_STAT_STATEMENTS_COLS_V1_12	52
#define PG_STAT_STATEMENTS_COLS_V1_13	54
#define PG_STAT_STATEMENTS_COLS			54	/* maximum of above */

/*
 * Retrieve statement statistics.
 *
 * The SQL API of this function has changed multiple times, and will likely
 * do so again in future.  To support the case where a newer version of this
 * loadable module is being used with an old SQL declaration of the function,
 * we continue to support the older API versions.  For 1.2 and later, the
 * expected API version is identified by embedding it in the C name of the
 * function.  Unfortunately we weren't bright enough to do that for 1.1.
 */
Datum
pg_stat_statements_1_14(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	/* No new columns in 1.14; uses the same layout as 1.13 */
	pg_stat_statements_internal(fcinfo, PGSS_V1_13, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_13(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_13, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_12(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_12, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_11(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_11, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_10(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_10, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_9(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_9, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_8(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_8, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_3(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_3, showtext);

	return (Datum) 0;
}

Datum
pg_stat_statements_1_2(PG_FUNCTION_ARGS)
{
	bool		showtext = PG_GETARG_BOOL(0);

	pg_stat_statements_internal(fcinfo, PGSS_V1_2, showtext);

	return (Datum) 0;
}

/*
 * Legacy entry point for pg_stat_statements() API versions 1.0 and 1.1.
 * This can be removed someday, perhaps.
 */
Datum
pg_stat_statements(PG_FUNCTION_ARGS)
{
	/* If it's really API 1.1, we'll figure that out below */
	pg_stat_statements_internal(fcinfo, PGSS_V1_0, true);

	return (Datum) 0;
}

/*
 * pg_stat_statements_internal
 *
 * Scan the per-kind pgstat dshash for all entries, reading counters and
 * metadata directly from the shared body.
 */
static void
pg_stat_statements_internal(FunctionCallInfo fcinfo,
							pgssVersion api_version,
							bool showtext)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	dshash_seq_status hstat;
	PgStatShared_HashEntry *p;
	Oid			userid = GetUserId();
	bool		is_allowed_role;

	/*
	 * Superusers or roles with the privileges of pg_read_all_stats members
	 * are allowed.
	 */
	is_allowed_role = has_privs_of_role(userid, ROLE_PG_READ_ALL_STATS);

	/* Flush pending stats so we can read up-to-date counters */
	pgstat_report_stat(true);

	InitMaterializedSRF(fcinfo, 0);

	/*
	 * Check we have the expected number of output arguments.  Aside from
	 * being a good safety check, we need a kluge here to detect API version
	 * 1.1, which was wedged into the code in an ill-considered way.
	 */
	switch (rsinfo->setDesc->natts)
	{
		case PG_STAT_STATEMENTS_COLS_V1_0:
			if (api_version != PGSS_V1_0)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_1:
			/* pg_stat_statements() should have told us 1.0 */
			if (api_version != PGSS_V1_0)
				elog(ERROR, "incorrect number of output arguments");
			api_version = PGSS_V1_1;
			break;
		case PG_STAT_STATEMENTS_COLS_V1_2:
			if (api_version != PGSS_V1_2)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_3:
			if (api_version != PGSS_V1_3)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_8:
			if (api_version != PGSS_V1_8)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_9:
			if (api_version != PGSS_V1_9)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_10:
			if (api_version != PGSS_V1_10)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_11:
			if (api_version != PGSS_V1_11)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_12:
			if (api_version != PGSS_V1_12)
				elog(ERROR, "incorrect number of output arguments");
			break;
		case PG_STAT_STATEMENTS_COLS_V1_13:
			if (api_version != PGSS_V1_13)
				elog(ERROR, "incorrect number of output arguments");
			break;
		default:
			elog(ERROR, "incorrect number of output arguments");
	}

	dshash_seq_init(&hstat, pgss_hash, false);
	while ((p = dshash_seq_next(&hstat)) != NULL)
	{
		Datum		values[PG_STAT_STATEMENTS_COLS];
		bool		nulls[PG_STAT_STATEMENTS_COLS];
		int			i = 0;
		PgStatShared_Pgss *shared;
		pgssCounters tmp;
		double		stddev;

		if (p->dropped)
			continue;

		memset(values, 0, sizeof(values));
		memset(nulls, 0, sizeof(nulls));

		shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), p->body);

		LWLockAcquire(&shared->header.lock, LW_SHARED);
		tmp = shared->counters;
		LWLockRelease(&shared->header.lock);

		if (tmp.calls[PGSS_EXEC] == 0 && tmp.calls[PGSS_PLAN] == 0)
			continue;

		values[i++] = ObjectIdGetDatum(shared->key.userid);
		values[i++] = ObjectIdGetDatum(shared->key.dbid);
		if (api_version >= PGSS_V1_9)
			values[i++] = BoolGetDatum(shared->key.toplevel);

		if (is_allowed_role || shared->key.userid == userid)
		{
			if (api_version >= PGSS_V1_2)
				values[i++] = Int64GetDatumFast(shared->key.queryid);

			if (showtext)
			{
				if (DsaPointerIsValid(shared->query_text) && shared->query_len >= 0)
				{
					char	   *qstr = dsa_get_address(pgss_qtext_dsa, shared->query_text);
					char	   *enc = pg_any_to_server(qstr, shared->query_len, shared->encoding);

					values[i++] = CStringGetTextDatum(enc);
					if (enc != qstr)
						pfree(enc);
				}
				else
					nulls[i++] = true;
			}
			else
				nulls[i++] = true;
		}
		else
		{
			if (api_version >= PGSS_V1_2)
				nulls[i++] = true;

			if (showtext)
				values[i++] = CStringGetTextDatum("<insufficient privilege>");
			else
				nulls[i++] = true;
		}

		/* Note: PGSS_PLAN is 0, PGSS_EXEC is 1 */
		for (int kind = 0; kind < PGSS_NUMKIND; kind++)
		{
			if (kind == PGSS_EXEC || api_version >= PGSS_V1_8)
			{
				values[i++] = Int64GetDatumFast(tmp.calls[kind]);
				values[i++] = Float8GetDatumFast(tmp.total_time[kind]);
			}

			if ((kind == PGSS_EXEC && api_version >= PGSS_V1_3) ||
				api_version >= PGSS_V1_8)
			{
				values[i++] = Float8GetDatumFast(tmp.min_time[kind]);
				values[i++] = Float8GetDatumFast(tmp.max_time[kind]);
				values[i++] = Float8GetDatumFast(tmp.mean_time[kind]);

				/*
				 * Note we are calculating the population variance here, not
				 * the sample variance, as we have data for the whole
				 * population, so Bessel's correction is not used, and we
				 * don't divide by tmp.calls - 1.
				 */
				if (tmp.calls[kind] > 1)
					stddev = sqrt(tmp.sum_var_time[kind] / tmp.calls[kind]);
				else
					stddev = 0.0;
				values[i++] = Float8GetDatumFast(stddev);
			}
		}

		values[i++] = Int64GetDatumFast(tmp.rows);
		values[i++] = Int64GetDatumFast(tmp.shared_blks_hit);
		values[i++] = Int64GetDatumFast(tmp.shared_blks_read);
		if (api_version >= PGSS_V1_1)
			values[i++] = Int64GetDatumFast(tmp.shared_blks_dirtied);
		values[i++] = Int64GetDatumFast(tmp.shared_blks_written);
		values[i++] = Int64GetDatumFast(tmp.local_blks_hit);
		values[i++] = Int64GetDatumFast(tmp.local_blks_read);
		if (api_version >= PGSS_V1_1)
			values[i++] = Int64GetDatumFast(tmp.local_blks_dirtied);
		values[i++] = Int64GetDatumFast(tmp.local_blks_written);
		values[i++] = Int64GetDatumFast(tmp.temp_blks_read);
		values[i++] = Int64GetDatumFast(tmp.temp_blks_written);
		if (api_version >= PGSS_V1_1)
		{
			values[i++] = Float8GetDatumFast(tmp.shared_blk_read_time);
			values[i++] = Float8GetDatumFast(tmp.shared_blk_write_time);
		}
		if (api_version >= PGSS_V1_11)
		{
			values[i++] = Float8GetDatumFast(tmp.local_blk_read_time);
			values[i++] = Float8GetDatumFast(tmp.local_blk_write_time);
		}
		if (api_version >= PGSS_V1_10)
		{
			values[i++] = Float8GetDatumFast(tmp.temp_blk_read_time);
			values[i++] = Float8GetDatumFast(tmp.temp_blk_write_time);
		}
		if (api_version >= PGSS_V1_8)
		{
			char		buf[256];
			Datum		wal_bytes;

			values[i++] = Int64GetDatumFast(tmp.wal_records);
			values[i++] = Int64GetDatumFast(tmp.wal_fpi);

			snprintf(buf, sizeof buf, UINT64_FORMAT, tmp.wal_bytes);

			/* Convert to numeric. */
			wal_bytes = DirectFunctionCall3(numeric_in,
											CStringGetDatum(buf),
											ObjectIdGetDatum(0),
											Int32GetDatum(-1));
			values[i++] = wal_bytes;
		}
		if (api_version >= PGSS_V1_12)
			values[i++] = Int64GetDatumFast(tmp.wal_buffers_full);
		if (api_version >= PGSS_V1_10)
		{
			values[i++] = Int64GetDatumFast(tmp.jit_functions);
			values[i++] = Float8GetDatumFast(tmp.jit_generation_time);
			values[i++] = Int64GetDatumFast(tmp.jit_inlining_count);
			values[i++] = Float8GetDatumFast(tmp.jit_inlining_time);
			values[i++] = Int64GetDatumFast(tmp.jit_optimization_count);
			values[i++] = Float8GetDatumFast(tmp.jit_optimization_time);
			values[i++] = Int64GetDatumFast(tmp.jit_emission_count);
			values[i++] = Float8GetDatumFast(tmp.jit_emission_time);
		}
		if (api_version >= PGSS_V1_11)
		{
			values[i++] = Int64GetDatumFast(tmp.jit_deform_count);
			values[i++] = Float8GetDatumFast(tmp.jit_deform_time);
		}
		if (api_version >= PGSS_V1_12)
		{
			values[i++] = Int64GetDatumFast(tmp.parallel_workers_to_launch);
			values[i++] = Int64GetDatumFast(tmp.parallel_workers_launched);
		}
		if (api_version >= PGSS_V1_13)
		{
			values[i++] = Int64GetDatumFast(tmp.generic_plan_calls);
			values[i++] = Int64GetDatumFast(tmp.custom_plan_calls);
		}
		if (api_version >= PGSS_V1_11)
		{
			values[i++] = TimestampTzGetDatum(shared->stats_since);
			values[i++] = TimestampTzGetDatum(shared->minmax_stats_since);
		}

		Assert(i == (api_version == PGSS_V1_0 ? PG_STAT_STATEMENTS_COLS_V1_0 :
					 api_version == PGSS_V1_1 ? PG_STAT_STATEMENTS_COLS_V1_1 :
					 api_version == PGSS_V1_2 ? PG_STAT_STATEMENTS_COLS_V1_2 :
					 api_version == PGSS_V1_3 ? PG_STAT_STATEMENTS_COLS_V1_3 :
					 api_version == PGSS_V1_8 ? PG_STAT_STATEMENTS_COLS_V1_8 :
					 api_version == PGSS_V1_9 ? PG_STAT_STATEMENTS_COLS_V1_9 :
					 api_version == PGSS_V1_10 ? PG_STAT_STATEMENTS_COLS_V1_10 :
					 api_version == PGSS_V1_11 ? PG_STAT_STATEMENTS_COLS_V1_11 :
					 api_version == PGSS_V1_12 ? PG_STAT_STATEMENTS_COLS_V1_12 :
					 api_version == PGSS_V1_13 ? PG_STAT_STATEMENTS_COLS_V1_13 :
					 -1 /* fail if you forget to update this assert */ ));

		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	dshash_seq_term(&hstat);
}

/* Number of output arguments (columns) for pg_stat_statements_info */
#define PG_STAT_STATEMENTS_INFO_COLS	2

/*
 * Return statistics of pg_stat_statements.
 */
Datum
pg_stat_statements_info(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[PG_STAT_STATEMENTS_INFO_COLS] = {0};
	bool		nulls[PG_STAT_STATEMENTS_INFO_COLS] = {0};

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	values[0] = Int64GetDatum((int64) pg_atomic_read_u64(&pgss_shared->dealloc));
	values[1] = TimestampTzGetDatum((TimestampTz) pg_atomic_read_u64(&pgss_shared->stats_reset));

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

static void
pgss_reset_timestamp_cb(PgStatShared_Common *header, TimestampTz ts)
{
	PgStatShared_Pgss *shared = (PgStatShared_Pgss *) header;

	shared->stats_since = ts;
	shared->minmax_stats_since = ts;
}

static bool
pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssResetFilter *filter = (pgssResetFilter *) DatumGetPointer(match_data);
	PgStatShared_Pgss *shared;

	shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), p->body);

	if (filter->userid && shared->key.userid != filter->userid)
		return false;
	if (filter->dbid && shared->key.dbid != filter->dbid)
		return false;
	if (filter->queryid && shared->key.queryid != filter->queryid)
		return false;

	return true;
}

static bool
pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	PgStatShared_Pgss *shared;

	if (!pgss_match_entry(p, match_data))
		return false;

	shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), p->body);

	if (DsaPointerIsValid(shared->query_text))
	{
		dsa_free(pgss_qtext_dsa, shared->query_text);
		shared->query_text = InvalidDsaPointer;
	}

	return true;
}

static TimestampTz
entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only)
{
	TimestampTz stats_reset;
	pgssResetFilter filter;

	stats_reset = GetCurrentTimestamp();

	filter.userid = userid;
	filter.dbid = dbid;
	filter.queryid = queryid;

	/*
	 * XXX: The core pgstat infrastructure only supports full entry resets
	 * (zeroing the entire data region).  For minmax_only we need a partial
	 * reset, so we scan and update the entries ourselves.
	 */
	if (minmax_only)
	{
		dshash_seq_status hstat;
		PgStatShared_HashEntry *p;

		dshash_seq_init(&hstat, pgss_hash, false);
		while ((p = dshash_seq_next(&hstat)) != NULL)
		{
			PgStatShared_Pgss *shared;

			if (p->dropped)
				continue;
			if (!pgss_match_entry(p, PointerGetDatum(&filter)))
				continue;

			shared = (PgStatShared_Pgss *) dsa_get_address(pgstat_get_dsa_for_kind(PGSTAT_KIND_PGSS), p->body);

			LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
			shared->minmax_stats_since = stats_reset;
			for (int kind = 0; kind < PGSS_NUMKIND; kind++)
			{
				shared->counters.min_time[kind] = 0;
				shared->counters.max_time[kind] = 0;
				shared->counters.mean_time[kind] = 0;
				shared->counters.sum_var_time[kind] = 0;
			}
			LWLockRelease(&shared->header.lock);
		}
		dshash_seq_term(&hstat);

		return stats_reset;
	}

	if (userid != 0 && dbid != 0 && queryid != INT64CONST(0))
	{
		pgssHashKey key;
		uint64		objid;

		memset(&key, 0, sizeof(pgssHashKey));
		key.userid = userid;
		key.dbid = dbid;
		key.queryid = queryid;

		key.toplevel = false;
		objid = pgss_hash_key(&key);
		pgstat_drop_entry(PGSTAT_KIND_PGSS, key.dbid, objid, true);

		key.toplevel = true;
		objid = pgss_hash_key(&key);
		pgstat_drop_entry(PGSTAT_KIND_PGSS, key.dbid, objid, true);

		pgstat_request_entry_refs_gc();
	}
	else
	{
		pgstat_drop_matching_entries(pgss_drop_matching_entry,
									 PointerGetDatum(&filter));
	}

	if (!userid && !dbid && !queryid)
	{
		pg_atomic_write_u64(&pgss_shared->dealloc, 0);
		pg_atomic_write_u64(&pgss_shared->stats_reset, (uint64) stats_reset);
	}

	return stats_reset;
}

/*
 * Given a query string (not necessarily null-terminated), allocate a new
 * entry in the DSA query text area and store the string there.
 *
 * On success, updates entry->query_text, entry->query_len, and
 * entry->encoding.  On allocation failure, sets query_text to
 * InvalidDsaPointer.
 */
static void
qtext_store(PgStatShared_Pgss *entry, const char *query, int query_len,
			int encoding)
{
	dsa_pointer dp;

	dp = dsa_allocate_extended(pgss_qtext_dsa, query_len + 1, DSA_ALLOC_NO_OOM);
	if (DsaPointerIsValid(dp))
	{
		char	   *dst = dsa_get_address(pgss_qtext_dsa, dp);

		memcpy(dst, query, query_len);
		dst[query_len] = '\0';
		entry->query_text = dp;
		entry->query_len = query_len;
		entry->encoding = encoding;
	}
	else
	{
		entry->query_text = InvalidDsaPointer;
		entry->query_len = -1;
		entry->encoding = encoding;
	}
}

/*
 * Generate a normalized version of the query string that will be used to
 * represent all similar queries.
 *
 * Note that the normalized representation may well vary depending on
 * just which "equivalent" query is used to create the hashtable entry.
 * We assume this is OK.
 *
 * If query_loc > 0, then "query" has been advanced by that much compared to
 * the original string start, so we need to translate the provided locations
 * to compensate.  (This lets us avoid re-scanning statements before the one
 * of interest, so it's worth doing.)
 *
 * *query_len_p contains the input string length, and is updated with
 * the result string length on exit.  The resulting string might be longer
 * or shorter depending on what happens with replacement of constants.
 *
 * Returns a palloc'd string.
 */
static char *
generate_normalized_query(const JumbleState *jstate, const char *query,
						  int query_loc, int *query_len_p)
{
	char	   *norm_query;
	int			query_len = *query_len_p;
	int			norm_query_buflen,	/* Space allowed for norm_query */
				len_to_wrt,		/* Length (in bytes) to write */
				quer_loc = 0,	/* Source query byte location */
				n_quer_loc = 0, /* Normalized query byte location */
				last_off = 0,	/* Offset from start for previous tok */
				last_tok_len = 0;	/* Length (in bytes) of that tok */
	int			num_constants_replaced = 0;
	LocationLen *locs = NULL;

	/*
	 * Determine constants' lengths (core system only gives us locations), and
	 * return a sorted copy of jstate's LocationLen data with lengths filled
	 * in.
	 */
	locs = ComputeConstantLengths(jstate, query, query_loc);

	/*
	 * Allow for $n symbols to be longer than the constants they replace.
	 * Constants must take at least one byte in text form, while a $n symbol
	 * certainly isn't more than 11 bytes, even if n reaches INT_MAX.  We
	 * could refine that limit based on the max value of n for the current
	 * query, but it hardly seems worth any extra effort to do so.
	 */
	norm_query_buflen = query_len + jstate->clocations_count * 10;

	/* Allocate result buffer */
	norm_query = palloc(norm_query_buflen + 1);

	for (int i = 0; i < jstate->clocations_count; i++)
	{
		int			off,		/* Offset from start for cur tok */
					tok_len;	/* Length (in bytes) of that tok */

		/*
		 * If we have an external param at this location, but no lists are
		 * being squashed across the query, then we skip here; this will make
		 * us print the characters found in the original query that represent
		 * the parameter in the next iteration (or after the loop is done),
		 * which is a bit odd but seems to work okay in most cases.
		 */
		if (locs[i].extern_param && !jstate->has_squashed_lists)
			continue;

		off = locs[i].location;

		/* Adjust recorded location if we're dealing with partial string */
		off -= query_loc;

		tok_len = locs[i].length;

		if (tok_len < 0)
			continue;			/* ignore any duplicates */

		/* Copy next chunk (what precedes the next constant) */
		len_to_wrt = off - last_off;
		len_to_wrt -= last_tok_len;
		Assert(len_to_wrt >= 0);
		memcpy(norm_query + n_quer_loc, query + quer_loc, len_to_wrt);
		n_quer_loc += len_to_wrt;

		/*
		 * And insert a param symbol in place of the constant token; and, if
		 * we have a squashable list, insert a placeholder comment starting
		 * from the list's second value.
		 */
		n_quer_loc += sprintf(norm_query + n_quer_loc, "$%d%s",
							  num_constants_replaced + 1 + jstate->highest_extern_param_id,
							  locs[i].squashed ? " /*, ... */" : "");
		num_constants_replaced++;

		/* move forward */
		quer_loc = off + tok_len;
		last_off = off;
		last_tok_len = tok_len;
	}

	/* Clean up, if needed */
	if (locs)
		pfree(locs);

	/*
	 * We've copied up until the last ignorable constant.  Copy over the
	 * remaining bytes of the original query string.
	 */
	len_to_wrt = query_len - quer_loc;

	Assert(len_to_wrt >= 0);
	memcpy(norm_query + n_quer_loc, query + quer_loc, len_to_wrt);
	n_quer_loc += len_to_wrt;

	Assert(n_quer_loc <= norm_query_buflen);
	norm_query[n_quer_loc] = '\0';

	*query_len_p = n_quer_loc;
	return norm_query;
}
