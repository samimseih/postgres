/*-------------------------------------------------------------------------
 *
 * pg_stat_statements.c
 *		Track statement planning and execution times as well as resource
 *		usage across a whole database cluster.
 *
 * Execution costs are totaled for each distinct source query, and kept in
 * a custom pgstat kind entry.
 *
 * Starting in Postgres 9.2, this module normalized query entries.  As of
 * Postgres 14, the normalization is done by the core if compute_query_id is
 * enabled, or optionally by third-party modules.
 *
 * To facilitate presenting entries to users, we create "representative" query
 * strings in which constants are replaced with parameter symbols ($n), to
 * make it clearer what a normalized entry can represent.   Representative query
 * strings are stored in a dedicated DSA area with the pointer tracked by the
 * pgstat entry.
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
#include "lib/stringinfo.h"
#include "mb/pg_wchar.h"
#include "miscadmin.h"
#include "nodes/plannodes.h"
#include "nodes/queryjumble.h"
#include "optimizer/planner.h"
#include "parser/analyze.h"
#include "pgstat.h"
#include "storage/dsm_registry.h"
#include "storage/lwlock.h"
#include "tcop/utility.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/dsa.h"
#include "utils/guc.h"
#include "utils/memutils.h"
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
	pg_atomic_uint32 entry_live;	/* counted in pgss_shared->entry_count */
	pg_atomic_uint64 calls_eviction;	/* approximate calls score for
										 * eviction */
	pgssCounters counters;
} PgStatShared_Pgss;

static void pgss_register_kind(void);
static bool pgss_stats_ready(void);
static void pgss_require_ready(void);
static PgStatShared_Pgss *pgss_shared_from_hash_entry(PgStatShared_HashEntry *p);
static bool pgss_evict_for_entry(void);
static bool pgss_entry_eviction_try_start(void);
static void pgss_entry_eviction_finish(void);
static char *pgss_query_text_address(PgStatShared_Pgss *shared);
static uint64 pgss_entry_count(void);
static uint64 pgss_entry_dealloc_count(void);
static uint64 pgss_entry_dsa_size(void);
static uint64 pgss_qtext_dsa_size(void);
static uint64 pgss_qtext_count(void);
static uint64 pgss_qtext_total_len(void);
static void pgss_gc_query_texts(void);
static TimestampTz pgss_stats_reset_time(void);
static void pgss_advance_calls_eviction(PgStatShared_Pgss *shared,
										uint64 calls);
static TimestampTz pgss_entry_reset(Oid userid, Oid dbid, int64 queryid,
									bool minmax_only);
static void pgss_store(const char *query, int64 queryId,
					   int query_location, int query_len,
					   bool toplevel,
					   pgssStoreKind kind,
					   double total_time, uint64 rows,
					   const BufferUsage *bufusage,
					   const WalUsage *walusage,
					   const JitInstrumentation *jitusage,
					   const JumbleState *jstate,
					   int parallel_workers_to_launch,
					   int parallel_workers_launched,
					   PlannedStmtOrigin planOrigin);

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

static int	pgss_max = 5000;	/* target number of statements to track */
static int	pgss_track = PGSS_TRACK_TOP;	/* tracking level */
static bool pgss_track_utility = true;	/* whether to track utility commands */
static bool pgss_track_planning = false;	/* whether to track planning
											 * duration */
static bool pgss_save = true;	/* whether to save stats across shutdown */

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
static void pg_stat_statements_internal(FunctionCallInfo fcinfo,
										pgssVersion api_version,
										bool showtext);

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
	pgss_register_kind();

	/*
	 * Define (or redefine) custom GUC variables.
	 */
	DefineCustomIntVariable("pg_stat_statements.max",
							"Sets the target number of statement entries retained by pg_stat_statements.",
							"Entries can temporarily exceed this target; cold entries are evicted when retained query texts exceed it by 25 percent.",
							&pgss_max,
							5000,
							0,
							INT_MAX / 2,
							PGC_SIGHUP,
							0,
							NULL,
							NULL,
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
 * Custom pgstat kind, query text storage, and counter storage support.
 *
 * Execution costs are totaled for each distinct source query and kept in a
 * custom pgstat kind entry.  Representative query strings are stored in a
 * dedicated DSA area with the pointer tracked by the pgstat entry.
 */

/* Local read/write helpers for stats serialization */
#define write_chunk(fpout, ptr, len) (fwrite(ptr, len, 1, fpout) == 1)
#define write_chunk_s(fpout, ptr) write_chunk(fpout, ptr, sizeof(*ptr))
#define read_chunk(fpin, ptr, len) (fread(ptr, 1, len, fpin) == (len))
#define read_chunk_s(fpin, ptr) read_chunk(fpin, ptr, sizeof(*ptr))

#define PGSS_QTEXT_EVICT_NUMERATOR	5	/* evict at 1.25x pgss_max */
#define PGSS_QTEXT_EVICT_DENOMINATOR	4
#define PGSS_QTEXT_EVICT_THRESHOLD() \
	((uint64) pgss_max * PGSS_QTEXT_EVICT_NUMERATOR / \
	 PGSS_QTEXT_EVICT_DENOMINATOR)

/*
 * Global shared state
 *
 * XXX: This could be represented as fixed-size pgstat state, but the
 * statement entries already use PGSTAT_KIND_PGSS as a variable-numbered kind,
 * and we do not want to reserve another kind ID for this small control state.
 * Keep it in a named DSM segment for now.
 */
typedef struct pgssSharedState
{
	pg_atomic_uint64 entry_count;	/* # of logically live entries */
	pg_atomic_uint64 entry_dealloc; /* # of entry eviction passes */
	pg_atomic_uint64 qtext_count;	/* # of retained query texts */
	pg_atomic_uint64 qtext_total_len;	/* logical bytes of query text */
	pg_atomic_uint64 stats_reset;	/* timestamp with all stats reset */
	pg_atomic_flag entry_evicting;	/* only one backend evicts entries */
	LWLock		qtext_gc_lock;	/* serializes query-text garbage collection */
} pgssSharedState;

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

/* Links to shared memory state */
static pgssSharedState *pgss_shared = NULL;
static dsa_area *pgss_qtext_dsa = NULL;
static dshash_table *pgss_hash = NULL;

static void pgss_init_shmem(void *ptr, void *arg);
static void pgss_attach_shmem_cb(void);
static void pgss_flush_kind(pgssCounters *shared, pgssCounters *pending,
							pgssStoreKind kind);
static void pgss_mark_entry_live(PgStatShared_Pgss *shared);
static void pgss_mark_entry_dead(PgStatShared_Pgss *shared);
static PgStat_FlushResult pgss_flush_pending_cb(PgStat_EntryRef *entry_ref,
												bool nowait,
												bool xact_boundary);
static bool pgss_to_serialized_data(const PgStat_HashKey *key,
									const PgStatShared_Common *header,
									FILE *statfile);
static bool pgss_from_serialized_data(const PgStat_HashKey *key,
									  PgStatShared_Common *header,
									  FILE *statfile);
static void pgss_accumulate_pending(PgStat_PgssPending *pending,
									const pgssHashKey *key,
									pgssStoreKind kind,
									double total_time, uint64 rows,
									const BufferUsage *bufusage,
									const WalUsage *walusage,
									const JitInstrumentation *jitusage,
									int parallel_workers_to_launch,
									int parallel_workers_launched,
									PlannedStmtOrigin planOrigin);
static void pgss_reset_timestamp_cb(PgStatShared_Common *header, TimestampTz ts);
static bool pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data);
static bool pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data);
static void pgss_free_query_text(PgStatShared_Pgss *shared);
static bool pgss_free_query_text_locked(PgStatShared_Pgss *shared);
static void pgss_drop_entry(Oid dbid, uint64 objid);
static bool pgss_has_query_text(PgStatShared_Pgss *shared);
static dsa_pointer qtext_store(const char *query, int query_len);
static void pgss_attach_query_text(PgStatShared_Pgss *shpgss,
								   dsa_pointer query_text,
								   int query_len, int encoding);
static void pgss_store_query_text(PgStatShared_Pgss *shared,
								  const char *query,
								  int query_location, int query_len,
								  int encoding, const JumbleState *jstate,
								  bool initialize_query_text);
static char *generate_normalized_query(const JumbleState *jstate,
									   const char *query,
									   int query_loc, int *query_len_p);

/*
 * Custom pgstat kind definition
 */
static const PgStat_KindInfo pgss_kind_info = {
	.name = "pg_stat_statements",
	.fixed_amount = false,
	.write_to_file = true,
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

static void
pgss_register_kind(void)
{
	pgstat_register_kind(PGSTAT_KIND_PGSS, &pgss_kind_info);
}

static void
pgss_init_shmem(void *ptr, void *arg)
{
	pgssSharedState *state = (pgssSharedState *) ptr;

	pg_atomic_init_u64(&state->entry_count, 0);
	pg_atomic_init_u64(&state->entry_dealloc, 0);
	pg_atomic_init_u64(&state->qtext_count, 0);
	pg_atomic_init_u64(&state->qtext_total_len, 0);
	pg_atomic_init_u64(&state->stats_reset, (uint64) GetCurrentTimestamp());
	pg_atomic_init_flag(&state->entry_evicting);
	LWLockInitialize(&state->qtext_gc_lock,
					 LWLockNewTrancheId("pg_stat_statements query text GC"));
}

static void
pgss_attach_shmem_cb(void)
{
	bool		found;

	if (!pgstat_get_kind_info(PGSTAT_KIND_PGSS))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via shared_preload_libraries")));

	if (pgss_shared == NULL)
		pgss_shared = GetNamedDSMSegment("pg_stat_statements_state",
										 sizeof(pgssSharedState),
										 pgss_init_shmem,
										 &found, NULL);

	if (pgss_qtext_dsa == NULL)
		pgss_qtext_dsa = GetNamedDSA("pg_stat_statements_qtext", &found);

	if (pgss_hash == NULL)
		pgss_hash = pgStatLocal.kind_hash[PGSTAT_KIND_PGSS];
}

static bool
pgss_stats_ready(void)
{
	return pgss_hash != NULL;
}

static void
pgss_require_ready(void)
{
	if (pgss_hash == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via shared_preload_libraries")));
}

static PgStatShared_Pgss *
pgss_shared_from_hash_entry(PgStatShared_HashEntry *p)
{
	return (PgStatShared_Pgss *)
		dsa_get_address(pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS], p->body);
}

static bool
pgss_free_query_text_locked(PgStatShared_Pgss *shared)
{
	bool		freed = false;

	Assert(LWLockHeldByMeInMode(&shared->header.lock, LW_EXCLUSIVE));

	if (DsaPointerIsValid(shared->query_text))
	{
		freed = true;
		pg_atomic_fetch_sub_u64(&pgss_shared->qtext_count, 1);

		if (shared->query_len >= 0)
			pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
									(uint64) shared->query_len);

		dsa_free(pgss_qtext_dsa, shared->query_text);
		shared->query_text = InvalidDsaPointer;
	}

	shared->query_len = -1;

	return freed;
}

static char *
pgss_query_text_address(PgStatShared_Pgss *shared)
{
	return dsa_get_address(pgss_qtext_dsa, shared->query_text);
}

static uint64
pgss_entry_count(void)
{
	return pg_atomic_read_u64(&pgss_shared->entry_count);
}

static uint64
pgss_entry_dealloc_count(void)
{
	return pg_atomic_read_u64(&pgss_shared->entry_dealloc);
}

static uint64
pgss_qtext_dsa_size(void)
{
	if (!pgss_qtext_dsa)
		return 0;

	return (uint64) dsa_get_total_size(pgss_qtext_dsa);
}

static uint64
pgss_qtext_count(void)
{
	return pg_atomic_read_u64(&pgss_shared->qtext_count);
}

static uint64
pgss_qtext_total_len(void)
{
	return pg_atomic_read_u64(&pgss_shared->qtext_total_len);
}

static uint64
pgss_entry_dsa_size(void)
{
	if (!pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS])
		return 0;

	return (uint64) dsa_get_total_size(pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS]);
}

static TimestampTz
pgss_stats_reset_time(void)
{
	return (TimestampTz) pg_atomic_read_u64(&pgss_shared->stats_reset);
}

static void
pgss_advance_calls_eviction(PgStatShared_Pgss *shared, uint64 calls)
{
	uint64		old_calls;

	old_calls = pg_atomic_read_u64(&shared->calls_eviction);
	while (old_calls < calls &&
		   !pg_atomic_compare_exchange_u64(&shared->calls_eviction,
										   &old_calls, calls))
		;
}

static void
pgss_mark_entry_live(PgStatShared_Pgss *shared)
{
	uint32		expected = 0;

	if (pg_atomic_compare_exchange_u32(&shared->entry_live, &expected, 1))
		pg_atomic_fetch_add_u64(&pgss_shared->entry_count, 1);
}

static void
pgss_mark_entry_dead(PgStatShared_Pgss *shared)
{
	uint32		expected = 1;

	if (pg_atomic_compare_exchange_u32(&shared->entry_live, &expected, 0))
		pg_atomic_fetch_sub_u64(&pgss_shared->entry_count, 1);
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
static PgStat_FlushResult
pgss_flush_pending_cb(PgStat_EntryRef *entry_ref, bool nowait,
					  bool xact_boundary)
{
	PgStat_PgssPending *pending;
	PgStatShared_Pgss *shared;

	pending = (PgStat_PgssPending *) entry_ref->pending;
	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return PGSTAT_FLUSH_LOCK_CONFLICT;

	pgss_flush_kind(&shared->counters, &pending->counters, PGSS_EXEC);

	if (pgss_track_planning && pending->counters.calls[PGSS_PLAN] > 0)
		pgss_flush_kind(&shared->counters, &pending->counters, PGSS_PLAN);

#define PGSS_ACCUM_COUNTER(item)		\
	(shared)->counters.item += (pending)->counters.item

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

	pgss_advance_calls_eviction(shared,
								(uint64) shared->counters.calls[PGSS_EXEC]);

	pgstat_unlock_entry(entry_ref);

	memset(pending, 0, sizeof(*pending));

	return PGSTAT_FLUSH_DONE;
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
	bool		save_entry = pgss_save;
	char	   *qtext = NULL;
	int			qtext_len = 0;
	bool		ok = true;

	if (!write_chunk_s(statfile, &save_entry))
		return false;

	if (!pgss_save)
		return true;

	LWLockAcquire(&shpgss->header.lock, LW_SHARED);

	if (ok && !write_chunk_s(statfile, &shpgss->key))
		ok = false;
	if (ok && !write_chunk_s(statfile, &shpgss->encoding))
		ok = false;
	if (ok && !write_chunk_s(statfile, &shpgss->stats_since))
		ok = false;
	if (ok && !write_chunk_s(statfile, &shpgss->minmax_stats_since))
		ok = false;

	/* Write query text */
	if (ok && DsaPointerIsValid(shpgss->query_text) && shpgss->query_len >= 0)
		qtext = dsa_get_address(pgss_qtext_dsa, shpgss->query_text);

	if (ok && qtext)
	{
		qtext_len = shpgss->query_len;
		if (!write_chunk_s(statfile, &qtext_len))
			ok = false;
		if (ok && !write_chunk(statfile, qtext, qtext_len + 1))
			ok = false;
	}
	else if (ok)
	{
		qtext_len = -1;
		if (!write_chunk_s(statfile, &qtext_len))
			ok = false;
	}

	LWLockRelease(&shpgss->header.lock);

	return ok;
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
	bool		saved_entry;
	dsa_pointer dp;
	int			qtext_len = -1;

	if (!read_chunk_s(statfile, &saved_entry))
		return false;

	if (!saved_entry)
		goto drop_entry;

	if (!read_chunk_s(statfile, &shpgss->key))
		return false;
	if (!read_chunk_s(statfile, &shpgss->encoding))
		return false;
	if (!read_chunk_s(statfile, &shpgss->stats_since))
		return false;
	if (!read_chunk_s(statfile, &shpgss->minmax_stats_since))
		return false;
	if (!read_chunk_s(statfile, &qtext_len))
		return false;

	dp = InvalidDsaPointer;

	if (qtext_len >= 0)
	{
		char	   *qtext;

		qtext = palloc(qtext_len + 1);
		if (!read_chunk(statfile, qtext, qtext_len + 1))
		{
			pfree(qtext);
			return false;
		}

		dp = qtext_store(qtext, qtext_len);

		pfree(qtext);
	}

	pg_atomic_init_u32(&shpgss->entry_live, 0);
	pg_atomic_init_u64(&shpgss->calls_eviction,
					   (uint64) shpgss->counters.calls[PGSS_EXEC]);
	pgss_mark_entry_live(shpgss);
	pgss_attach_query_text(shpgss, dp, qtext_len, shpgss->encoding);

	return true;

drop_entry:
	pgstat_drop_entry(PGSTAT_KIND_PGSS, key->dboid, key->objid, false);
	return true;
}

static void
pgss_accumulate_pending(PgStat_PgssPending *pending, const pgssHashKey *key,
						pgssStoreKind kind,
						double total_time, uint64 rows,
						const BufferUsage *bufusage,
						const WalUsage *walusage,
						const JitInstrumentation *jitusage,
						int parallel_workers_to_launch,
						int parallel_workers_launched,
						PlannedStmtOrigin planOrigin)
{
	Assert(kind == PGSS_PLAN || kind == PGSS_EXEC);

	pending->key = *key;

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

	if (bufusage)
	{
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
	}

	if (walusage)
	{
#define PGSS_ACCUM_WALUSAGE(item)		\
		pending->counters.item += walusage->item

		PGSS_ACCUM_WALUSAGE(wal_records);
		PGSS_ACCUM_WALUSAGE(wal_fpi);
		PGSS_ACCUM_WALUSAGE(wal_bytes);
		PGSS_ACCUM_WALUSAGE(wal_buffers_full);
#undef PGSS_ACCUM_WALUSAGE
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
		   bool toplevel,
		   pgssStoreKind kind,
		   double total_time, uint64 rows,
		   const BufferUsage *bufusage,
		   const WalUsage *walusage,
		   const JitInstrumentation *jitusage,
		   const JumbleState *jstate,
		   int parallel_workers_to_launch,
		   int parallel_workers_launched,
		   PlannedStmtOrigin planOrigin)
{
	pgssHashKey key;

	uint64		objid;
	PgStat_EntryRef *entry_ref;
	PgStatShared_Pgss *shared;
	bool		created_entry = false;

	Assert(query != NULL);

	if (queryId == INT64CONST(0))
		return;

	memset(&key, 0, sizeof(pgssHashKey));
	key.userid = GetUserId();
	key.dbid = MyDatabaseId;
	key.queryid = queryId;
	key.toplevel = toplevel;

	objid = hash_bytes_extended((const unsigned char *) &key,
								sizeof(pgssHashKey), 0);

	entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, key.dbid, objid,
									 false, NULL);
	if (!entry_ref)
	{
		/*
		 * XXX: This is a soft high-water mark.  Concurrent miss creators can
		 * all observe a count below the threshold and briefly admit query
		 * texts past it.  A hard cap would need an atomic query-text
		 * reservation protocol on the admission path.
		 */
		if (pgss_qtext_count() >=
			PGSS_QTEXT_EVICT_THRESHOLD() &&
			!pgss_evict_for_entry())
			return;

		entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, key.dbid, objid,
										 true, &created_entry);
	}

	if (!entry_ref)
		return;

	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (created_entry)
	{
		int			encoding = GetDatabaseEncoding();

		pgstat_lock_entry(entry_ref, false);
		shared->key = key;
		pg_atomic_init_u32(&shared->entry_live, 0);
		pg_atomic_init_u64(&shared->calls_eviction, 0);
		shared->stats_since = GetCurrentTimestamp();
		shared->minmax_stats_since = shared->stats_since;
		pgss_mark_entry_live(shared);
		pgstat_unlock_entry(entry_ref);

		pgss_store_query_text(shared, query, query_location, query_len,
							  encoding, jstate, true);
	}

	if (!created_entry)
	{
		bool		metadata_valid;

		/*
		 * Another backend can create the pgstat entry and reach this point
		 * before it has initialized the metadata.  Don't record counters
		 * until the entry is ready; query text itself is allowed to be
		 * absent.
		 */
		pgstat_lock_entry_shared(entry_ref, false);
		metadata_valid = (shared->stats_since != 0);
		pgstat_unlock_entry(entry_ref);

		if (!metadata_valid)
			return;

		if (!pgss_has_query_text(shared))
			pgss_store_query_text(shared, query, query_location, query_len,
								  GetDatabaseEncoding(), jstate, false);
	}

	pgstat_prep_pending_from_entry_ref(entry_ref);

	if (!jstate)
	{
		pgss_accumulate_pending((PgStat_PgssPending *) entry_ref->pending,
								&key, kind, total_time, rows, bufusage,
								walusage, jitusage,
								parallel_workers_to_launch,
								parallel_workers_launched, planOrigin);

		if (kind == PGSS_EXEC)
			pg_atomic_fetch_add_u64(&shared->calls_eviction, 1);
	}
}

static void
pgss_reset_timestamp_cb(PgStatShared_Common *header, TimestampTz ts)
{
	PgStatShared_Pgss *shared = (PgStatShared_Pgss *) header;

	pg_atomic_write_u64(&shared->calls_eviction, 0);
	shared->stats_since = ts;
	shared->minmax_stats_since = ts;
}

static bool
pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssResetFilter *filter = (pgssResetFilter *) DatumGetPointer(match_data);
	PgStatShared_Pgss *shared;

	if (p->key.kind != PGSTAT_KIND_PGSS)
		return false;

	shared = pgss_shared_from_hash_entry(p);

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

	shared = pgss_shared_from_hash_entry(p);

	pgss_free_query_text(shared);
	pgss_mark_entry_dead(shared);

	return true;
}

static void
pgss_free_query_text(PgStatShared_Pgss *shared)
{
	LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
	pgss_free_query_text_locked(shared);
	LWLockRelease(&shared->header.lock);
}

static void
pgss_drop_entry(Oid dbid, uint64 objid)
{
	PgStat_EntryRef *entry_ref;
	bool		freed;

	entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, dbid, objid,
									 false, NULL);
	if (entry_ref)
	{
		PgStatShared_Pgss *shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

		pgss_free_query_text(shared);
		pgss_mark_entry_dead(shared);
	}

	freed = pgstat_drop_entry(PGSTAT_KIND_PGSS, dbid, objid, true);
	if (!freed)
		pgstat_request_entry_refs_gc();
}

static bool
pgss_entry_eviction_try_start(void)
{
	return pg_atomic_test_set_flag(&pgss_shared->entry_evicting);
}

static void
pgss_entry_eviction_finish(void)
{
	pg_atomic_clear_flag(&pgss_shared->entry_evicting);
}

static TimestampTz
pgss_entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only)
{
	TimestampTz stats_reset;
	pgssResetFilter filter;

	stats_reset = GetCurrentTimestamp();

	filter.userid = userid;
	filter.dbid = dbid;
	filter.queryid = queryid;

	/*
	 * The core pgstat infrastructure only supports full entry resets (zeroing
	 * the entire data region).  For minmax_only we need a partial reset, so
	 * scan and update the entries here.
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

			shared = pgss_shared_from_hash_entry(p);

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
		objid = hash_bytes_extended((const unsigned char *) &key,
									sizeof(pgssHashKey), 0);
		pgss_drop_entry(key.dbid, objid);

		key.toplevel = true;
		objid = hash_bytes_extended((const unsigned char *) &key,
									sizeof(pgssHashKey), 0);
		pgss_drop_entry(key.dbid, objid);
	}
	else
	{
		pgstat_drop_matching_entries(pgss_drop_matching_entry,
									 PointerGetDatum(&filter));
	}

	if (!userid && !dbid && !queryid)
	{
		pg_atomic_write_u64(&pgss_shared->entry_dealloc, 0);
		pg_atomic_write_u64(&pgss_shared->entry_count, 0);
		pg_atomic_write_u64(&pgss_shared->qtext_count, 0);
		pg_atomic_write_u64(&pgss_shared->qtext_total_len, 0);
		pg_atomic_write_u64(&pgss_shared->stats_reset, (uint64) stats_reset);
	}

	return stats_reset;
}

static bool
pgss_has_query_text(PgStatShared_Pgss *shared)
{
	bool		has_query_text;

	LWLockAcquire(&shared->header.lock, LW_SHARED);
	has_query_text = DsaPointerIsValid(shared->query_text);
	LWLockRelease(&shared->header.lock);

	return has_query_text;
}

/*
 * Given a query string (not necessarily null-terminated), allocate space in
 * the DSA query text area and store the string there.
 *
 * If query-text allocation fails, return InvalidDsaPointer.
 */
static dsa_pointer
qtext_store(const char *query, int query_len)
{
	dsa_pointer dp = InvalidDsaPointer;
	Size		query_size = (Size) query_len + 1;

	dp = dsa_allocate_extended(pgss_qtext_dsa, query_size, DSA_ALLOC_NO_OOM);

	if (DsaPointerIsValid(dp))
	{
		char	   *dst = dsa_get_address(pgss_qtext_dsa, dp);

		memcpy(dst, query, query_len);
		dst[query_len] = '\0';
		pg_atomic_fetch_add_u64(&pgss_shared->qtext_count, 1);
		pg_atomic_fetch_add_u64(&pgss_shared->qtext_total_len,
								(uint64) query_len);
	}

	return dp;
}

static void
pgss_attach_query_text(PgStatShared_Pgss *shpgss, dsa_pointer query_text,
					   int query_len, int encoding)
{
	if (!DsaPointerIsValid(query_text))
		query_len = -1;

	LWLockAcquire(&shpgss->header.lock, LW_EXCLUSIVE);

	/*
	 * Query text is allocated before taking the entry lock.  A concurrent
	 * eviction can mark the entry dead in that window while this backend
	 * still holds a pgstat ref, so don't attach text that PGSS can no longer
	 * reach.
	 */
	if (pg_atomic_read_u32(&shpgss->entry_live) == 0)
	{
		if (DsaPointerIsValid(query_text))
		{
			pg_atomic_fetch_sub_u64(&pgss_shared->qtext_count, 1);

			if (query_len >= 0)
				pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
										(uint64) query_len);

			dsa_free(pgss_qtext_dsa, query_text);
		}

		LWLockRelease(&shpgss->header.lock);
		return;
	}

	if (DsaPointerIsValid(shpgss->query_text))
		pgss_free_query_text_locked(shpgss);

	shpgss->query_text = query_text;
	shpgss->query_len = query_len;
	shpgss->encoding = encoding;

	if (DsaPointerIsValid(query_text))
		Assert(query_len >= 0);

	LWLockRelease(&shpgss->header.lock);
}

static void
pgss_store_query_text(PgStatShared_Pgss *shared, const char *query,
					  int query_location, int query_len, int encoding,
					  const JumbleState *jstate, bool initialize_query_text)
{
	char	   *norm_query = NULL;
	dsa_pointer query_text;

	query = CleanQuerytext(query, &query_location, &query_len);
	if (jstate && jstate->clocations_count > 0)
		norm_query = generate_normalized_query(jstate, query,
											   query_location,
											   &query_len);

	query_text = qtext_store(norm_query ? norm_query : query, query_len);
	if (DsaPointerIsValid(query_text) || initialize_query_text)
		pgss_attach_query_text(shared, query_text, query_len, encoding);

	if (norm_query)
		pfree(norm_query);
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
	StringInfoData norm_query;
	int			query_len = *query_len_p;
	int			len_to_wrt,		/* Length (in bytes) to write */
				quer_loc = 0,	/* Source query byte location */
				last_off = 0,	/* Offset from start for previous tok */
				last_tok_len = 0;	/* Length (in bytes) of that tok */
	int			num_constants_replaced = 0;
	LocationLen *locs = NULL;

	/*
	 * Our output buffer is an expansible StringInfo, but avoid enlarging it
	 * in most cases by reserving extra space for each constant location.
	 */
	Assert(jstate->clocations_count > 0);
	initStringInfoExt(&norm_query, query_len + jstate->clocations_count * 10);

	/*
	 * Determine constants' lengths (core system only gives us locations), and
	 * return a sorted copy of jstate's LocationLen data with lengths filled
	 * in.
	 */
	locs = ComputeConstantLengths(jstate, query, query_loc);

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
		appendBinaryStringInfo(&norm_query, query + quer_loc, len_to_wrt);

		/*
		 * And insert a param symbol in place of the constant token; and, if
		 * we have a squashable list, insert a placeholder comment starting
		 * from the list's second value.
		 */
		appendStringInfo(&norm_query, "$%d%s",
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
	appendBinaryStringInfo(&norm_query, query + quer_loc, len_to_wrt);

	*query_len_p = norm_query.len;
	return norm_query.data;
}

static void
pgss_gc_query_texts(void)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *p;
	uint64		threshold;

	if (!pgss_hash)
		return;

	/*
	 * Normal PGSS eviction/drop paths free query text immediately, and attach
	 * refuses to install text on already-dead entries.  This covers less
	 * direct cases, such as dead entries still reachable through pgstat refs
	 * or entries dropped by core paths that don't call back into
	 * PGSS-specific cleanup.
	 */
	threshold = PGSS_QTEXT_EVICT_THRESHOLD();
	if (pgss_qtext_count() < threshold)
		return;

	LWLockAcquire(&pgss_shared->qtext_gc_lock, LW_EXCLUSIVE);

	if (pgss_qtext_count() < threshold)
	{
		LWLockRelease(&pgss_shared->qtext_gc_lock);
		return;
	}

	dshash_seq_init(&hstat, pgss_hash, false);
	while ((p = dshash_seq_next(&hstat)) != NULL)
	{
		PgStatShared_Pgss *shared;

		shared = pgss_shared_from_hash_entry(p);

		if (!p->dropped &&
			pg_atomic_read_u32(&shared->entry_live) != 0)
			continue;

		LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
		if (p->dropped ||
			pg_atomic_read_u32(&shared->entry_live) == 0)
			pgss_free_query_text_locked(shared);
		LWLockRelease(&shared->header.lock);
	}
	dshash_seq_term(&hstat);

	LWLockRelease(&pgss_shared->qtext_gc_lock);
}

typedef struct pgssEvictCandidate
{
	Oid			dbid;
	uint64		objid;
	uint64		calls;
}			pgssEvictCandidate;

static int	pgss_evict_candidate_cmp(const void *a, const void *b);

static int
pgss_evict_candidate_cmp(const void *a, const void *b)
{
	const		pgssEvictCandidate *left = (const pgssEvictCandidate *) a;
	const		pgssEvictCandidate *right = (const pgssEvictCandidate *) b;

	if (left->calls < right->calls)
		return -1;
	if (left->calls > right->calls)
		return 1;
	return 0;
}

/*
 * Try to reduce query-text pressure by evicting low-frequency entries from
 * pg_stat_statements' dedicated pgstat storage.
 *
 * Victim selection scans all eligible hash entries and sorts a backend-local
 * candidate array by flushed execution-call score.  Enough low-frequency
 * entries are evicted to bring retained query texts back near pgss_max, and
 * the candidate array is discarded after the eviction attempt.
 *
 * The score may briefly lag the SQL-visible calls counter, but eviction does
 * not need an exact real-time ranking.  Reading the atomic score without
 * taking each entry's stats-body lock keeps PGSS from acquiring another lock
 * for every hash entry scanned under storage pressure.
 */
static bool
pgss_evict_for_entry(void)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *p;
	pgssEvictCandidate *candidates = NULL;
	uint64		entry_count;
	uint64		qtext_count;
	uint64		threshold;
	int			capacity;
	int			target;
	int			ncandidates = 0;
	bool		evicted = false;
	bool		oom = false;

	if (!pgss_hash)
		return false;

	/*
	 * Only one backend should perform the full-hash victim selection for a
	 * query-text pressure event.  Concurrent backends skip eviction rather
	 * than waiting; their current observation is dropped while query-text
	 * storage is above the high-water mark.
	 */
	if (!pgss_entry_eviction_try_start())
		return false;

	threshold = PGSS_QTEXT_EVICT_THRESHOLD();
	qtext_count = pgss_qtext_count();
	if (qtext_count < threshold)
		goto done;

	entry_count = pgss_entry_count();
	if (entry_count == 0)
		goto done;

	if (entry_count > (uint64) Min(MaxAllocSize / sizeof(pgssEvictCandidate),
								   INT_MAX))
		goto done;

	capacity = (int) entry_count;
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

		if (ncandidates == capacity)
		{
			pgssEvictCandidate *newcandidates;
			int			newcapacity;

			if (capacity > Min(MaxAllocSize / sizeof(pgssEvictCandidate),
							   INT_MAX) / 2)
			{
				oom = true;
				break;
			}

			newcapacity = capacity * 2;
			newcandidates = repalloc_extended(candidates,
											  (Size) newcapacity *
											  sizeof(pgssEvictCandidate),
											  MCXT_ALLOC_NO_OOM);
			if (!newcandidates)
			{
				oom = true;
				break;
			}

			candidates = newcandidates;
			capacity = newcapacity;
		}

		candidates[ncandidates].dbid = p->key.dboid;
		candidates[ncandidates].objid = p->key.objid;
		candidates[ncandidates].calls =
			pg_atomic_read_u64(&shared->calls_eviction);
		ncandidates++;
	}
	dshash_seq_term(&hstat);

	if (oom || ncandidates == 0)
		goto done;

	if (qtext_count <= (uint64) pgss_max)
		goto done;

	target = (int) Min(qtext_count - (uint64) pgss_max,
					   (uint64) ncandidates);
	if (target <= 0)
		goto done;

	qsort(candidates, ncandidates, sizeof(pgssEvictCandidate),
		  pgss_evict_candidate_cmp);

	for (int i = 0; i < target; i++)
		pgss_drop_entry(candidates[i].dbid, candidates[i].objid);

	pg_atomic_fetch_add_u64(&pgss_shared->entry_dealloc, 1);

	if (pgss_qtext_count() >= threshold)
		pgss_gc_query_texts();

	evicted = true;

done:
	if (candidates)
		pfree(candidates);
	pgss_entry_eviction_finish();

	return evicted;
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
	if (!pgss_stats_ready() || !pgss_enabled(nesting_level))
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
				   nesting_level == 0,
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
	bool		track_planning;
	instr_time	start;
	instr_time	duration;
	BufferUsage bufusage_start,
				bufusage;
	WalUsage	walusage_start,
				walusage;

	/*
	 * We can't process the query if no query_string is provided, as
	 * pgss_store needs it.  We also ignore query without queryid, as it would
	 * be treated as a utility statement, which may not be the case.
	 */
	track_planning = pgss_enabled(nesting_level) &&
		pgss_track_planning && query_string &&
		parse->queryId != INT64CONST(0);

	if (track_planning)
	{
		/* We need to track buffer usage as the planner can access them. */
		bufusage_start = pgBufferUsage;

		/*
		 * Similarly the planner could write some WAL records in some cases
		 * (e.g. setting a hint bit with those being WAL-logged)
		 */
		walusage_start = pgWalUsage;
		INSTR_TIME_SET_CURRENT(start);
	}

	/* Preserve existing nesting behavior. */
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

	if (track_planning)
	{
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
				   nesting_level == 0,
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
				   nesting_level == 0,
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
	bool		track_utility;
	bool		bump_level;
	instr_time	start;
	instr_time	duration;
	uint64		rows;
	BufferUsage bufusage_start,
				bufusage;
	WalUsage	walusage_start,
				walusage;

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
	/* Preserve existing nesting behavior, evaluating the condition once. */
	bump_level =
		!IsA(parsetree, ExecuteStmt) &&
		!IsA(parsetree, PrepareStmt);
	track_utility = enabled && bump_level;

	if (track_utility)
	{
		bufusage_start = pgBufferUsage;
		walusage_start = pgWalUsage;
		INSTR_TIME_SET_CURRENT(start);
	}

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

	if (track_utility)
	{
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
				   nesting_level == 0,
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

	pgss_require_ready();

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = PG_GETARG_INT64(2);

	pgss_entry_reset(userid, dbid, queryid, false);

	PG_RETURN_VOID();
}

Datum
pg_stat_statements_reset_1_11(PG_FUNCTION_ARGS)
{
	Oid			userid;
	Oid			dbid;
	int64		queryid;
	bool		minmax_only;

	pgss_require_ready();

	userid = PG_GETARG_OID(0);
	dbid = PG_GETARG_OID(1);
	queryid = PG_GETARG_INT64(2);
	minmax_only = PG_GETARG_BOOL(3);

	PG_RETURN_TIMESTAMPTZ(pgss_entry_reset(userid, dbid, queryid, minmax_only));
}

/*
 * Reset statement statistics.
 */
Datum
pg_stat_statements_reset(PG_FUNCTION_ARGS)
{
	pgss_require_ready();

	pgss_entry_reset(0, 0, 0, false);

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

	pgss_require_ready();

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

		shared = pgss_shared_from_hash_entry(p);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
			continue;

		LWLockAcquire(&shared->header.lock, LW_SHARED);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
		{
			LWLockRelease(&shared->header.lock);
			continue;
		}
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
				int			query_col = i++;

				LWLockAcquire(&shared->header.lock, LW_SHARED);
				if (DsaPointerIsValid(shared->query_text) &&
					shared->query_len >= 0)
				{
					char	   *qstr = pgss_query_text_address(shared);
					char	   *enc = pg_any_to_server(qstr, shared->query_len, shared->encoding);

					values[query_col] = CStringGetTextDatum(enc);
					if (enc != qstr)
						pfree(enc);
				}
				else
					nulls[query_col] = true;
				LWLockRelease(&shared->header.lock);
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
#define PG_STAT_STATEMENTS_INFO_COLS	6

/*
 * Return statistics of pg_stat_statements.
 */
Datum
pg_stat_statements_info(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[PG_STAT_STATEMENTS_INFO_COLS] = {0};
	bool		nulls[PG_STAT_STATEMENTS_INFO_COLS] = {0};

	pgss_require_ready();

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	switch (tupdesc->natts)
	{
		case 2:
			values[0] = Int64GetDatum((int64) pgss_entry_dealloc_count());
			values[1] = TimestampTzGetDatum(pgss_stats_reset_time());
			break;
		case PG_STAT_STATEMENTS_INFO_COLS:
			values[0] = Int64GetDatum((int64) pgss_entry_dealloc_count());
			values[1] = Int64GetDatum((int64) pgss_entry_dsa_size());
			values[2] = Int64GetDatum((int64) pgss_qtext_dsa_size());
			values[3] = Int64GetDatum((int64) pgss_qtext_count());
			values[4] = Int64GetDatum((int64) pgss_qtext_total_len());
			values[5] = TimestampTzGetDatum(pgss_stats_reset_time());
			break;
		default:
			elog(ERROR, "incorrect number of output arguments");
	}

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}
