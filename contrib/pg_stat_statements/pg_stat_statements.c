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
 * For display, representative query strings replace constants with parameter
 * symbols.  The strings live in a dedicated DSA area and pgstat entries keep
 * their DSA pointers.
 *
 * Note about locking issues: an entry's metadata (key, query text pointer,
 * timestamps) and its counters are protected by the entry's pgstat LWLock.
 * Counters are only modified by the pending-stats flush callback.  The usage
 * score, sticky flag and live marker are atomics updated without the entry
 * lock.  Eviction is serialized by pgss_shared->entry_evict_lock, acquired
 * conditionally so that a single backend pays for victim selection while the
 * others continue to admit entries.
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
 * Statement identity.  We separate queries by user and by database even if
 * they are otherwise identical.
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
 * Stats counters stored in the custom pgstat kind.
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
 */
typedef struct PgStatShared_Pgss
{
	PgStatShared_Common header;
	pgssHashKey key;
	dsa_pointer query_text;		/* query text, or InvalidDsaPointer */
	int			query_len;		/* # of valid bytes in query text */
	int			encoding;		/* query text encoding */
	TimestampTz stats_since;	/* timestamp of entry allocation */
	TimestampTz minmax_stats_since; /* timestamp of last min/max values reset */
	pg_atomic_uint32 entry_live;	/* entry is visible to PGSS */
	pg_atomic_uint32 sticky;	/* no planning or execution recorded yet */
	pg_atomic_uint64 usage;		/* eviction score, fixed-point */
	Counters	counters;
} PgStatShared_Pgss;

StaticAssertDecl(offsetof(PgStatShared_Pgss, usage) ==
				 MAXALIGN(offsetof(PgStatShared_Pgss, usage)),
				 "PgStatShared_Pgss.usage must be maxaligned");

/* Read/write helpers for stats file serialization, as in pgstat.c */
#define pgss_write_chunk(fpout, ptr, len) (fwrite(ptr, len, 1, fpout) == 1)
#define pgss_write_chunk_s(fpout, ptr) pgss_write_chunk(fpout, ptr, sizeof(*ptr))
#define pgss_read_chunk(fpin, ptr, len) (fread(ptr, 1, len, fpin) == (len))
#define pgss_read_chunk_s(fpin, ptr) pgss_read_chunk(fpin, ptr, sizeof(*ptr))

/*
 * Eviction starts once retained query texts or entries exceed pgss_max by
 * this much, and never evicts more than this fraction of the live entries in
 * one pass.
 */
#define PGSS_EVICT_THRESHOLD_PERCENT	110
#define PGSS_EVICT_PASS_PERCENT		10
#define PGSS_QUERY_TEXT_MEMORY_DEFAULT	4096

/*
 * Extra slots for arrays sized from the live-entry gauge.  Entries can be
 * admitted while scans run, so this avoids immediate repalloc for small drift.
 */
#define PGSS_ENTRY_SCAN_HEADROOM	16

/*
 * Global shared state
 *
 * Kept in a named DSM segment: a stats kind is either fixed- or
 * variable-numbered, and PGSTAT_KIND_PGSS is the latter.
 */
typedef struct pgssSharedState
{
	pg_atomic_uint64 entry_dealloc; /* # of entry eviction passes */
	pg_atomic_uint64 entry_count;	/* # of live entries */
	pg_atomic_uint64 qtext_count;	/* # of retained query texts */
	pg_atomic_uint64 qtext_total_len;	/* logical bytes of query text */
	pg_atomic_uint64 stats_reset;	/* timestamp with all stats reset */
	pg_atomic_uint64 cur_median_usage;	/* current median usage of entries */
	LWLock		entry_evict_lock;	/* serializes entry eviction */
} pgssSharedState;

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

typedef struct pgssRestoredText
{
	dsa_pointer text;
	int			len;
} pgssRestoredText;

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

/* Eviction victim selection */
typedef struct pgssEvictCandidate
{
	uint64		objid;
	uint64		usage;
	int			query_len;
	bool		has_query_text;
} pgssEvictCandidate;

typedef struct pgssEvictContext
{
	pgssEvictCandidate *candidates;
	int			target;
} pgssEvictContext;

/*---- Local variables ----*/

/* Current nesting depth of planner/ExecutorRun/ProcessUtility calls */
static int	nesting_level = 0;

/* Links to shared memory state */
static pgssSharedState *pgss_shared = NULL;
static dsa_area *pgss_qtext_dsa = NULL;
static dshash_table *pgss_hash = NULL;
static pgssRestoredText *pgss_restored_texts = NULL;
static int	pgss_restored_ntexts = 0;
static int	pgss_restored_maxtexts = 0;
static bool pgss_restored_entries = false;


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
static int	pgss_query_text_memory = PGSS_QUERY_TEXT_MEMORY_DEFAULT;	/* in KB */

#define pgss_enabled(level) \
	(!IsParallelWorker() && \
	(pgss_track == PGSS_TRACK_ALL || \
	(pgss_track == PGSS_TRACK_TOP && (level) == 0)))

/*
 * Number of retained query texts or entries at which eviction starts.  Always
 * above pgss_max, so very small settings still get some headroom.
 */
static inline uint64
pgss_evict_threshold(void)
{
	return Max((uint64) pgss_max + 1,
			   (uint64) pgss_max * PGSS_EVICT_THRESHOLD_PERCENT / 100);
}

static inline uint64
pgss_query_text_memory_bytes(void)
{
	return (uint64) pgss_query_text_memory * 1024;
}

static inline bool
pgss_query_text_memory_full(void)
{
	return pg_atomic_read_u64(&pgss_shared->qtext_total_len) >=
		pgss_query_text_memory_bytes();
}

static inline bool
pgss_admission_pressure(uint64 threshold)
{
	return pg_atomic_read_u64(&pgss_shared->qtext_count) >= threshold ||
		pg_atomic_read_u64(&pgss_shared->entry_count) >= threshold ||
		pgss_query_text_memory_full();
}

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
static void pgss_require_ready(void);
static PgStatShared_Pgss *pgss_shared_from_hash_entry(PgStatShared_HashEntry *p);
static void pgss_mark_entry_live(PgStatShared_Pgss *shared);
static bool pgss_mark_entry_dead(PgStatShared_Pgss *shared);
static void pgss_evict_for_entry(void);
static void pgss_free_dsa_query_text(dsa_pointer query_text, int query_len);
static uint64 pgss_dsa_size(dsa_area *dsa);
static TimestampTz pgss_entry_reset(Oid userid, Oid dbid, int64 queryid,
									bool minmax_only);
static void pgss_store(const char *query, int64 queryId,
					   int query_location, int query_len,
					   pgssStoreKind kind,
					   double total_time, uint64 rows,
					   const BufferUsage *bufusage,
					   const WalUsage *walusage,
					   const JitInstrumentation *jitusage,
					   const JumbleState *jstate,
					   int parallel_workers_to_launch,
					   int parallel_workers_launched,
					   PlannedStmtOrigin planOrigin);

static void pgss_init_shmem(void *ptr, void *arg);
static void pgss_attach_shmem_cb(void);
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
static bool pgss_remember_restored_text(dsa_pointer text, int len);
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
static bool pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data);
static void pgss_drop_entry_and_text(PgStatShared_Pgss *shared);
static void pgss_free_query_text_locked(PgStatShared_Pgss *shared);
static bool pgss_drop_evicted_entry(PgStatShared_HashEntry *p, Datum match_data);
static void pgss_drop_entry(uint64 objid);
static bool pgss_reserve_query_text(int query_len);
static dsa_pointer qtext_store(const char *query, int query_len);
static void pgss_attach_query_text(PgStatShared_Pgss *shpgss,
								   dsa_pointer query_text,
								   int query_len, int encoding);
static dsa_pointer pgss_store_query_text(const char *query,
										 int query_location,
										 int *query_len,
										 const JumbleState *jstate);
static char *generate_normalized_query(const JumbleState *jstate,
									   const char *query,
									   int query_loc, int *query_len_p);

static const PgStat_KindInfo pgss_kind_info = {
	.name = "pg_stat_statements",
	.fixed_amount = false,
	.write_to_file = true,
	.accessed_across_databases = true,
	.own_hash = true,
	.shared_size = sizeof(PgStatShared_Pgss),
	.shared_data_off = offsetof(PgStatShared_Pgss, counters),
	.shared_data_len = sizeof(Counters),
	.pending_size = sizeof(PgStat_PgssPending),
	.init_backend_cb = pgss_attach_shmem_cb,
	.flush_pending_cb = pgss_flush_pending_cb,
	.to_serialized_data = pgss_to_serialized_data,
	.from_serialized_data = pgss_from_serialized_data,
	.finish = pgss_stats_file_finish,
};


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
							"Sets the target number of statement entries retained by pg_stat_statements.",
							"The least-used entries are evicted once this target is exceeded by 10 percent.",
							&pgss_max,
							5000,
							1,
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

	DefineCustomIntVariable("pg_stat_statements.query_text_memory",
							"Sets the maximum amount of query text kept by pg_stat_statements.",
							"New statement entries are not admitted while this limit is reached.",
							&pgss_query_text_memory,
							PGSS_QUERY_TEXT_MEMORY_DEFAULT,
							1,
							MAX_KILOBYTES,
							PGC_SIGHUP,
							GUC_UNIT_KB,
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
 * uses the cumulative statistics system, so pgss_shared and pgss_qtext_dsa
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

	if (pgss_qtext_dsa == NULL)
		pgss_qtext_dsa = GetNamedDSA("pg_stat_statements_qtext", &found);

	if (pgss_hash == NULL)
		pgss_hash = pgStatLocal.kind_hash[PGSTAT_KIND_PGSS];
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

static void
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

static void
pgss_free_dsa_query_text(dsa_pointer query_text, int query_len)
{
	if (!DsaPointerIsValid(query_text))
		return;

	Assert(query_len >= 0);

	pg_atomic_fetch_sub_u64(&pgss_shared->qtext_count, 1);
	pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
							(uint64) query_len);

	dsa_free(pgss_qtext_dsa, query_text);
}

static void
pgss_free_query_text_locked(PgStatShared_Pgss *shared)
{
	Assert(LWLockHeldByMeInMode(&shared->header.lock, LW_EXCLUSIVE));

	pgss_free_dsa_query_text(shared->query_text, shared->query_len);
	shared->query_text = InvalidDsaPointer;
	shared->query_len = 0;
}

static uint64
pgss_dsa_size(dsa_area *dsa)
{
	if (!dsa)
		return 0;

	return (uint64) dsa_get_total_size(dsa);
}

static bool
pgss_reserve_query_text(int query_len)
{
	uint64		limit = pgss_query_text_memory_bytes();
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
		if (pg_atomic_read_u32(&shared->sticky) != 0)
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

	/* Pending entry was flushed and retained; nothing new. */
	if (pending->counters.calls[PGSS_PLAN] == 0 &&
		pending->counters.calls[PGSS_EXEC] == 0)
		return PGSTAT_FLUSH_DONE;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return PGSTAT_FLUSH_LOCK_CONFLICT;

	for (int kind = 0; kind < PGSS_NUMKIND; kind++)
	{
		if (pending->counters.calls[kind] > 0)
			pgss_flush_kind(&shared->counters, &pending->counters,
							(pgssStoreKind) kind,
							shared->minmax_stats_since,
							pending->minmax_since[kind]);
	}

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

	save_entry = pg_atomic_read_u32(&shpgss->entry_live) != 0;
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
	qtext_len = DsaPointerIsValid(shpgss->query_text) ? shpgss->query_len : -1;

	ok = pgss_write_chunk_s(statfile, &shpgss->key) &&
		pgss_write_chunk_s(statfile, &shpgss->encoding) &&
		pgss_write_chunk_s(statfile, &shpgss->stats_since) &&
		pgss_write_chunk_s(statfile, &shpgss->minmax_stats_since) &&
		pgss_write_chunk_s(statfile, &entry_dealloc) &&
		pgss_write_chunk_s(statfile, &stats_reset) &&
		pgss_write_chunk_s(statfile, &usage) &&
		pgss_write_chunk_s(statfile, &qtext_len);

	if (ok && qtext_len >= 0)
		ok = pgss_write_chunk(statfile,
							  dsa_get_address(pgss_qtext_dsa, shpgss->query_text),
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

	pgss_restored_entries = true;

	if (!pgss_read_chunk_s(statfile, &shpgss->key) ||
		!pgss_read_chunk_s(statfile, &shpgss->encoding) ||
		!pgss_read_chunk_s(statfile, &shpgss->stats_since) ||
		!pgss_read_chunk_s(statfile, &shpgss->minmax_stats_since) ||
		!pgss_read_chunk_s(statfile, &entry_dealloc) ||
		!pgss_read_chunk_s(statfile, &stats_reset) ||
		!pgss_read_chunk_s(statfile, &usage) ||
		!pgss_read_chunk_s(statfile, &qtext_len))
		return false;

	/* Sanity-check what we can; the rest is opaque. */
	if (!PG_VALID_BE_ENCODING(shpgss->encoding) ||
		(shpgss->key.toplevel != false && shpgss->key.toplevel != true) ||
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
			dp = qtext_store(qtext, qtext_len);
			pfree(qtext);
		}
	}

	/* Nobody else can see the entry yet, so plain initialization is fine. */
	pg_atomic_init_u32(&shpgss->sticky,
					   (shpgss->counters.calls[PGSS_PLAN] +
						shpgss->counters.calls[PGSS_EXEC]) == 0 ? 1 : 0);
	pg_atomic_init_u64(&shpgss->usage, usage);
	pg_atomic_init_u32(&shpgss->entry_live, 0);
	pgss_mark_entry_live(shpgss);
	pg_atomic_write_u64(&pgss_shared->entry_dealloc, entry_dealloc);
	pg_atomic_write_u64(&pgss_shared->stats_reset, stats_reset);
	if (DsaPointerIsValid(dp))
	{
		if (pgss_remember_restored_text(dp, qtext_len))
			pgss_attach_query_text(shpgss, dp, qtext_len, shpgss->encoding);
		else
			pgss_free_dsa_query_text(dp, qtext_len);
	}

	return true;
}

/*
 * Remember a query text restored from the stats file, so that it can be
 * released if core later discards the file.
 */
static bool
pgss_remember_restored_text(dsa_pointer text, int len)
{
	if (!DsaPointerIsValid(text))
		return true;

	if (pgss_restored_ntexts >= pgss_restored_maxtexts)
	{
		int			newmax = Max(pgss_restored_maxtexts * 2, 1024);
		Size		newsize;

		if (!AllocSizeIsValid((Size) newmax * sizeof(pgssRestoredText)))
			return false;
		newsize = (Size) newmax * sizeof(pgssRestoredText);
		if (pgss_restored_texts == NULL)
			pgss_restored_texts =
				MemoryContextAllocExtended(TopMemoryContext, newsize,
										   MCXT_ALLOC_NO_OOM);
		else
		{
			pgssRestoredText *newtexts;

			newtexts = repalloc_extended(pgss_restored_texts, newsize,
										 MCXT_ALLOC_NO_OOM);
			if (newtexts == NULL)
				return false;
			pgss_restored_texts = newtexts;
		}

		if (pgss_restored_texts == NULL)
			return false;

		pgss_restored_maxtexts = newmax;
	}

	pgss_restored_texts[pgss_restored_ntexts].text = text;

	pgss_restored_texts[pgss_restored_ntexts].len = len;
	pgss_restored_ntexts++;

	return true;
}

/*
 * Stats file read is over.  On discard, core has dropped the restored
 * entries; release their query texts and fix up the query-text gauges.
 */
static void
pgss_stats_file_finish(PgStat_StatsFileOp status)
{
	if (status == STATS_DISCARD && pgss_shared != NULL)
	{
		for (int i = 0; i < pgss_restored_ntexts; i++)
			pgss_free_dsa_query_text(pgss_restored_texts[i].text,
									 pgss_restored_texts[i].len);

		if (pgss_restored_entries)
		{
			pg_atomic_write_u64(&pgss_shared->entry_dealloc, 0);
			pg_atomic_write_u64(&pgss_shared->entry_count, 0);
			pg_atomic_write_u64(&pgss_shared->stats_reset,
								(uint64) GetCurrentTimestamp());
			pg_atomic_write_u64(&pgss_shared->cur_median_usage,
								PGSS_USAGE_MEDIAN_INIT);
		}
	}

	if (pgss_restored_texts != NULL)
		pfree(pgss_restored_texts);
	pgss_restored_texts = NULL;
	pgss_restored_ntexts = 0;
	pgss_restored_maxtexts = 0;
	pgss_restored_entries = false;
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
 *
 * Another backend may be creating the same entry; if its metadata is not yet
 * initialized we skip this store rather than wait.
 */
static void
pgss_store(const char *query, int64 queryId,
		   int query_location, int query_len,
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
	dsa_pointer query_text = InvalidDsaPointer;
	int			stored_query_len = 0;
	bool		created_entry = false;

	Assert(query != NULL);

	/* Safety check... */
	if (pgss_shared == NULL)
		return;

	/*
	 * Nothing to do if compute_query_id isn't enabled and no other module
	 * computed a query identifier.
	 */
	if (queryId == INT64CONST(0))
		return;

	/* Set up key; clear padding, as the whole struct is hashed */
	memset(&key, 0, sizeof(pgssHashKey));
	key.userid = GetUserId();
	key.dbid = MyDatabaseId;
	key.queryid = queryId;
	key.toplevel = (nesting_level == 0);

	objid = hash_bytes_extended((const unsigned char *) &key,
								sizeof(pgssHashKey), 0);

	entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, PGSS_STATS_DBOID,
									 objid, false, NULL);
	if (!entry_ref)
	{
		uint64		threshold;

		/*
		 * Make room first.  pgss_max is a target, with admission allowed up
		 * to the eviction threshold.  Drop this observation if entry, query
		 * text, or query text memory pressure remains after eviction.
		 */
		threshold = pgss_evict_threshold();
		if (pgss_admission_pressure(threshold))
			pgss_evict_for_entry();

		if (pgss_admission_pressure(threshold))
			return;

		stored_query_len = query_len;
		query_text = pgss_store_query_text(query, query_location,
										   &stored_query_len, jstate);
		if (!DsaPointerIsValid(query_text))
			return;

		entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, PGSS_STATS_DBOID,
										 objid, true, &created_entry);
	}

	if (!entry_ref)
	{
		if (DsaPointerIsValid(query_text))
			pgss_free_dsa_query_text(query_text, stored_query_len);
		return;
	}

	shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

	if (created_entry)
	{
		/*
		 * The body is zero-filled by core.  Other backends can already see
		 * the entry, so use plain atomic writes, not initialization, and do
		 * not mark it live until the metadata is complete.
		 */
		pgstat_lock_entry(entry_ref, false);
		shared->key = key;
		shared->stats_since = GetCurrentTimestamp();
		shared->minmax_stats_since = shared->stats_since;
		pg_atomic_write_u32(&shared->sticky, jstate != NULL ? 1 : 0);
		pg_atomic_write_u64(&shared->usage,
							jstate != NULL ?
							pg_atomic_read_u64(&pgss_shared->cur_median_usage) :
							PGSS_USAGE_ONE);
		shared->query_text = query_text;
		shared->query_len = stored_query_len;
		shared->encoding = GetDatabaseEncoding();
		query_text = InvalidDsaPointer;
		pgss_mark_entry_live(shared);
		pgstat_unlock_entry(entry_ref);
	}
	else if (entry_ref->pending == NULL)
	{
		bool		metadata_valid;
		bool		key_matches;
		bool		has_query_text;

		pgstat_lock_entry_shared(entry_ref, false);
		metadata_valid = (shared->stats_since != 0);
		key_matches = metadata_valid &&
			memcmp(&shared->key, &key, sizeof(pgssHashKey)) == 0;
		has_query_text = DsaPointerIsValid(shared->query_text);
		pgstat_unlock_entry(entry_ref);

		if (!metadata_valid)
			goto done;

		/*
		 * The pgstat object ID is a hash of the full PGSS key.  A collision
		 * at this layer is rare, but would otherwise merge distinct
		 * userid/dbid/queryid/toplevel entries.  This is separate from
		 * ordinary queryid collisions, which pg_stat_statements has always
		 * treated as the same statement.
		 */
		if (!key_matches)
			goto done;

		/* Query text is best-effort; retry if an earlier attempt failed. */
		if (!has_query_text)
		{
			if (!DsaPointerIsValid(query_text))
			{
				stored_query_len = query_len;
				query_text = pgss_store_query_text(query, query_location,
												   &stored_query_len, jstate);
			}

			if (DsaPointerIsValid(query_text))
			{
				pgss_attach_query_text(shared, query_text, stored_query_len,
									   GetDatabaseEncoding());
				query_text = InvalidDsaPointer;
			}
		}
	}

	/* Increment the counts, except when jstate is not NULL */
	if (!jstate)
	{
		pgstat_prep_pending_from_entry_ref(entry_ref);

		pgss_accumulate_pending((PgStat_PgssPending *) entry_ref->pending,
								kind, total_time, rows, bufusage,
								walusage, jitusage,
								parallel_workers_to_launch,
								parallel_workers_launched, planOrigin);

		/* "Unstick" entry if it was previously sticky */
		if (pg_atomic_read_u32(&shared->sticky) != 0 &&
			pg_atomic_exchange_u32(&shared->sticky, 0) != 0)
			pg_atomic_write_u64(&shared->usage, PGSS_USAGE_ONE);

		pg_atomic_fetch_add_u64(&shared->usage, PGSS_USAGE_ONE);
	}

done:
	if (DsaPointerIsValid(query_text))
		pgss_free_dsa_query_text(query_text, stored_query_len);
}

/*
 * Does the entry match a pgssResetFilter?  Zero fields match anything.
 */
static bool
pgss_match_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssResetFilter *filter = (pgssResetFilter *) DatumGetPointer(match_data);
	PgStatShared_Pgss *shared;

	shared = pgss_shared_from_hash_entry(p);

	if (pg_atomic_read_u32(&shared->entry_live) == 0)
		return false;

	return (!filter->userid || shared->key.userid == filter->userid) &&
		(!filter->dbid || shared->key.dbid == filter->dbid) &&
		(!filter->queryid || shared->key.queryid == filter->queryid);
}

static bool
pgss_drop_matching_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	if (!pgss_match_entry(p, match_data))
		return false;

	pgss_drop_entry_and_text(pgss_shared_from_hash_entry(p));

	return true;
}

/*
 * Mark an entry dead and release its query text, before handing it back to
 * core pgstat for dropping.
 */
static void
pgss_drop_entry_and_text(PgStatShared_Pgss *shared)
{
	LWLockAcquire(&shared->header.lock, LW_EXCLUSIVE);
	if (pgss_mark_entry_dead(shared))
		pgss_free_query_text_locked(shared);
	LWLockRelease(&shared->header.lock);
}

static void
pgss_drop_entry(uint64 objid)
{
	PgStat_EntryRef *entry_ref;

	entry_ref = pgstat_get_entry_ref(PGSTAT_KIND_PGSS, PGSS_STATS_DBOID,
									 objid, false, NULL);
	if (entry_ref)
	{
		PgStatShared_Pgss *shared = (PgStatShared_Pgss *) entry_ref->shared_stats;

		pgss_drop_entry_and_text(shared);
	}

	if (!pgstat_drop_entry(PGSTAT_KIND_PGSS, PGSS_STATS_DBOID, objid, true))
		pgstat_request_entry_refs_gc();
}

/*
 * Reset entries corresponding to parameters passed.
 */
static TimestampTz
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
			shared->minmax_stats_since = stats_reset;
			for (int kind = 0; kind < PGSS_NUMKIND; kind++)
			{
				shared->counters.min_time[kind] = 0;
				shared->counters.max_time[kind] = 0;
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

		/* If all the parameters are available, use the fast path. */
		memset(&key, 0, sizeof(pgssHashKey));
		key.userid = userid;
		key.dbid = dbid;
		key.queryid = queryid;

		key.toplevel = false;
		objid = hash_bytes_extended((const unsigned char *) &key,
									sizeof(pgssHashKey), 0);
		pgss_drop_entry(objid);

		key.toplevel = true;
		objid = hash_bytes_extended((const unsigned char *) &key,
									sizeof(pgssHashKey), 0);
		pgss_drop_entry(objid);
	}
	else
	{
		pgstat_drop_matching_entries_of_kind(PGSTAT_KIND_PGSS,
											 pgss_drop_matching_entry,
											 PointerGetDatum(&filter));
	}

	/* Reset global statistics too when all entries were targeted */
	if (!userid && !dbid && !queryid)
	{
		pg_atomic_write_u64(&pgss_shared->entry_dealloc, 0);
		pg_atomic_write_u64(&pgss_shared->stats_reset, (uint64) stats_reset);
	}

	return stats_reset;
}

/* Store query text in DSA, returning InvalidDsaPointer on allocation failure. */
static dsa_pointer
qtext_store(const char *query, int query_len)
{
	dsa_pointer dp = InvalidDsaPointer;
	char	   *dst;
	Size		query_size;

	if (!pgss_reserve_query_text(query_len))
		return InvalidDsaPointer;

	query_size = (Size) query_len + 1;
	dp = dsa_allocate_extended(pgss_qtext_dsa, query_size, DSA_ALLOC_NO_OOM);

	if (!DsaPointerIsValid(dp))
	{
		pg_atomic_fetch_sub_u64(&pgss_shared->qtext_total_len,
								(uint64) query_len);
		return InvalidDsaPointer;
	}

	dst = dsa_get_address(pgss_qtext_dsa, dp);
	memcpy(dst, query, query_len);
	dst[query_len] = '\0';
	pg_atomic_fetch_add_u64(&pgss_shared->qtext_count, 1);

	return dp;
}

/*
 * Attach a stored query text to an entry, replacing any existing one.
 * The text is released instead if the entry died in the meantime.
 */
static void
pgss_attach_query_text(PgStatShared_Pgss *shpgss, dsa_pointer query_text,
					   int query_len, int encoding)
{
	Assert(DsaPointerIsValid(query_text) && query_len >= 0);

	LWLockAcquire(&shpgss->header.lock, LW_EXCLUSIVE);

	if (pg_atomic_read_u32(&shpgss->entry_live) == 0)
		pgss_free_dsa_query_text(query_text, query_len);
	else
	{
		pgss_free_query_text_locked(shpgss);
		shpgss->query_text = query_text;
		shpgss->query_len = query_len;
		shpgss->encoding = encoding;
	}

	LWLockRelease(&shpgss->header.lock);
}

/*
 * Store the (normalized, if jstate is given) query text in DSA.
 */
static dsa_pointer
pgss_store_query_text(const char *query, int query_location, int *query_len,
					  const JumbleState *jstate)
{
	char	   *norm_query = NULL;
	dsa_pointer query_text;

	query = CleanQuerytext(query, &query_location, query_len);
	if (jstate && jstate->clocations_count > 0)
		norm_query = generate_normalized_query(jstate, query,
											   query_location,
											   query_len);

	query_text = qtext_store(norm_query ? norm_query : query, *query_len);

	if (norm_query)
		pfree(norm_query);

	return query_text;
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

static bool
pgss_evict_context_contains(pgssEvictContext *context, uint64 objid)
{
	pgssEvictCandidate key;

	key.objid = objid;

	return bsearch(&key, context->candidates, context->target,
				   sizeof(pgssEvictCandidate),
				   pgss_evict_candidate_objid_cmp) != NULL;
}

static bool
pgss_drop_evicted_entry(PgStatShared_HashEntry *p, Datum match_data)
{
	pgssEvictContext *context = (pgssEvictContext *) DatumGetPointer(match_data);

	if (!pgss_evict_context_contains(context, p->key.objid))
		return false;

	pgss_drop_entry_and_text(pgss_shared_from_hash_entry(p));

	return true;
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
static void
pgss_evict_for_entry(void)
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
	uint64		qtext_count;
	uint64		qtext_victims;
	uint64		qtext_memory;
	uint64		qtext_total_len;
	uint64		threshold;
	uint64		victim_count;
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
	qtext_memory = pgss_query_text_memory_bytes();
	qtext_count = pg_atomic_read_u64(&pgss_shared->qtext_count);
	qtext_total_len = pg_atomic_read_u64(&pgss_shared->qtext_total_len);
	live_entry_count = pg_atomic_read_u64(&pgss_shared->entry_count);
	if (qtext_count < threshold && live_entry_count < threshold &&
		qtext_total_len < qtext_memory)
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

		candidates[ncandidates].objid = p->key.objid;
		candidates[ncandidates].has_query_text =
			DsaPointerIsValid(shared->query_text);
		candidates[ncandidates].query_len =
			candidates[ncandidates].has_query_text ? shared->query_len : 0;
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

	qtext_victims = qtext_count > (uint64) pgss_max ?
		qtext_count - (uint64) pgss_max : 0;
	entry_victims = live_entry_count > (uint64) pgss_max ?
		live_entry_count - (uint64) pgss_max : 0;
	victim_count = Max(qtext_victims, entry_victims);

	if (qtext_total_len >= qtext_memory)
	{
		qtext_bytes_to_free = qtext_total_len - qtext_memory + 1;
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
	if (target == 0)
		goto done;

	/* Sort the victims by objid for pgss_evict_context_contains() */
	qsort(candidates, target, sizeof(pgssEvictCandidate),
		  pgss_evict_candidate_objid_cmp);

	context.candidates = candidates;
	context.target = target;
	pgstat_drop_matching_entries_of_kind(PGSTAT_KIND_PGSS,
										 pgss_drop_evicted_entry,
										 PointerGetDatum(&context));

	pg_atomic_fetch_add_u64(&pgss_shared->entry_dealloc, 1);

	/* Give freed query text memory back to the OS when possible */
	dsa_trim(pgss_qtext_dsa);

done:
	if (candidates)
		pfree(candidates);
	LWLockRelease(&pgss_shared->entry_evict_lock);
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
	if (pgss_hash == NULL || !pgss_enabled(nesting_level))
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

	/*
	 * Even when not tracking plan time, we must increment the nesting level,
	 * to ensure that functions evaluated during planning are not seen as
	 * top-level calls.
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
	 *
	 * Even when not tracking the statement, we must still bump the nesting
	 * level so that functions evaluated within it are not seen as top-level
	 * calls; but not for EXECUTE or PREPARE, for the reasons above.  To be
	 * absolutely certain we don't mess up the nesting level, evaluate the
	 * condition just once.
	 */
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

/* Common code for all versions of pg_stat_statements() */
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
	pgssViewRow *rows;
	int			nrows;
	int			maxrows;

	pgss_require_ready();

	/*
	 * Superusers or roles with the privileges of pg_read_all_stats members
	 * are allowed.
	 */
	is_allowed_role = has_privs_of_role(userid, ROLE_PG_READ_ALL_STATS);

	/*
	 * Flush this backend's pending stats so its completed statements are
	 * visible.  This reports all pending stats kinds, not only PGSS.
	 */
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

	/*
	 * Copy the entries out of shared memory first.  Building tuples can error
	 * out (encoding conversion, tuplestore spill), which must not happen
	 * inside the dshash scan or with an entry lock held.
	 */
	nrows = 0;
	maxrows = (int) Min(pg_atomic_read_u64(&pgss_shared->entry_count) +
						PGSS_ENTRY_SCAN_HEADROOM, INT_MAX / 2);
	rows = palloc_array(pgssViewRow, maxrows);

	dshash_seq_init(&hstat, pgss_hash, false);
	while ((p = dshash_seq_next(&hstat)) != NULL)
	{
		PgStatShared_Pgss *shared;
		pgssViewRow *row;

		if (p->dropped)
			continue;

		shared = pgss_shared_from_hash_entry(p);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
			continue;

		if (nrows >= maxrows)
		{
			int			newmaxrows;

			if (maxrows > INT_MAX / 2)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("too many pg_stat_statements entries")));
			newmaxrows = maxrows * 2;
			rows = repalloc_array(rows, pgssViewRow, newmaxrows);
			maxrows = newmaxrows;
		}
		row = &rows[nrows];
		row->query = NULL;

		LWLockAcquire(&shared->header.lock, LW_SHARED);
		if (pg_atomic_read_u32(&shared->entry_live) == 0)
		{
			LWLockRelease(&shared->header.lock);
			continue;
		}

		row->key = shared->key;
		row->stats_since = shared->stats_since;
		row->minmax_stats_since = shared->minmax_stats_since;
		row->encoding = shared->encoding;
		row->counters = shared->counters;
		if (showtext && DsaPointerIsValid(shared->query_text) &&
			(is_allowed_role || shared->key.userid == userid))
		{
			Size		query_size;

			row->query_len = shared->query_len;
			if (row->query_len < 0 ||
				!AllocSizeIsValid((Size) row->query_len + 1))
			{
				LWLockRelease(&shared->header.lock);
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("query text is too long")));
			}
			query_size = (Size) row->query_len + 1;
			row->query = palloc_extended(query_size, MCXT_ALLOC_NO_OOM);
			if (row->query == NULL)
			{
				LWLockRelease(&shared->header.lock);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("out of memory")));
			}
			memcpy(row->query,
				   dsa_get_address(pgss_qtext_dsa, shared->query_text),
				   row->query_len + 1);
		}
		LWLockRelease(&shared->header.lock);

		/* Skip entry if unexecuted (ie, it's a pending "sticky" entry) */
		if (row->counters.calls[PGSS_EXEC] == 0 &&
			row->counters.calls[PGSS_PLAN] == 0)
		{
			if (row->query)
				pfree(row->query);
			continue;
		}

		nrows++;
	}
	dshash_seq_term(&hstat);

	for (int r = 0; r < nrows; r++)
	{
		pgssViewRow *row = &rows[r];
		Counters	tmp = row->counters;
		Datum		values[PG_STAT_STATEMENTS_COLS];
		bool		nulls[PG_STAT_STATEMENTS_COLS];
		int			i = 0;
		double		stddev;

		memset(values, 0, sizeof(values));
		memset(nulls, 0, sizeof(nulls));

		values[i++] = ObjectIdGetDatum(row->key.userid);
		values[i++] = ObjectIdGetDatum(row->key.dbid);
		if (api_version >= PGSS_V1_9)
			values[i++] = BoolGetDatum(row->key.toplevel);

		if (is_allowed_role || row->key.userid == userid)
		{
			if (api_version >= PGSS_V1_2)
				values[i++] = Int64GetDatumFast(row->key.queryid);

			if (row->query)
			{
				char	   *enc = pg_any_to_server(row->query, row->query_len,
												   row->encoding);

				values[i++] = CStringGetTextDatum(enc);
				if (enc != row->query)
					pfree(enc);
			}
			else
			{
				/* Query text not requested, or not available */
				nulls[i++] = true;
			}
		}
		else
		{
			/* Don't show queryid */
			if (api_version >= PGSS_V1_2)
				nulls[i++] = true;

			/*
			 * Don't show query text, but hint as to the reason for not doing
			 * so if it was requested
			 */
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
			values[i++] = TimestampTzGetDatum(row->stats_since);
			values[i++] = TimestampTzGetDatum(row->minmax_stats_since);
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

		if (row->query)
			pfree(row->query);
	}
	pfree(rows);
}

/* Number of output arguments (columns) for pg_stat_statements_info */
#define PG_STAT_STATEMENTS_INFO_COLS_V1_9	2
#define PG_STAT_STATEMENTS_INFO_COLS		7	/* maximum of above */

/*
 * Return statistics of pg_stat_statements.
 */
Datum
pg_stat_statements_info(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	uint64		entry_dealloc;
	TimestampTz stats_reset;
	Datum		values[PG_STAT_STATEMENTS_INFO_COLS] = {0};
	bool		nulls[PG_STAT_STATEMENTS_INFO_COLS] = {0};

	pgss_require_ready();

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	entry_dealloc = pg_atomic_read_u64(&pgss_shared->entry_dealloc);
	stats_reset = (TimestampTz) pg_atomic_read_u64(&pgss_shared->stats_reset);

	switch (tupdesc->natts)
	{
		case PG_STAT_STATEMENTS_INFO_COLS_V1_9:
			values[0] = Int64GetDatum((int64) entry_dealloc);
			values[1] = TimestampTzGetDatum(stats_reset);
			break;
		case PG_STAT_STATEMENTS_INFO_COLS:
			values[0] = Int64GetDatum((int64) entry_dealloc);
			values[1] = Int64GetDatum((int64)
									  pg_atomic_read_u64(&pgss_shared->entry_count));
			values[2] = Int64GetDatum((int64)
									  pgss_dsa_size(pgStatLocal.kind_dsa[PGSTAT_KIND_PGSS]));
			values[3] = Int64GetDatum((int64) pgss_dsa_size(pgss_qtext_dsa));
			values[4] = Int64GetDatum((int64)
									  pg_atomic_read_u64(&pgss_shared->qtext_count));
			values[5] = Int64GetDatum((int64)
									  pg_atomic_read_u64(&pgss_shared->qtext_total_len));
			values[6] = TimestampTzGetDatum(stats_reset);
			break;
		default:
			elog(ERROR, "incorrect number of output arguments");
	}

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}
