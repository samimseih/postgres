/*-------------------------------------------------------------------------
 *
 * pg_stat_statements.c
 *		Track statement planning and execution times as well as resource
 *		usage across a whole database cluster.
 *
 * Execution costs are totaled for each distinct source query.
 *
 * Starting in Postgres 9.2, this module normalized query entries.  As of
 * Postgres 14, the normalization is done by the core if compute_query_id is
 * enabled, or optionally by third-party modules.
 *
 * To facilitate presenting entries to users, we create "representative" query
 * strings in which constants are replaced with parameter symbols ($n), to
 * make it clearer what a normalized entry can represent.  To save on shared
 * memory, and to avoid having to truncate oversized query strings, we store
 * these strings in a temporary external query-texts file.  Offsets into this
 * file are kept in shared memory.
 *
 * Note about locking issues: statement entries live in the cumulative stats
 * shared hash, and their fixed-size pgssEntry data is protected by the core
 * per-entry stats lock.  Backend-local pending counters are merged through
 * the cumulative stats flush callback.
 *
 * The external query text file has separate synchronization.  Creating a new
 * query text mapping holds pgss->qtext_lock shared while reserving file space
 * and initializing the entry's offset/length.  Reading the file also holds
 * pgss->qtext_lock shared: concurrent writers only fill in space reserved
 * beyond the texts of initialized entries, and qtext_fetch() rejects texts
 * that were not completely written when the file was read.  Rewriting or
 * truncating the file requires pgss->qtext_lock exclusively.  The shared
 * state variable pgss->extent is accessed while holding pgss->mutex.
 *
 * Entry eviction is coordinated by pgss->dealloc_lock.  When the entry count
 * is near pg_stat_statements.max, this lock is acquired conditionally so that
 * backends do not pile up behind eviction; if the lock is unavailable,
 * pg_stat_statements.max is treated as a soft target and the backend
 * continues with entry creation.  As the entry count drifts further over the
 * target, an increasing fraction of backends wait for pgss->dealloc_lock
 * before creating more entries.  If the entry count drifts far enough over the
 * target, all backends wait.
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
#include <sys/stat.h>
#include <unistd.h>

#include "access/htup_details.h"
#include "access/parallel.h"
#include "catalog/pg_authid.h"
#include "common/hashfn.h"
#include "common/pg_prng.h"
#include "executor/instrument.h"
#include "funcapi.h"
#include "jit/jit.h"
#include "mb/pg_wchar.h"
#include "miscadmin.h"
#include "nodes/queryjumble.h"
#include "optimizer/planner.h"
#include "parser/analyze.h"
#include "pgstat.h"
#include "storage/fd.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "tcop/utility.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/pgstat_internal.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"

PG_MODULE_MAGIC_EXT(
					.name = "pg_stat_statements",
					.version = PG_VERSION
);

/*
 * Location of external query text file.
 */
#define PGSS_TEXT_FILE	PG_STAT_TMP_DIR "/pgss_query_texts.stat"

/*
 * XXX: reserve dedicated custom cumulative stats kind IDs for
 * pg_stat_statements before this becomes committable.
 */
#define PGSS_STATS_KIND PGSTAT_KIND_EXPERIMENTAL
#define PGSS_GLOBAL_STATS_KIND (PGSTAT_KIND_EXPERIMENTAL + 1)

/* XXX: Should USAGE_EXEC reflect execution time and/or buffer usage? */
#define USAGE_EXEC(duration)	(1.0)
#define USAGE_INIT				(1.0)	/* including initial planning */
#define ASSUMED_MEDIAN_INIT		(10.0)	/* initial assumed median usage */
#define ASSUMED_LENGTH_INIT		1024	/* initial assumed mean query length */
#define USAGE_DECREASE_FACTOR	(0.99)	/* decreased every entry_dealloc */
#define STICKY_DECREASE_FACTOR	(0.50)	/* factor for sticky entries */
#define USAGE_DEALLOC_PERCENT	5	/* free this % of entries at once */
#define USAGE_DEALLOC_BLOCK_PERCENT	20	/* max % over target before block */
#define USAGE_DEALLOC_BLOCK_MAX_EXTRA	20000	/* max extra entries before
												 * block */
#define IS_STICKY(c)	((c.calls[PGSS_PLAN] + c.calls[PGSS_EXEC]) == 0)

#define PGSS_ACCUM_COUNTER_FIELD(dst, src, field) \
	(dst)->field += (src)->field
#define PGSS_ACCUM_INSTR_TIME_FIELD(dst, src, field) \
	(dst)->field += INSTR_TIME_GET_MILLISEC((src)->field)

#define PGSS_BUFFER_USAGE_FIELDS(accum_field, dst, src) \
	accum_field(dst, src, shared_blks_hit); \
	accum_field(dst, src, shared_blks_read); \
	accum_field(dst, src, shared_blks_dirtied); \
	accum_field(dst, src, shared_blks_written); \
	accum_field(dst, src, local_blks_hit); \
	accum_field(dst, src, local_blks_read); \
	accum_field(dst, src, local_blks_dirtied); \
	accum_field(dst, src, local_blks_written); \
	accum_field(dst, src, temp_blks_read); \
	accum_field(dst, src, temp_blks_written);

#define PGSS_BUFFER_TIME_FIELDS(accum_field, dst, src) \
	accum_field(dst, src, shared_blk_read_time); \
	accum_field(dst, src, shared_blk_write_time); \
	accum_field(dst, src, local_blk_read_time); \
	accum_field(dst, src, local_blk_write_time); \
	accum_field(dst, src, temp_blk_read_time); \
	accum_field(dst, src, temp_blk_write_time);

#define PGSS_COUNTER_ACCUM_FIELDS(accum_field, dst, src) \
	accum_field(dst, src, rows); \
	PGSS_BUFFER_USAGE_FIELDS(accum_field, dst, src) \
	PGSS_BUFFER_TIME_FIELDS(accum_field, dst, src) \
	accum_field(dst, src, usage); \
	accum_field(dst, src, wal_records); \
	accum_field(dst, src, wal_fpi); \
	accum_field(dst, src, wal_bytes); \
	accum_field(dst, src, wal_buffers_full); \
	accum_field(dst, src, jit_functions); \
	accum_field(dst, src, jit_generation_time); \
	accum_field(dst, src, jit_inlining_count); \
	accum_field(dst, src, jit_deform_time); \
	accum_field(dst, src, jit_deform_count); \
	accum_field(dst, src, jit_inlining_time); \
	accum_field(dst, src, jit_optimization_count); \
	accum_field(dst, src, jit_optimization_time); \
	accum_field(dst, src, jit_emission_count); \
	accum_field(dst, src, jit_emission_time); \
	accum_field(dst, src, parallel_workers_to_launch); \
	accum_field(dst, src, parallel_workers_launched); \
	accum_field(dst, src, generic_plan_calls); \
	accum_field(dst, src, custom_plan_calls)

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
 * Key that defines the identity of a statement entry.  We separate queries by
 * user and by database even if they are otherwise identical.
 *
 * Callers zero this struct before setting fields, including the padding
 * bytes, because pgss_stats_objid() hashes the complete key to produce the
 * uint64 objid that identifies the entry in the cumulative stats system.
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

/*
 * Global statistics for pg_stat_statements
 */
typedef struct pgssGlobalStats
{
	int64		dealloc;		/* # of times entries were deallocated */
	TimestampTz stats_reset;	/* timestamp with all stats reset */
} pgssGlobalStats;

/*
 * Statistics per statement
 *
 * Note: in event of a failure in garbage collection of the query text file,
 * we reset query_offset to zero and query_len to -1.  This will be seen as
 * an invalid state by qtext_fetch().
 */
typedef struct pgssEntry
{
	pgssHashKey key;			/* statement identity */
	Counters	counters;		/* the statistics for this query */
	Size		query_offset;	/* query text offset in external file */
	int			query_len;		/* # of valid bytes in query string, or -1 */
	int			encoding;		/* query text encoding */
	bool		executed;		/* planned or executed, so not "sticky" */
	TimestampTz stats_since;	/* timestamp of entry allocation */
	TimestampTz minmax_stats_since; /* timestamp of last min/max values reset */
} pgssEntry;

/*
 * Backend-local pending statistics before flush to shared memory.
 *
 * Plain counters can be accumulated directly, and mean/stddev state can be
 * merged later using Chan's parallel variance algorithm.  Min/max statistics
 * are different because pg_stat_statements_reset(minmax_only => true)
 * establishes a reset boundary.  minmax_since conservatively tags the pending
 * min/max window, so flush can avoid merging values that began before the
 * shared reset.
 *
 * XXX: replace this timestamp boundary with a per-entry min/max generation,
 * so pending entries can discard pre-reset min/max state while still retaining
 * post-reset observations in the same pending batch.
 */
typedef struct pgssPendingEntry
{
	TimestampTz minmax_since[PGSS_NUMKIND]; /* min/max window start */
	Counters	counters;		/* pending statistics */
} pgssPendingEntry;

/*
 * Shared memory statistics entry visible to all backends.
 */
typedef struct pgssSharedEntry
{
	PgStatShared_Common header; /* standard pgstat entry header */
	pgssEntry	stats;			/* pg_stat_statements statistics data */
} pgssSharedEntry;

/*
 * Shared state for the external query text file.
 */
typedef struct pgssSharedState
{
	LWLock		qtext_lock;		/* protects query text file rewrites */
	LWLock		dealloc_lock;	/* coordinates nonblocking entry eviction */
	pg_atomic_uint64 cur_entry_count;	/* estimated live entry count */
	double		cur_median_usage;	/* current median usage in stats */
	Size		mean_query_len; /* current mean entry text length */
	slock_t		mutex;			/* protects following fields only: */
	Size		extent;			/* current extent of query file */
} pgssSharedState;

typedef struct pgssEntryDealloc
{
	PgStat_HashKey key;			/* core stats key of entry */
	double		usage;			/* usage factor */
} pgssEntryDealloc;

/* Links to shared memory state */
static pgssSharedState *pgss;

static void pgss_shmem_request(void *arg);
static void pgss_shmem_init(void *arg);

static const ShmemCallbacks pgss_shmem_callbacks = {
	.request_fn = pgss_shmem_request,
	.init_fn = pgss_shmem_init,
};

/*---- Local variables ----*/

/* Current nesting depth of planner/ExecutorRun/ProcessUtility calls */
static int	nesting_level = 0;

/*
 * State saved in a QueryDesc at ExecutorStart, for use at ExecutorEnd.  A
 * query may end at a different nesting level than the one it started at,
 * for example a cursor declared in a transaction and closed at COMMIT.
 */
typedef struct pgssQueryState
{
	int			nesting_level;	/* nesting level at ExecutorStart */
	pgssHashKey key;			/* statement identity at ExecutorStart */
} pgssQueryState;

/* ID of pg_stat_statements for QueryDesc extension state */
static int	pgss_queryDesc_id = -1;

/* Query text file image loaded while writing the stats file */
static char *qtext_serialize_buffer = NULL;
static Size qtext_serialize_buffer_size = 0;
static bool qtext_serialize_buffer_loaded = false;

/*
 * Set when this backend has stale entry references to release.  The GC
 * request is deferred so that it is made once rather than per entry.
 */
static bool pgss_refs_gc_pending = false;

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

static int	pgss_max = 5000;	/* target # statements to track */
static int	pgss_track = PGSS_TRACK_TOP;	/* tracking level */
static bool pgss_track_utility = true;	/* whether to track utility commands */
static bool pgss_track_planning = false;	/* whether to track planning
											 * duration */
static bool pgss_save = true;	/* whether to save query texts across shutdown */

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
					   bool toplevel,
					   const pgssHashKey *cached_key,
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
static PgStat_EntryRef *entry_alloc(const pgssHashKey *key,
									uint64 stats_objid, const char *query,
									int query_location, int query_len,
									int encoding, const JumbleState *jstate);
static uint64 entry_dealloc_soft_limit(void);
static uint64 entry_dealloc_hard_limit(void);
static bool entry_dealloc_should_wait(uint64 live_entry_count);
static void entry_dealloc(bool wait);
static bool qtext_store(const char *query, int query_len,
						Size *query_offset);
static char *qtext_load_file(Size *buffer_size);
static char *qtext_fetch(Size query_offset, int query_len,
						 char *buffer, Size buffer_size);
static void qtext_reset_file(void);
static bool pgss_stats_to_serialized_data(const PgStat_HashKey *key,
										  const PgStatShared_Common *header,
										  FILE *statfile);
static bool pgss_stats_from_serialized_data(const PgStat_HashKey *key,
											PgStatShared_Common *header,
											FILE *statfile);
static void pgss_stats_file_finished(PgStat_StatsFileOp status);
static bool need_gc_qtexts(void);
static void gc_qtexts(void);
static TimestampTz entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only);
static char *generate_normalized_query(const JumbleState *jstate,
									   const char *query,
									   int query_loc, int *query_len_p);
static void entry_adjust_count(uint64 old_count, uint64 new_count);
static void entry_rebuild_runtime_state(void);
static void entry_reset_usage_state(void);
static void pgss_global_stats_init_shmem_cb(void *stats);
static void pgss_global_stats_snapshot_cb(void);
static void pgss_global_stats_reset(TimestampTz stats_reset);
static pgssGlobalStats pgss_global_stats_fetch(void);
static void pgss_global_stats_increment_dealloc(void);
static uint64 pgss_stats_objid(const pgssHashKey *key);
static inline pgssSharedEntry *pgss_get_shared_entry(PgStatShared_HashEntry *hash_entry);
static inline bool pgss_stats_entry_ref_is_stale(PgStat_EntryRef *entry_ref);
static PgStat_EntryRef *pgss_stats_get_entry_ref(const pgssHashKey *key,
												 uint64 stats_objid);
static bool pgss_drop_entry(const pgssHashKey *key);
static bool pgss_entry_matches(const pgssEntry *entry,
							   const pgssHashKey *match);
static bool pgss_stats_entry_matches(PgStatShared_HashEntry *hash_entry,
									 Datum match_data);
static uint64 pgss_stats_count_visible_entries(void);
static void pgss_stats_reset_minmax(pgssEntry *entry, TimestampTz stats_reset);
static void pgss_stats_minmax_reset_entry(const pgssHashKey *key,
										  TimestampTz stats_reset);
static void pgss_stats_minmax_reset_matching(const pgssHashKey *match,
											 TimestampTz stats_reset);
static void pgss_stats_record(PgStat_EntryRef *entry_ref, pgssStoreKind kind,
							  double total_time, uint64 rows,
							  const BufferUsage *bufusage,
							  const WalUsage *walusage,
							  const struct JitInstrumentation *jitusage,
							  int parallel_workers_to_launch,
							  int parallel_workers_launched,
							  PlannedStmtOrigin planOrigin);
static void pgss_entry_mark_executed(PgStat_EntryRef *entry_ref);
static PgStat_FlushResult pgss_stats_flush_pending(PgStat_EntryRef *entry_ref,
												   bool nowait,
												   bool xact_boundary);
static void pgss_stats_flush_kind(Counters *dst, const Counters *src,
								  pgssStoreKind kind,
								  TimestampTz minmax_stats_since,
								  TimestampTz minmax_since);

static const PgStat_KindInfo pgss_stats = {
	.name = "pg_stat_statements",
	.fixed_amount = false,
	.write_to_file = true,
	.accessed_across_databases = true,
	.own_hash = true,
	.release_ref_after_flush = true,
	.shared_size = sizeof(pgssSharedEntry),
	.shared_data_off = offsetof(pgssSharedEntry, stats),
	.shared_data_len = sizeof(((pgssSharedEntry *) 0)->stats),
	.pending_size = sizeof(pgssPendingEntry),
	.flush_pending_cb = pgss_stats_flush_pending,
	.to_serialized_data = pgss_stats_to_serialized_data,
	.from_serialized_data = pgss_stats_from_serialized_data,
	.finish = pgss_stats_file_finished,
};

static const PgStat_KindInfo pgss_global_stats = {
	.name = "pg_stat_statements_info",
	.fixed_amount = true,
	.write_to_file = true,
	.shared_size = sizeof(pgssGlobalStats),
	.shared_data_off = 0,
	.shared_data_len = sizeof(pgssGlobalStats),
	.init_shmem_cb = pgss_global_stats_init_shmem_cb,
	.reset_all_cb = pgss_global_stats_reset,
	.snapshot_cb = pgss_global_stats_snapshot_cb,
};

/*
 * Module load callback
 */
void
_PG_init(void)
{
	/*
	 * In order to create our shared memory area, we have to be loaded via
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

	/*
	 * Register our custom cumulative statistics kinds.
	 */
	pgstat_register_kind(PGSS_STATS_KIND, &pgss_stats);
	pgstat_register_kind(PGSS_GLOBAL_STATS_KIND, &pgss_global_stats);

	/*
	 * Define (or redefine) custom GUC variables.
	 */
	/* XXX: should we rename this to max_target? */
	DefineCustomIntVariable("pg_stat_statements.max",
							"Sets the target number of statements tracked by pg_stat_statements.",
							NULL,
							&pgss_max,
							5000,
							100,
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
							 "Save pg_stat_statements query texts across server shutdowns.",
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
	 * Register our shared memory needs.
	 */
	RegisterShmemCallbacks(&pgss_shmem_callbacks);

	/*
	 * Get an ID to store state in QueryDesc.
	 */
	pgss_queryDesc_id = GetQueryDescExtensionId("pg_stat_statements");

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
 * shmem request callback: Request shared memory resources.
 *
 * This is called at postmaster startup.  Note that the shared memory isn't
 * allocated here yet, this merely register our needs.
 *
 * In EXEC_BACKEND mode, this is also called in each backend, to re-attach to
 * the shared memory area that was already initialized.
 */
static void
pgss_shmem_request(void *arg)
{
	ShmemRequestStruct(.name = "pg_stat_statements",
					   .size = sizeof(pgssSharedState),
					   .ptr = (void **) &pgss,
		);
}

/*
 * shmem init callback: Initialize our shared query text state.
 */
static void
pgss_shmem_init(void *arg)
{
	FILE	   *qfile;

	LWLockInitialize(&pgss->qtext_lock,
					 LWLockNewTrancheId("pg_stat_statements query text"));
	LWLockInitialize(&pgss->dealloc_lock,
					 LWLockNewTrancheId("pg_stat_statements dealloc"));
	pg_atomic_init_u64(&pgss->cur_entry_count, 0);
	pgss->cur_median_usage = ASSUMED_MEDIAN_INIT;
	pgss->mean_query_len = ASSUMED_LENGTH_INIT;
	SpinLockInit(&pgss->mutex);
	pgss->extent = 0;

	/* Unlink query text file possibly left over from crash */
	unlink(PGSS_TEXT_FILE);

	/* Allocate new query text temp file */
	qfile = AllocateFile(PGSS_TEXT_FILE, PG_BINARY_W);
	if (qfile == NULL)
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not create file \"%s\": %m",
						PGSS_TEXT_FILE)));
	else
		FreeFile(qfile);
}

static void
pgss_drop_restored_entry(const PgStat_HashKey *key)
{
	if (!pgstat_drop_entry(PGSS_STATS_KIND, key->dboid, key->objid, true))
		pgss_refs_gc_pending = true;
}

static bool
pgss_stats_to_serialized_data(const PgStat_HashKey *key,
							  const PgStatShared_Common *header,
							  FILE *statfile)
{
	const pgssSharedEntry *shared_entry = (const pgssSharedEntry *) header;
	const pgssEntry *entry = &shared_entry->stats;
	char	   *query = NULL;
	int			query_len = -1;

	/*
	 * Core has already written pgssEntry.  Append query text only when
	 * pg_stat_statements.save requests it.
	 */
	if (entry->executed && pgss_save && entry->query_len >= 0)
	{
		if (!qtext_serialize_buffer_loaded)
		{
			qtext_serialize_buffer =
				qtext_load_file(&qtext_serialize_buffer_size);
			qtext_serialize_buffer_loaded = true;
		}

		query = qtext_fetch(entry->query_offset, entry->query_len,
							qtext_serialize_buffer,
							qtext_serialize_buffer_size);
		if (query)
			query_len = entry->query_len;
	}

	if (fwrite(&query_len, sizeof(query_len), 1, statfile) != 1)
		return false;

	if (query_len >= 0 &&
		fwrite(query, 1, query_len, statfile) != (Size) query_len)
		return false;

	return true;
}

static bool
pgss_stats_from_serialized_data(const PgStat_HashKey *key,
								PgStatShared_Common *header,
								FILE *statfile)
{
	pgssSharedEntry *shared_entry = (pgssSharedEntry *) header;
	pgssEntry  *entry = &shared_entry->stats;
	char	   *query;
	int			query_len;
	Size		query_offset = 0;

	entry->query_offset = 0;
	entry->query_len = -1;

	if (fread(&query_len, sizeof(query_len), 1, statfile) != 1)
		return false;

	if (query_len < 0)
	{
		pgss_drop_restored_entry(key);
		return true;
	}

	if ((Size) query_len >= MaxAllocHugeSize)
		return false;

	/*
	 * Entries are not restored if pg_stat_statements.save is off at startup,
	 * and sticky entries are never restored.  Also give up on just this entry
	 * if we can't allocate memory for its text.
	 */
	if (!pgss_save || !entry->executed)
		query = NULL;
	else
		query = (char *) palloc_extended(Max(query_len, 1),
										 MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM);
	if (query == NULL)
	{
		if (fseek(statfile, query_len, SEEK_CUR) != 0)
			return false;
		pgss_drop_restored_entry(key);
		return true;
	}

	if (query_len > 0 &&
		fread(query, 1, query_len, statfile) != (Size) query_len)
	{
		pfree(query);
		return false;
	}

	if (qtext_store(query, query_len, &query_offset))
	{
		entry->query_offset = query_offset;
		entry->query_len = query_len;
	}
	else
		pgss_drop_restored_entry(key);

	pfree(query);

	return true;
}

static void
pgss_stats_file_finished(PgStat_StatsFileOp status)
{
	if (status == STATS_READ)
	{
		entry_rebuild_runtime_state();

		if (pg_atomic_read_u64(&pgss->cur_entry_count) >=
			entry_dealloc_soft_limit())
			entry_dealloc(true);
	}
	else if (status == STATS_DISCARD)
	{
		TimestampTz stats_reset = GetCurrentTimestamp();

		LWLockAcquire(&pgss->qtext_lock, LW_EXCLUSIVE);
		LWLockAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE);
		qtext_reset_file();
		entry_reset_usage_state();
		LWLockRelease(&pgss->dealloc_lock);
		LWLockRelease(&pgss->qtext_lock);

		pgss_global_stats_reset(stats_reset);
	}

	if (pgss_refs_gc_pending)
	{
		pgstat_request_entry_refs_gc();
		pgss_refs_gc_pending = false;
	}

	if (qtext_serialize_buffer)
		pfree(qtext_serialize_buffer);

	qtext_serialize_buffer = NULL;
	qtext_serialize_buffer_size = 0;
	qtext_serialize_buffer_loaded = false;
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
	if (!pgss_enabled(nesting_level))
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
	 * immediately record the normalized form of the query string.  If there
	 * were no such constants, the normalized string would be the same as the
	 * query text anyway, so there's no need for an early entry.
	 */
	if (jstate && jstate->clocations_count > 0)
		pgss_store(pstate->p_sourcetext,
				   query->queryId,
				   query->stmt_location,
				   query->stmt_len,
				   nesting_level == 0,
				   NULL,
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
				   nesting_level == 0,
				   NULL,
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
	pgssQueryState *qstate;

	/*
	 * If query has queryId zero, don't track it.  This prevents double
	 * counting of optimizable statements that are directly contained in
	 * utility statements.
	 */
	if (pgss_enabled(nesting_level) && queryDesc->plannedstmt->queryId != INT64CONST(0))
	{
		/* Request all summary instrumentation, i.e. timing, buffers and WAL */
		queryDesc->query_instr_options |= INSTRUMENT_ALL;

		/* Remember the statement identity and nesting level for ExecutorEnd */
		qstate = MemoryContextAlloc(GetMemoryChunkContext(queryDesc),
									sizeof(pgssQueryState));
		qstate->nesting_level = nesting_level;
		memset(&qstate->key, 0, sizeof(pgssHashKey));
		qstate->key.userid = GetUserId();
		qstate->key.dbid = MyDatabaseId;
		qstate->key.queryid = queryDesc->plannedstmt->queryId;
		qstate->key.toplevel = nesting_level == 0;
		SetQueryDescExtensionState(queryDesc, pgss_queryDesc_id, qstate);
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
	pgssQueryState *qstate = GetQueryDescExtensionState(queryDesc,
														pgss_queryDesc_id);

	/*
	 * Use the nesting level saved at ExecutorStart, if available, since the
	 * query may end at a different level (e.g. a cursor closed at COMMIT).
	 */
	if (queryId != INT64CONST(0) && queryDesc->query_instr &&
		qstate != NULL &&
		pgss_enabled(qstate->nesting_level))
	{
		pgss_store(queryDesc->sourceText,
				   qstate->key.queryid,
				   queryDesc->plannedstmt->stmt_location,
				   queryDesc->plannedstmt->stmt_len,
				   qstate->key.toplevel,
				   &qstate->key,
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
				   nesting_level == 0,
				   NULL,
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
		   bool toplevel,
		   const pgssHashKey *cached_key,
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
	const pgssHashKey *store_key = cached_key;
	PgStat_EntryRef *entry_ref = NULL;
	uint64		stats_objid;
	int			encoding = GetDatabaseEncoding();

	Assert(query != NULL);

	/*
	 * Nothing to do if compute_query_id isn't enabled and no other module
	 * computed a query identifier.
	 */
	if (queryId == INT64CONST(0))
		return;

	/*
	 * Confine our attention to the relevant part of the string, if the query
	 * is a portion of a multi-statement source string, and update query
	 * location and length if needed.
	 */
	query = CleanQuerytext(query, &query_location, &query_len);

	if (store_key == NULL)
	{
		/* Set up key for stats entry search. */
		memset(&key, 0, sizeof(pgssHashKey));

		key.userid = GetUserId();
		key.dbid = MyDatabaseId;
		key.queryid = queryId;
		key.toplevel = toplevel;
		store_key = &key;
	}
	else
	{
		Assert(store_key->queryid == queryId);
		Assert(store_key->toplevel == toplevel);
	}
	stats_objid = pgss_stats_objid(store_key);

	/* Create new entry, if not present */
	entry_ref = pgss_stats_get_entry_ref(store_key, stats_objid);
	if (entry_ref == NULL)
	{
		entry_ref = entry_alloc(store_key, stats_objid, query, query_location,
								query_len, encoding, jstate);

		/*
		 * An entry created at parse analysis may never be executed, so don't
		 * keep a reference to it that could keep it from being freed.
		 */
		if (jstate)
		{
			(void) pgstat_release_entry_ref_if_unused(entry_ref);
			return;
		}
	}
	else if (jstate)
		return;

	/* Record pending statistics, except when jstate is not NULL. */
	if (!jstate)
		pgss_stats_record(entry_ref, kind, total_time, rows, bufusage,
						  walusage, jitusage, parallel_workers_to_launch,
						  parallel_workers_launched, planOrigin);
}

/*
 * Mark a "sticky" entry, created at parse analysis, as executed.  Its usage
 * started at the median so that it would survive until executed; reset it to
 * the usage of a newly executed entry.
 */
static void
pgss_entry_mark_executed(PgStat_EntryRef *entry_ref)
{
	pgssSharedEntry *shared_entry = (pgssSharedEntry *) entry_ref->shared_stats;

	/* Unlocked read; this only ever changes once for an entry. */
	if (likely(shared_entry->stats.executed))
		return;

	(void) pgstat_lock_entry(entry_ref, false);
	if (shared_entry->stats.stats_since != 0 && !shared_entry->stats.executed)
	{
		shared_entry->stats.executed = true;
		shared_entry->stats.counters.usage = USAGE_INIT;
	}
	pgstat_unlock_entry(entry_ref);
}

static void
pgss_stats_record(PgStat_EntryRef *entry_ref, pgssStoreKind kind,
				  double total_time, uint64 rows,
				  const BufferUsage *bufusage,
				  const WalUsage *walusage,
				  const JitInstrumentation *jitusage,
				  int parallel_workers_to_launch,
				  int parallel_workers_launched,
				  PlannedStmtOrigin planOrigin)
{
	pgssPendingEntry *pending;
	Counters   *counters;

	Assert(entry_ref != NULL);
	Assert(kind == PGSS_PLAN || kind == PGSS_EXEC);
	Assert(bufusage != NULL);
	Assert(walusage != NULL);

	/*
	 * Our calls only reach the shared counters at flush time, so mark a
	 * sticky entry as executed here, so that eviction treats it like other
	 * executed entries.  Checking this once per pending batch is enough.
	 */
	if (entry_ref->pending == NULL)
		pgss_entry_mark_executed(entry_ref);

	pgstat_prep_pending_from_entry_ref(entry_ref);
	pending = (pgssPendingEntry *) entry_ref->pending;
	counters = &pending->counters;

	/*
	 * This timestamp decides whether the batch's min/max values survive a
	 * concurrent min/max-only reset.  We set it once per pending batch, so a
	 * batch that spans such a reset may omit post-reset min/max observations
	 * until a later batch establishes new values.
	 */
	if (counters->calls[kind] == 0)
		pending->minmax_since[kind] = GetCurrentTimestamp();

	counters->calls[kind]++;
	counters->total_time[kind] += total_time;

	if (counters->calls[kind] == 1)
	{
		counters->min_time[kind] = total_time;
		counters->max_time[kind] = total_time;
		counters->mean_time[kind] = total_time;
	}
	else
	{
		/*
		 * Welford's online algorithm for accumulating mean and sum of squared
		 * deviations.
		 */
		double		old_mean = counters->mean_time[kind];

		counters->mean_time[kind] +=
			(total_time - old_mean) / counters->calls[kind];
		counters->sum_var_time[kind] +=
			(total_time - old_mean) * (total_time - counters->mean_time[kind]);

		if (counters->min_time[kind] > total_time)
			counters->min_time[kind] = total_time;
		if (counters->max_time[kind] < total_time)
			counters->max_time[kind] = total_time;
	}

	counters->rows += rows;

	PGSS_BUFFER_USAGE_FIELDS(PGSS_ACCUM_COUNTER_FIELD, counters, bufusage);
	PGSS_BUFFER_TIME_FIELDS(PGSS_ACCUM_INSTR_TIME_FIELD, counters, bufusage);

	counters->usage += USAGE_EXEC(total_time);
	counters->wal_records += walusage->wal_records;
	counters->wal_fpi += walusage->wal_fpi;
	counters->wal_bytes += walusage->wal_bytes;
	counters->wal_buffers_full += walusage->wal_buffers_full;

	if (jitusage)
	{
		counters->jit_functions += jitusage->created_functions;
		counters->jit_generation_time +=
			INSTR_TIME_GET_MILLISEC(jitusage->generation_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->deform_counter))
			counters->jit_deform_count++;
		counters->jit_deform_time +=
			INSTR_TIME_GET_MILLISEC(jitusage->deform_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter))
			counters->jit_inlining_count++;
		counters->jit_inlining_time +=
			INSTR_TIME_GET_MILLISEC(jitusage->inlining_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter))
			counters->jit_optimization_count++;
		counters->jit_optimization_time +=
			INSTR_TIME_GET_MILLISEC(jitusage->optimization_counter);

		if (INSTR_TIME_GET_MILLISEC(jitusage->emission_counter))
			counters->jit_emission_count++;
		counters->jit_emission_time +=
			INSTR_TIME_GET_MILLISEC(jitusage->emission_counter);
	}

	counters->parallel_workers_to_launch += parallel_workers_to_launch;
	counters->parallel_workers_launched += parallel_workers_launched;

	if (planOrigin == PLAN_STMT_CACHE_GENERIC)
		counters->generic_plan_calls++;
	else if (planOrigin == PLAN_STMT_CACHE_CUSTOM)
		counters->custom_plan_calls++;
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
	Oid			userid = GetUserId();
	bool		is_allowed_role = false;
	char	   *qbuffer = NULL;
	Size		qbuffer_size = 0;
	dshash_seq_status hstat;
	PgStatShared_HashEntry *hash_entry;
	pgssEntry	entrybuf;
	pgssEntry  *entry = &entrybuf;

	/*
	 * Superusers or roles with the privileges of pg_read_all_stats members
	 * are allowed
	 */
	is_allowed_role = has_privs_of_role(userid, ROLE_PG_READ_ALL_STATS);

	/* hash table must exist already */
	if (!pgss || pgStatLocal.kind_hash[PGSS_STATS_KIND] == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via \"shared_preload_libraries\"")));

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
	 * Pending stats are accumulated by the leader, so a parallel worker would
	 * only see its own.  Extension versions 1.14 and later mark this function
	 * PARALLEL RESTRICTED, but older ones still declare it PARALLEL SAFE.
	 */
	if (IsParallelWorker())
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TRANSACTION_STATE),
				 errmsg("cannot read pg_stat_statements during a parallel operation"),
				 errhint("Update the extension with ALTER EXTENSION pg_stat_statements UPDATE.")));

	/* Make this backend's pending pgss stats visible to this read. */
	pgstat_force_next_flush();

	if (showtext)
	{
		LWLockAcquire(&pgss->qtext_lock, LW_SHARED);
		qbuffer = qtext_load_file(&qbuffer_size);
	}

	dshash_seq_init(&hstat, pgStatLocal.kind_hash[PGSS_STATS_KIND], false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;
		Datum		values[PG_STAT_STATEMENTS_COLS];
		bool		nulls[PG_STAT_STATEMENTS_COLS];
		int			i = 0;
		Counters	tmp;
		double		stddev;
		int64		queryid;
		TimestampTz stats_since;
		TimestampTz minmax_stats_since;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
			continue;

		/* copy the entry to a local variable to keep locking time short */
		LWLockAcquire(&shared_entry->header.lock, LW_SHARED);
		entrybuf = shared_entry->stats;
		LWLockRelease(&shared_entry->header.lock);

		tmp = entry->counters;
		stats_since = entry->stats_since;
		minmax_stats_since = entry->minmax_stats_since;
		queryid = entry->key.queryid;

		if (stats_since == 0 || IS_STICKY(tmp))
			continue;

		memset(values, 0, sizeof(values));
		memset(nulls, 0, sizeof(nulls));

		values[i++] = ObjectIdGetDatum(entry->key.userid);
		values[i++] = ObjectIdGetDatum(entry->key.dbid);
		if (api_version >= PGSS_V1_9)
			values[i++] = BoolGetDatum(entry->key.toplevel);

		if (is_allowed_role || entry->key.userid == userid)
		{
			if (api_version >= PGSS_V1_2)
				values[i++] = Int64GetDatumFast(queryid);

			if (showtext)
			{
				char	   *qstr = qtext_fetch(entry->query_offset,
											   entry->query_len,
											   qbuffer,
											   qbuffer_size);

				if (qstr)
				{
					char	   *enc;

					enc = pg_any_to_server(qstr,
										   entry->query_len,
										   entry->encoding);

					values[i++] = CStringGetTextDatum(enc);

					if (enc != qstr)
						pfree(enc);
				}
				else
				{
					/* Just return a null if we fail to find the text */
					nulls[i++] = true;
				}
			}
			else
			{
				/* Query text not requested */
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

		/* Note that we rely on PGSS_PLAN being 0 and PGSS_EXEC being 1. */
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
		{
			values[i++] = Int64GetDatumFast(tmp.wal_buffers_full);
		}
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
			values[i++] = TimestampTzGetDatum(stats_since);
			values[i++] = TimestampTzGetDatum(minmax_stats_since);
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

	if (qbuffer)
		pfree(qbuffer);
	if (showtext)
		LWLockRelease(&pgss->qtext_lock);
}

/* Number of output arguments (columns) for pg_stat_statements_info */
#define PG_STAT_STATEMENTS_INFO_COLS	2

/*
 * Return statistics of pg_stat_statements.
 */
Datum
pg_stat_statements_info(PG_FUNCTION_ARGS)
{
	pgssGlobalStats stats;
	TupleDesc	tupdesc;
	Datum		values[PG_STAT_STATEMENTS_INFO_COLS] = {0};
	bool		nulls[PG_STAT_STATEMENTS_INFO_COLS] = {0};

	if (!pgss || pgStatLocal.kind_hash[PGSS_STATS_KIND] == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via \"shared_preload_libraries\"")));

	/* Build a tuple descriptor for our result type */
	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");

	stats = pgss_global_stats_fetch();

	values[0] = Int64GetDatum(stats.dealloc);
	values[1] = TimestampTzGetDatum(stats.stats_reset);

	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

/*
 * Allocate a new entry.
 */
static PgStat_EntryRef *
entry_alloc(const pgssHashKey *key, uint64 stats_objid, const char *query,
			int query_location, int query_len, int encoding,
			const JumbleState *jstate)
{
	PgStat_EntryRef *entry_ref;
	pgssSharedEntry *shared_entry;
	char	   *norm_query = NULL;
	uint64		live_entry_count;
	bool		do_gc = false;

	/* Make space once the estimated live entry count reaches the soft limit. */
	live_entry_count = pg_atomic_read_u64(&pgss->cur_entry_count);
	if (live_entry_count >= entry_dealloc_soft_limit())
		entry_dealloc(entry_dealloc_should_wait(live_entry_count));

	if (jstate)
		norm_query = generate_normalized_query(jstate, query, query_location,
											   &query_len);

	LWLockAcquire(&pgss->qtext_lock, LW_SHARED);
	entry_ref = pgstat_get_entry_ref(PGSS_STATS_KIND, key->dbid,
									 stats_objid, true, NULL);
	shared_entry = (pgssSharedEntry *) entry_ref->shared_stats;

	(void) pgstat_lock_entry(entry_ref, false);
	if (shared_entry->stats.stats_since == 0)
	{
		Size		query_offset = 0;
		double		usage;

		if (!qtext_store(norm_query ? norm_query : query, query_len,
						 &query_offset))
			query_len = -1;

		shared_entry->stats.key = *key;

		/*
		 * The core entry is visible before pgss metadata is initialized.
		 * Preserve any pending stats that won the core entry lock and flushed
		 * first.
		 */
		if (jstate)
		{
			if (LWLockConditionalAcquire(&pgss->dealloc_lock, LW_SHARED))
			{
				usage = pgss->cur_median_usage;
				LWLockRelease(&pgss->dealloc_lock);
			}
			else
				usage = ASSUMED_MEDIAN_INIT;
		}
		else
			usage = USAGE_INIT;

		shared_entry->stats.counters.usage += usage;
		shared_entry->stats.query_offset = query_offset;
		shared_entry->stats.query_len = query_len;
		shared_entry->stats.encoding = encoding;
		shared_entry->stats.executed = (jstate == NULL);
		shared_entry->stats.stats_since = GetCurrentTimestamp();
		shared_entry->stats.minmax_stats_since = shared_entry->stats.stats_since;

		pg_atomic_fetch_add_u64(&pgss->cur_entry_count, 1);
		do_gc = need_gc_qtexts();
	}
	pgstat_unlock_entry(entry_ref);
	LWLockRelease(&pgss->qtext_lock);

	if (norm_query)
		pfree(norm_query);

	if (do_gc)
		gc_qtexts();

	return entry_ref;
}

/*
 * qsort comparator for sorting into increasing usage order
 */
static int
entry_cmp(const void *lhs, const void *rhs)
{
	double		l_usage = ((const pgssEntryDealloc *) lhs)->usage;
	double		r_usage = ((const pgssEntryDealloc *) rhs)->usage;

	if (l_usage < r_usage)
		return -1;
	else if (l_usage > r_usage)
		return +1;
	else
		return 0;
}

/*
 * Should this backend wait for eviction before creating another entry?
 *
 * pg_stat_statements.max is treated as a target.  We allow entries to grow
 * past that target by a bounded amount.  Once the entry count reaches the soft
 * limit, backends make conditional eviction attempts and continue if another
 * backend is already evicting.  As the entry count drifts toward the hard
 * limit, an increasing fraction of backends wait for eviction; at the hard
 * limit, all do.
 */
static uint64
entry_dealloc_allowance(void)
{
	uint64		target = (uint64) pgss_max;
	uint64		allowance;

	allowance = target * USAGE_DEALLOC_BLOCK_PERCENT / 100;
	if (allowance == 0)
		allowance = 1;
	allowance = Min(allowance, (uint64) USAGE_DEALLOC_BLOCK_MAX_EXTRA);

	return allowance;
}

static uint64
entry_dealloc_soft_limit(void)
{
	return (uint64) pgss_max + Max(UINT64CONST(1),
								   entry_dealloc_allowance() / 2);
}

static uint64
entry_dealloc_hard_limit(void)
{
	return (uint64) pgss_max + entry_dealloc_allowance();
}

static bool
entry_dealloc_should_wait(uint64 live_entry_count)
{
	uint64		soft_limit = entry_dealloc_soft_limit();
	uint64		hard_limit = entry_dealloc_hard_limit();
	uint64		wait_margin;
	uint64		over_soft_limit;

	if (live_entry_count < soft_limit)
		return false;

	if (live_entry_count >= hard_limit)
		return true;

	wait_margin = hard_limit - soft_limit;
	if (wait_margin == 0)
		return true;

	over_soft_limit = live_entry_count - soft_limit + 1;
	return pg_prng_uint64_range(&pg_global_prng_state, 1, wait_margin) <=
		over_soft_limit;
}

/*
 * Collect the identity, usage and query text length of all initialized
 * entries into a palloc'd array returned in *entries_p, and return the number
 * of entries.  nalloc is the initial array size.  *mean_query_len is set to
 * the mean length of the valid query texts seen.
 *
 * If decay is true, also apply the usage decay factor to each entry.  Entry
 * locks are then only acquired conditionally, and busy entries are skipped
 * and counted in *nskipped.  *saw_dropped is set if dropped entries were
 * seen.
 */
static int
entry_collect(bool decay, Size nalloc, pgssEntryDealloc **entries_p,
			  Size *mean_query_len, bool *saw_dropped, int *nskipped)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *hash_entry;
	pgssEntryDealloc *entries;
	int			nentries = 0;
	Size		tottextlen = 0;
	int			nvalidtexts = 0;

	entries = palloc_array(pgssEntryDealloc, nalloc);

	dshash_seq_init(&hstat, pgStatLocal.kind_hash[PGSS_STATS_KIND], false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;
		pgssEntry  *entry;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
		{
			*saw_dropped = true;
			continue;
		}

		if (!decay)
			LWLockAcquire(&shared_entry->header.lock, LW_SHARED);
		else if (!LWLockConditionalAcquire(&shared_entry->header.lock,
										   LW_EXCLUSIVE))
		{
			(*nskipped)++;
			continue;
		}

		entry = &shared_entry->stats;
		if (entry->stats_since == 0)
		{
			LWLockRelease(&shared_entry->header.lock);
			continue;
		}

		if (decay)
		{
			/* "Sticky" entries get a different usage decay rate. */
			if (!entry->executed)
				entry->counters.usage *= STICKY_DECREASE_FACTOR;
			else
				entry->counters.usage *= USAGE_DECREASE_FACTOR;
		}

		if ((Size) nentries >= nalloc)
		{
			nalloc *= 2;
			entries = repalloc_array(entries, pgssEntryDealloc, nalloc);
		}

		entries[nentries].key = hash_entry->key;
		entries[nentries].usage = entry->counters.usage;
		nentries++;

		/* In the mean length computation, ignore dropped texts. */
		if (entry->query_len >= 0)
		{
			tottextlen += entry->query_len + 1;
			nvalidtexts++;
		}

		LWLockRelease(&shared_entry->header.lock);
	}
	dshash_seq_term(&hstat);

	if (nvalidtexts > 0)
		*mean_query_len = tottextlen / nvalidtexts;
	else
		*mean_query_len = ASSUMED_LENGTH_INIT;

	*entries_p = entries;
	return nentries;
}

/*
 * Deallocate least-used entries.
 */
static void
entry_dealloc(bool wait)
{
	pgssEntryDealloc *entries;
	int			nvictims;
	int			i;
	int			nentries;
	Size		mean_query_len;
	uint64		live_entry_count;
	uint64		nlive;
	uint64		soft_limit;
	bool		request_gc = false;
	bool		saw_dropped = false;
	int			nskipped = 0;

	/*
	 * Avoid having backends queue up behind eviction until the entry count
	 * gets far enough past the target.  Once it does, wait here to apply
	 * stronger pressure before creating another entry.
	 */
	if (wait)
		LWLockAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE);
	else if (!LWLockConditionalAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE))
		return;

	soft_limit = entry_dealloc_soft_limit();
	live_entry_count = pg_atomic_read_u64(&pgss->cur_entry_count);
	if (live_entry_count < soft_limit)
	{
		LWLockRelease(&pgss->dealloc_lock);
		return;
	}

	/*
	 * Sort entries by usage and deallocate USAGE_DEALLOC_PERCENT of them.
	 * While we're scanning the table, apply the decay factor to the usage
	 * values, and update the mean query length.
	 *
	 * Note that the mean query length is almost immediately obsolete, since
	 * we compute it before not after discarding the least-used entries.
	 * Hopefully, that doesn't affect the mean too much; it doesn't seem worth
	 * making two passes to get a more current result.  Likewise, the new
	 * cur_median_usage includes the entries we're about to zap.
	 */
	nentries = entry_collect(true, (Size) live_entry_count, &entries,
							 &mean_query_len, &saw_dropped, &nskipped);

	/*
	 * Entries that were busy during the scan are live too.  The count also
	 * drifts when entries are dropped by others, e.g. by DROP DATABASE, so
	 * resynchronize it with what the scan found.
	 */
	nlive = (uint64) nentries + (uint64) nskipped;
	if (nlive < soft_limit)
	{
		entry_adjust_count(live_entry_count, nlive);

		pfree(entries);
		LWLockRelease(&pgss->dealloc_lock);

		if (saw_dropped)
			pgstat_request_entry_refs_gc();
		return;
	}

	/* Sort into increasing order by usage */
	qsort(entries, nentries, sizeof(pgssEntryDealloc), entry_cmp);

	/* Record the (approximate) median usage */
	if (nentries > 0)
		pgss->cur_median_usage = entries[nentries / 2].usage;
	/* Record the mean query length */
	pgss->mean_query_len = mean_query_len;

	/* Now zap an appropriate fraction of lowest-usage entries */
	nvictims = Max(10, nentries * USAGE_DEALLOC_PERCENT / 100);
	if ((uint64) nentries >= soft_limit)
		nvictims = Max(nvictims, (int) ((uint64) nentries - soft_limit + 1));
	nvictims = Min(nvictims, nentries);

	for (i = 0; i < nvictims; i++)
	{
		if (!pgstat_drop_entry(PGSS_STATS_KIND, entries[i].key.dboid,
							   entries[i].key.objid, true))
			request_gc = true;
	}
	entry_adjust_count(live_entry_count, nlive - (uint64) nvictims);

	pfree(entries);

	pgss_global_stats_increment_dealloc();

	LWLockRelease(&pgss->dealloc_lock);

	if (request_gc)
		pgstat_request_entry_refs_gc();
}

/*
 * Given a query string (not necessarily null-terminated), allocate a new
 * entry in the external query text file and store the string there.
 *
 * If successful, returns true, and stores the new entry's offset in the file
 * into *query_offset.
 */
static bool
qtext_store(const char *query, int query_len, Size *query_offset)
{
	Size		off;
	int			fd = -1;

	/*
	 * We use a spinlock to protect extent, so that multiple processes may
	 * execute this function concurrently.
	 */
	SpinLockAcquire(&pgss->mutex);
	off = pgss->extent;
	if (unlikely(off >= MaxAllocHugeSize ||
				 (Size) query_len >= MaxAllocHugeSize - off))
	{
		SpinLockRelease(&pgss->mutex);
		errno = EFBIG;
		goto error;
	}
	pgss->extent += (Size) query_len + 1;
	SpinLockRelease(&pgss->mutex);

	*query_offset = off;

	/* Now write the data into the successfully-reserved part of the file */
	fd = OpenTransientFile(PGSS_TEXT_FILE, O_RDWR | O_CREAT | PG_BINARY);
	if (fd < 0)
		goto error;

	if (pg_pwrite(fd, query, query_len, off) != query_len)
		goto error;
	if (pg_pwrite(fd, "\0", 1, off + query_len) != 1)
		goto error;

	CloseTransientFile(fd);

	return true;

error:
	ereport(LOG,
			(errcode_for_file_access(),
			 errmsg("could not write file \"%s\": %m",
					PGSS_TEXT_FILE)));

	if (fd >= 0)
		CloseTransientFile(fd);

	return false;
}

/*
 * Read the external query text file into a palloc'd buffer.
 *
 * Returns NULL (without throwing an error) if unable to read, eg
 * file not there or insufficient memory.
 *
 * On success, the buffer size is also returned into *buffer_size.
 */
static char *
qtext_load_file(Size *buffer_size)
{
	char	   *buf;
	int			fd;
	struct stat stat;
	Size		nread;

	fd = OpenTransientFile(PGSS_TEXT_FILE, O_RDONLY | PG_BINARY);
	if (fd < 0)
	{
		if (errno != ENOENT)
			ereport(LOG,
					(errcode_for_file_access(),
					 errmsg("could not read file \"%s\": %m",
							PGSS_TEXT_FILE)));
		return NULL;
	}

	/* Get file length */
	if (fstat(fd, &stat))
	{
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not stat file \"%s\": %m",
						PGSS_TEXT_FILE)));
		CloseTransientFile(fd);
		return NULL;
	}

	/* Allocate buffer; beware that off_t might be wider than size_t */
	if (stat.st_size <= MaxAllocHugeSize)
		buf = (char *) palloc_extended(stat.st_size, MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM);
	else
		buf = NULL;
	if (buf == NULL)
	{
		ereport(LOG,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory"),
				 errdetail("Could not allocate enough memory to read file \"%s\".",
						   PGSS_TEXT_FILE)));
		CloseTransientFile(fd);
		return NULL;
	}

	/*
	 * OK, slurp in the file.  Windows fails if we try to read more than
	 * INT_MAX bytes at once, and other platforms might not like that either,
	 * so read a very large file in 1GB segments.
	 */
	nread = 0;
	while (nread < stat.st_size)
	{
		int			toread = Min(1024 * 1024 * 1024, stat.st_size - nread);

		/*
		 * If we get a short read and errno doesn't get set, the reason is
		 * probably that garbage collection truncated the file since we did
		 * the fstat(), so we don't log a complaint --- but we don't return
		 * the data, either, since it's most likely corrupt due to concurrent
		 * writes from garbage collection.
		 */
		errno = 0;
		if (read(fd, buf + nread, toread) != toread)
		{
			if (errno)
				ereport(LOG,
						(errcode_for_file_access(),
						 errmsg("could not read file \"%s\": %m",
								PGSS_TEXT_FILE)));
			pfree(buf);
			CloseTransientFile(fd);
			return NULL;
		}
		nread += toread;
	}

	if (CloseTransientFile(fd) != 0)
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not close file \"%s\": %m", PGSS_TEXT_FILE)));

	*buffer_size = nread;
	return buf;
}

/*
 * Locate a query text in the file image previously read by qtext_load_file().
 *
 * We validate the given offset/length, and return NULL if bogus.  Otherwise,
 * the result points to a null-terminated string within the buffer.
 */
static char *
qtext_fetch(Size query_offset, int query_len,
			char *buffer, Size buffer_size)
{
	/* File read failed? */
	if (buffer == NULL)
		return NULL;
	/* Bogus offset/length? */
	if (query_len < 0 ||
		query_offset + query_len >= buffer_size)
		return NULL;
	/* As a further sanity check, make sure there's a trailing null */
	if (buffer[query_offset + query_len] != '\0')
		return NULL;
	/* Space reserved but not yet written when the file was read is zeroes */
	if (strnlen(buffer + query_offset, query_len) != (size_t) query_len)
		return NULL;
	/* Looks OK */
	return buffer + query_offset;
}

/*
 * Reset the external query text file.
 *
 * Caller must hold qtext_lock exclusively so no new entry can be initialized
 * with an offset into the file while it is being truncated.
 */
static void
qtext_reset_file(void)
{
	FILE	   *qfile;

	Assert(LWLockHeldByMeInMode(&pgss->qtext_lock, LW_EXCLUSIVE));

	qfile = AllocateFile(PGSS_TEXT_FILE, PG_BINARY_W);
	if (qfile == NULL)
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not create file \"%s\": %m",
						PGSS_TEXT_FILE)));
	else
	{
		/* If ftruncate fails, log it, but it's not a fatal problem */
		if (ftruncate(fileno(qfile), 0) != 0)
			ereport(LOG,
					(errcode_for_file_access(),
					 errmsg("could not truncate file \"%s\": %m",
							PGSS_TEXT_FILE)));

		FreeFile(qfile);
	}

	SpinLockAcquire(&pgss->mutex);
	pgss->extent = 0;
	SpinLockRelease(&pgss->mutex);
}

static bool
need_gc_qtexts(void)
{
	Size		extent;

	/* Read shared extent pointer */
	SpinLockAcquire(&pgss->mutex);
	extent = pgss->extent;
	SpinLockRelease(&pgss->mutex);

	/*
	 * Don't proceed if file does not exceed 512 bytes per possible entry.
	 *
	 * Here and in the next test, 32-bit machines have overflow hazards if
	 * pgss_max and/or mean_query_len are large.  Force the multiplications
	 * and comparisons to be done in uint64 arithmetic to forestall trouble.
	 */
	if ((uint64) extent < (uint64) 512 * pgss_max)
		return false;

	/*
	 * Don't proceed if file is less than about 50% bloat.  Nothing can or
	 * should be done in the event of unusually large query texts accounting
	 * for file's large size.  We go to the trouble of maintaining the mean
	 * query length in order to prevent garbage collection from thrashing
	 * uselessly.
	 */
	if ((uint64) extent < (uint64) pgss->mean_query_len * pgss_max * 2)
		return false;

	return true;
}

/*
 * Garbage-collect orphaned query texts in external file.
 *
 * This won't be called often in the typical case.  Reset and restart can
 * compact the file, but long-running high-churn servers still need in-place
 * cleanup.
 *
 * At the first sign of trouble we unlink the query text file to get a clean
 * slate (although existing statistics are retained), rather than risk
 * thrashing by allowing the same problem case to recur indefinitely.
 */
static void
gc_qtexts(void)
{
	char	   *qbuffer;
	Size		qbuffer_size;
	FILE	   *qfile = NULL;
	dshash_table *hash = pgStatLocal.kind_hash[PGSS_STATS_KIND];
	dshash_seq_status hstat;
	PgStatShared_HashEntry *hash_entry;
	Size		extent;
	int			nentries;

	LWLockAcquire(&pgss->qtext_lock, LW_EXCLUSIVE);
	LWLockAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE);

	/*
	 * When called from entry_alloc(), some other session might have proceeded
	 * with garbage collection first.
	 */
	if (!need_gc_qtexts())
	{
		LWLockRelease(&pgss->dealloc_lock);
		LWLockRelease(&pgss->qtext_lock);
		return;
	}

	/*
	 * Load the old texts file.  If we fail (out of memory, for instance),
	 * invalidate query texts.  Hopefully this is rare.  It might seem better
	 * to leave things alone on an OOM failure, but the problem is that the
	 * file is only going to get bigger; hoping for a future non-OOM result is
	 * risky and can easily lead to complete denial of service.
	 */
	qbuffer = qtext_load_file(&qbuffer_size);
	if (qbuffer == NULL)
		goto gc_fail;

	/*
	 * We overwrite the query texts file in place, so as to reduce the risk of
	 * an out-of-disk-space failure.  Since the file is guaranteed not to get
	 * larger, this should always work on traditional filesystems; though we
	 * could still lose on copy-on-write filesystems.
	 */
	qfile = AllocateFile(PGSS_TEXT_FILE, PG_BINARY_W);
	if (qfile == NULL)
	{
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not write file \"%s\": %m",
						PGSS_TEXT_FILE)));
		goto gc_fail;
	}

	extent = 0;
	nentries = 0;

	dshash_seq_init(&hstat, hash, false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;
		pgssEntry  *entry;
		int			query_len;
		char	   *qry;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
			continue;

		LWLockAcquire(&shared_entry->header.lock, LW_EXCLUSIVE);
		entry = &shared_entry->stats;

		if (entry->stats_since == 0)
		{
			LWLockRelease(&shared_entry->header.lock);
			continue;
		}

		query_len = entry->query_len;
		qry = qtext_fetch(entry->query_offset, query_len,
						  qbuffer, qbuffer_size);

		if (qry == NULL)
		{
			/* Trouble ... drop the text */
			entry->query_offset = 0;
			entry->query_len = -1;
			LWLockRelease(&shared_entry->header.lock);
			/* entry will not be counted in mean query length computation */
			continue;
		}

		if (fwrite(qry, 1, query_len + 1, qfile) != query_len + 1)
		{
			ereport(LOG,
					(errcode_for_file_access(),
					 errmsg("could not write file \"%s\": %m",
							PGSS_TEXT_FILE)));
			LWLockRelease(&shared_entry->header.lock);
			dshash_seq_term(&hstat);
			goto gc_fail;
		}

		entry->query_offset = extent;
		extent += query_len + 1;
		nentries++;

		LWLockRelease(&shared_entry->header.lock);
	}
	dshash_seq_term(&hstat);

	/*
	 * Truncate away any now-unused space.  If this fails for some odd reason,
	 * we log it, but there's no need to fail.
	 */
	if (ftruncate(fileno(qfile), extent) != 0)
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not truncate file \"%s\": %m",
						PGSS_TEXT_FILE)));

	if (FreeFile(qfile))
	{
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not write file \"%s\": %m",
						PGSS_TEXT_FILE)));
		qfile = NULL;
		goto gc_fail;
	}

	elog(DEBUG1, "pgss gc of queries file shrunk size from %zu to %zu",
		 pgss->extent, extent);

	/* Reset the shared extent pointer */
	pgss->extent = extent;

	/*
	 * Also update the mean query length, to be sure that need_gc_qtexts()
	 * won't still think we have a problem.
	 */
	if (nentries > 0)
		pgss->mean_query_len = extent / nentries;
	else
		pgss->mean_query_len = ASSUMED_LENGTH_INIT;

	pfree(qbuffer);

	LWLockRelease(&pgss->dealloc_lock);
	LWLockRelease(&pgss->qtext_lock);

	return;

gc_fail:
	/* clean up resources */
	if (qfile)
		FreeFile(qfile);
	if (qbuffer)
		pfree(qbuffer);

	/*
	 * Since the contents of the external file are now uncertain, mark all
	 * entries as having invalid texts.
	 */
	dshash_seq_init(&hstat, hash, false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
			continue;

		LWLockAcquire(&shared_entry->header.lock, LW_EXCLUSIVE);
		shared_entry->stats.query_offset = 0;
		shared_entry->stats.query_len = -1;
		LWLockRelease(&shared_entry->header.lock);
	}
	dshash_seq_term(&hstat);

	/*
	 * Destroy the query text file and create a new, empty one
	 */
	(void) unlink(PGSS_TEXT_FILE);
	qfile = AllocateFile(PGSS_TEXT_FILE, PG_BINARY_W);
	if (qfile == NULL)
		ereport(LOG,
				(errcode_for_file_access(),
				 errmsg("could not recreate file \"%s\": %m",
						PGSS_TEXT_FILE)));
	else
		FreeFile(qfile);

	/* Reset the shared extent pointer */
	pgss->extent = 0;

	/* Reset mean_query_len to match the new state */
	pgss->mean_query_len = ASSUMED_LENGTH_INIT;

	LWLockRelease(&pgss->dealloc_lock);
	LWLockRelease(&pgss->qtext_lock);
}

/*
 * Reset entries corresponding to parameters passed.
 */
static TimestampTz
entry_reset(Oid userid, Oid dbid, int64 queryid, bool minmax_only)
{
	pgssHashKey key;
	bool		single_entry;
	TimestampTz stats_reset;
	uint64		visible_entries;

	if (!pgss || pgStatLocal.kind_hash[PGSS_STATS_KIND] == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_stat_statements must be loaded via \"shared_preload_libraries\"")));

	/*
	 * Pending stats are accumulated by the leader, so a reset run in a
	 * parallel worker could not discard them.  Extension versions 1.14 and
	 * later mark the reset functions PARALLEL RESTRICTED, but older ones
	 * still declare them PARALLEL SAFE.
	 */
	if (IsParallelWorker())
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TRANSACTION_STATE),
				 errmsg("cannot reset pg_stat_statements during a parallel operation"),
				 errhint("Update the extension with ALTER EXTENSION pg_stat_statements UPDATE.")));

	stats_reset = GetCurrentTimestamp();

	/*
	 * With all of userid, dbid and queryid given, the top-level and nested
	 * entries can be looked up directly.  Otherwise, key acts as a filter in
	 * which zero fields match anything and toplevel is ignored.
	 */
	memset(&key, 0, sizeof(pgssHashKey));
	key.userid = userid;
	key.dbid = dbid;
	key.queryid = queryid;
	single_entry = (userid != 0 && dbid != 0 && queryid != INT64CONST(0));

	if (minmax_only)
	{
		if (single_entry)
		{
			for (int toplevel = 0; toplevel <= 1; toplevel++)
			{
				key.toplevel = (bool) toplevel;
				pgss_stats_minmax_reset_entry(&key, stats_reset);
			}
		}
		else
			pgss_stats_minmax_reset_matching(&key, stats_reset);

		return stats_reset;
	}

	/*
	 * Block first-time query text mappings while entries are dropped, and
	 * eviction so that it doesn't adjust the entry count concurrently.
	 */
	LWLockAcquire(&pgss->qtext_lock, LW_EXCLUSIVE);
	LWLockAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE);

	if (single_entry)
	{
		bool		request_gc = false;

		for (int toplevel = 0; toplevel <= 1; toplevel++)
		{
			key.toplevel = (bool) toplevel;
			if (!pgss_drop_entry(&key))
				request_gc = true;
		}

		if (request_gc)
			pgstat_request_entry_refs_gc();
	}
	else
		pgstat_drop_matching_entries(PGSS_STATS_KIND,
									 pgss_stats_entry_matches,
									 PointerGetDatum(&key));

	visible_entries = pgss_stats_count_visible_entries();
	pg_atomic_write_u64(&pgss->cur_entry_count, visible_entries);

	if (visible_entries == 0)
	{
		qtext_reset_file();
		entry_reset_usage_state();
		pgss_global_stats_reset(stats_reset);
	}

	LWLockRelease(&pgss->dealloc_lock);
	LWLockRelease(&pgss->qtext_lock);

	return stats_reset;
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

/*
 * Move cur_entry_count from old_count, as read at the start of an eviction
 * pass, to new_count.  The caller must hold dealloc_lock, so the only
 * concurrent changes are increments by entry_alloc(), and those are kept.
 */
static void
entry_adjust_count(uint64 old_count, uint64 new_count)
{
	Assert(LWLockHeldByMeInMode(&pgss->dealloc_lock, LW_EXCLUSIVE));

	if (new_count != old_count)
		pg_atomic_fetch_add_u64(&pgss->cur_entry_count,
								(int64) new_count - (int64) old_count);
}

static void
entry_rebuild_runtime_state(void)
{
	pgssEntryDealloc *entries;
	int			nentries;
	double		cur_median_usage = ASSUMED_MEDIAN_INIT;
	Size		mean_query_len;
	bool		saw_dropped = false;
	int			nskipped = 0;

	nentries = entry_collect(false, 128, &entries, &mean_query_len,
							 &saw_dropped, &nskipped);

	if (nentries > 0)
	{
		qsort(entries, nentries, sizeof(pgssEntryDealloc), entry_cmp);
		cur_median_usage = entries[nentries / 2].usage;
	}

	LWLockAcquire(&pgss->dealloc_lock, LW_EXCLUSIVE);
	pg_atomic_write_u64(&pgss->cur_entry_count, nentries);
	pgss->cur_median_usage = cur_median_usage;
	pgss->mean_query_len = mean_query_len;
	LWLockRelease(&pgss->dealloc_lock);

	pfree(entries);
}

/*
 * Reset the eviction state after all entries are gone.  Caller must hold
 * dealloc_lock.
 */
static void
entry_reset_usage_state(void)
{
	Assert(LWLockHeldByMeInMode(&pgss->dealloc_lock, LW_EXCLUSIVE));

	pg_atomic_write_u64(&pgss->cur_entry_count, 0);
	pgss->cur_median_usage = ASSUMED_MEDIAN_INIT;
	pgss->mean_query_len = ASSUMED_LENGTH_INIT;
}

static void
pgss_global_stats_init_shmem_cb(void *stats)
{
	pgssGlobalStats *stats_shmem = (pgssGlobalStats *) stats;

	stats_shmem->dealloc = 0;
	stats_shmem->stats_reset = GetCurrentTimestamp();
}

static inline pgssGlobalStats *
pgss_global_stats_shmem(void)
{
	return (pgssGlobalStats *)
		pgstat_get_custom_shmem_data(PGSS_GLOBAL_STATS_KIND);
}

static void
pgss_global_stats_snapshot_cb(void)
{
	pgssGlobalStats *stats_shmem;
	pgssGlobalStats *stats_snap;

	stats_shmem = pgss_global_stats_shmem();
	stats_snap = (pgssGlobalStats *)
		pgstat_get_custom_snapshot_data(PGSS_GLOBAL_STATS_KIND);

	SpinLockAcquire(&pgss->mutex);
	*stats_snap = *stats_shmem;
	SpinLockRelease(&pgss->mutex);
}

static void
pgss_global_stats_reset(TimestampTz stats_reset)
{
	pgssGlobalStats *stats_shmem;

	stats_shmem = pgss_global_stats_shmem();

	SpinLockAcquire(&pgss->mutex);
	stats_shmem->dealloc = 0;
	stats_shmem->stats_reset = stats_reset;
	SpinLockRelease(&pgss->mutex);
}

static pgssGlobalStats
pgss_global_stats_fetch(void)
{
	pgssGlobalStats *stats_shmem;
	pgssGlobalStats stats;

	stats_shmem = pgss_global_stats_shmem();

	SpinLockAcquire(&pgss->mutex);
	stats = *stats_shmem;
	SpinLockRelease(&pgss->mutex);

	return stats;
}

static void
pgss_global_stats_increment_dealloc(void)
{
	pgssGlobalStats *stats_shmem;

	stats_shmem = pgss_global_stats_shmem();

	SpinLockAcquire(&pgss->mutex);
	stats_shmem->dealloc += 1;
	SpinLockRelease(&pgss->mutex);
}

/*
 * XXX: pgstat's variable-numbered API identifies entries by a uint64 objid.
 * For now, hash the full pg_stat_statements key into that value.  Before this
 * becomes committable, decide whether collision handling belongs here or
 * whether core needs a wider custom-key API.
 */
static uint64
pgss_stats_objid(const pgssHashKey *key)
{
	return hash_bytes_extended((const unsigned char *) key,
							   sizeof(pgssHashKey), 0);
}

/*
 * Return the shared entry for a hash entry seen during a scan of our stats
 * hash, or NULL if it has been dropped.
 */
static inline pgssSharedEntry *
pgss_get_shared_entry(PgStatShared_HashEntry *hash_entry)
{
	if (hash_entry->dropped)
		return NULL;

	return (pgssSharedEntry *)
		dsa_get_address(pgStatLocal.kind_dsa[PGSS_STATS_KIND],
						hash_entry->body);
}

static inline bool
pgss_stats_entry_ref_is_stale(PgStat_EntryRef *entry_ref)
{
	/*
	 * XXX: core should expose this stale-ref check for custom stats kinds.
	 */
	if (entry_ref->shared_entry == NULL)
		return true;
	if (entry_ref->shared_entry->dropped)
		return true;
	if (pg_atomic_read_u32(&entry_ref->shared_entry->generation) !=
		entry_ref->generation)
		return true;

	return false;
}

static PgStat_EntryRef *
pgss_stats_get_entry_ref(const pgssHashKey *key, uint64 stats_objid)
{
	PgStat_EntryRef *entry_ref;

	/* Release references found stale by earlier flushes. */
	if (unlikely(pgss_refs_gc_pending))
	{
		pgstat_request_entry_refs_gc();
		pgss_refs_gc_pending = false;
	}

	entry_ref = pgstat_get_entry_ref(PGSS_STATS_KIND, key->dbid,
									 stats_objid, false, NULL);
	if (entry_ref == NULL || likely(!pgss_stats_entry_ref_is_stale(entry_ref)))
		return entry_ref;

	if (entry_ref->pending != NULL)
		pgstat_delete_pending_entry(entry_ref);

	pgstat_request_entry_refs_gc();

	entry_ref = pgstat_get_entry_ref(PGSS_STATS_KIND, key->dbid,
									 stats_objid, false, NULL);
	if (entry_ref != NULL && unlikely(pgss_stats_entry_ref_is_stale(entry_ref)))
		return NULL;

	return entry_ref;
}

/*
 * Drop the entry for the given key, if any.  Returns false if the entry could
 * not be freed yet, in which case the caller should request a GC of entry
 * references.
 */
static bool
pgss_drop_entry(const pgssHashKey *key)
{
	return pgstat_drop_entry(PGSS_STATS_KIND, key->dbid,
							 pgss_stats_objid(key), true);
}

/*
 * Does an initialized entry match the reset filter?  Zero fields of match
 * match anything, and toplevel is ignored.
 */
static bool
pgss_entry_matches(const pgssEntry *entry, const pgssHashKey *match)
{
	if (entry->stats_since == 0)
		return false;

	if (match->userid != 0 && entry->key.userid != match->userid)
		return false;
	if (match->dbid != 0 && entry->key.dbid != match->dbid)
		return false;
	if (match->queryid != INT64CONST(0) &&
		entry->key.queryid != match->queryid)
		return false;

	return true;
}

static bool
pgss_stats_entry_matches(PgStatShared_HashEntry *hash_entry, Datum match_data)
{
	pgssHashKey *match = (pgssHashKey *) DatumGetPointer(match_data);
	pgssSharedEntry *shared_entry;
	bool		matches;

	if (match->dbid != 0 && hash_entry->key.dboid != match->dbid)
		return false;

	/*
	 * This also matches entries with stats_since == 0.  The caller holds
	 * qtext_lock exclusively, so none can be partway through initialization,
	 * and any others are not usable anyway, so dropping them is fine.
	 */
	if (match->userid == 0 && match->queryid == INT64CONST(0))
		return true;

	shared_entry = pgss_get_shared_entry(hash_entry);
	if (shared_entry == NULL)
		return false;

	LWLockAcquire(&shared_entry->header.lock, LW_SHARED);
	matches = pgss_entry_matches(&shared_entry->stats, match);
	LWLockRelease(&shared_entry->header.lock);

	return matches;
}

static uint64
pgss_stats_count_visible_entries(void)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *hash_entry;
	uint64		count = 0;

	dshash_seq_init(&hstat, pgStatLocal.kind_hash[PGSS_STATS_KIND], false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
			continue;

		LWLockAcquire(&shared_entry->header.lock, LW_SHARED);
		if (shared_entry->stats.stats_since != 0)
			count++;
		LWLockRelease(&shared_entry->header.lock);
	}
	dshash_seq_term(&hstat);

	return count;
}

static void
pgss_stats_reset_minmax(pgssEntry *entry, TimestampTz stats_reset)
{
	for (int kind = 0; kind < PGSS_NUMKIND; kind++)
	{
		entry->counters.max_time[kind] = 0;
		entry->counters.min_time[kind] = 0;
	}
	entry->minmax_stats_since = stats_reset;
}

static void
pgss_stats_minmax_reset_entry(const pgssHashKey *key, TimestampTz stats_reset)
{
	PgStat_EntryRef *entry_ref;
	pgssSharedEntry *shared_entry;

	entry_ref = pgss_stats_get_entry_ref(key, pgss_stats_objid(key));
	if (entry_ref == NULL)
		return;

	shared_entry = (pgssSharedEntry *) entry_ref->shared_stats;

	(void) pgstat_lock_entry(entry_ref, false);
	if (shared_entry->stats.stats_since != 0)
		pgss_stats_reset_minmax(&shared_entry->stats, stats_reset);
	pgstat_unlock_entry(entry_ref);

	(void) pgstat_release_entry_ref_if_unused(entry_ref);
}

static void
pgss_stats_minmax_reset_matching(const pgssHashKey *match,
								 TimestampTz stats_reset)
{
	dshash_seq_status hstat;
	PgStatShared_HashEntry *hash_entry;

	dshash_seq_init(&hstat, pgStatLocal.kind_hash[PGSS_STATS_KIND], false);
	while ((hash_entry = dshash_seq_next(&hstat)) != NULL)
	{
		pgssSharedEntry *shared_entry;

		if (match->dbid != 0 && hash_entry->key.dboid != match->dbid)
			continue;

		shared_entry = pgss_get_shared_entry(hash_entry);
		if (shared_entry == NULL)
			continue;

		LWLockAcquire(&shared_entry->header.lock, LW_EXCLUSIVE);
		if (pgss_entry_matches(&shared_entry->stats, match))
			pgss_stats_reset_minmax(&shared_entry->stats, stats_reset);
		LWLockRelease(&shared_entry->header.lock);
	}
	dshash_seq_term(&hstat);
}

/*
 * Flush pending pg_stat_statements statistics to shared memory.
 */
static PgStat_FlushResult
pgss_stats_flush_pending(PgStat_EntryRef *entry_ref, bool nowait,
						 bool xact_boundary)
{
	pgssPendingEntry *pending_entry;
	pgssSharedEntry *shared_entry;
	pgssEntry  *shared_stats;

	pending_entry = (pgssPendingEntry *) entry_ref->pending;

	if (unlikely(pgss_stats_entry_ref_is_stale(entry_ref)))
	{
		memset(pending_entry, 0, sizeof(pgssPendingEntry));
		pgss_refs_gc_pending = true;
		return PGSTAT_FLUSH_DONE;
	}

	shared_entry = (pgssSharedEntry *) entry_ref->shared_stats;
	shared_stats = &shared_entry->stats;

	if (!pgstat_lock_entry(entry_ref, nowait))
		return PGSTAT_FLUSH_LOCK_CONFLICT;

	/*
	 * The entry may have been dropped, and re-created by another backend,
	 * while we waited for the lock.  Discard our pending stats in that case,
	 * like above, so that they don't land in the new entry.
	 */
	if (unlikely(pgss_stats_entry_ref_is_stale(entry_ref)))
	{
		pgstat_unlock_entry(entry_ref);
		memset(pending_entry, 0, sizeof(pgssPendingEntry));
		pgss_refs_gc_pending = true;
		return PGSTAT_FLUSH_DONE;
	}

	/*
	 * A flush can win the core entry lock before first-time pgss metadata
	 * initialization.  First-time entry initialization preserves counters in
	 * that case.
	 */
	for (int kind = PGSS_PLAN; kind <= PGSS_EXEC; kind++)
		pgss_stats_flush_kind(&shared_stats->counters,
							  &pending_entry->counters,
							  (pgssStoreKind) kind,
							  shared_stats->minmax_stats_since,
							  pending_entry->minmax_since[kind]);

	PGSS_COUNTER_ACCUM_FIELDS(PGSS_ACCUM_COUNTER_FIELD,
							  &shared_stats->counters,
							  &pending_entry->counters);

	pgstat_unlock_entry(entry_ref);

	memset(pending_entry, 0, sizeof(pgssPendingEntry));

	return PGSTAT_FLUSH_DONE;
}

static void
pgss_stats_flush_kind(Counters *dst, const Counters *src, pgssStoreKind kind,
					  TimestampTz minmax_stats_since,
					  TimestampTz minmax_since)
{
	PgStat_Counter old_calls = dst->calls[kind];
	PgStat_Counter new_calls = src->calls[kind];
	PgStat_Counter total_calls;
	bool		merge_minmax;

	if (new_calls == 0)
		return;
	Assert(minmax_since != 0);

	total_calls = old_calls + new_calls;
	merge_minmax = minmax_since >= minmax_stats_since;

	dst->total_time[kind] += src->total_time[kind];

	if (merge_minmax)
	{
		if (old_calls == 0 ||
			(dst->min_time[kind] == 0 && dst->max_time[kind] == 0))
		{
			dst->min_time[kind] = src->min_time[kind];
			dst->max_time[kind] = src->max_time[kind];
		}
		else
		{
			if (dst->min_time[kind] > src->min_time[kind])
				dst->min_time[kind] = src->min_time[kind];
			if (dst->max_time[kind] < src->max_time[kind])
				dst->max_time[kind] = src->max_time[kind];
		}
	}

	if (old_calls == 0)
	{
		dst->mean_time[kind] = src->mean_time[kind];
		dst->sum_var_time[kind] = src->sum_var_time[kind];
	}
	else
	{
		/*
		 * Chan's parallel variance algorithm: combine two sets of (count,
		 * mean, sum of squared deviations).
		 */
		double		delta = src->mean_time[kind] - dst->mean_time[kind];

		dst->mean_time[kind] +=
			delta * ((double) new_calls / (double) total_calls);
		dst->sum_var_time[kind] += src->sum_var_time[kind] +
			delta * delta * (double) old_calls * (double) new_calls /
			(double) total_calls;
	}

	dst->calls[kind] = total_calls;
}
