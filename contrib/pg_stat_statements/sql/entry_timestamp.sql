--
-- statement timestamps
--

-- planning time is needed during tests
SET pg_stat_statements.track_planning = TRUE;

SELECT 1 AS "STMTTS1";
SELECT now() AS ref_ts \gset
SELECT 1,2 AS "STMTTS2";
SELECT stats_since >= :'ref_ts', count(*) FROM pg_stat_statements
WHERE query LIKE '%STMTTS%'
GROUP BY stats_since >= :'ref_ts'
ORDER BY stats_since >= :'ref_ts';

SELECT now() AS ref_ts \gset
SELECT
  count(*) as total,
  count(*) FILTER (
    WHERE min_plan_time + max_plan_time = 0
  ) as minmax_plan_zero,
  count(*) FILTER (
    WHERE min_exec_time + max_exec_time = 0
  ) as minmax_exec_zero,
  count(*) FILTER (
    WHERE minmax_stats_since >= :'ref_ts'
  ) as minmax_stats_since_after_ref,
  count(*) FILTER (
    WHERE stats_since >= :'ref_ts'
  ) as stats_since_after_ref
FROM pg_stat_statements
WHERE query LIKE '%STMTTS%';

-- Perform single min/max reset
SELECT pg_stat_statements_reset(0, 0, queryid, true) AS minmax_reset_ts
FROM pg_stat_statements
WHERE query LIKE '%STMTTS1%' \gset

-- check
SELECT
  count(*) as total,
  count(*) FILTER (
    WHERE min_plan_time + max_plan_time = 0
  ) as minmax_plan_zero,
  count(*) FILTER (
    WHERE min_exec_time + max_exec_time = 0
  ) as minmax_exec_zero,
  count(*) FILTER (
    WHERE minmax_stats_since >= :'ref_ts'
  ) as minmax_stats_since_after_ref,
  count(*) FILTER (
    WHERE stats_since >= :'ref_ts'
  ) as stats_since_after_ref
FROM pg_stat_statements
WHERE query LIKE '%STMTTS%';

-- check minmax reset timestamps
SELECT
query, minmax_stats_since = :'minmax_reset_ts' AS reset_ts_match
FROM pg_stat_statements
WHERE query LIKE '%STMTTS%'
ORDER BY query COLLATE "C";

-- check that minmax reset does not set stats_reset
SELECT
stats_reset = :'minmax_reset_ts' AS stats_reset_ts_match
FROM pg_stat_statements_info;

-- Perform common min/max reset
SELECT pg_stat_statements_reset(0, 0, 0, true) AS minmax_reset_ts \gset

-- check again
SELECT
  count(*) as total,
  count(*) FILTER (
    WHERE min_plan_time + max_plan_time = 0
  ) as minmax_plan_zero,
  count(*) FILTER (
    WHERE min_exec_time + max_exec_time = 0
  ) as minmax_exec_zero,
  count(*) FILTER (
    WHERE minmax_stats_since >= :'ref_ts'
  ) as minmax_ts_after_ref,
  count(*) FILTER (
    WHERE minmax_stats_since = :'minmax_reset_ts'
  ) as minmax_ts_match,
  count(*) FILTER (
    WHERE stats_since >= :'ref_ts'
  ) as stats_since_after_ref
FROM pg_stat_statements
WHERE query LIKE '%STMTTS%';

-- Execute first query once more to check stats update
SELECT 1 AS "STMTTS1";

-- check
-- we don't check planing times here to be independent of
-- plan caching approach
SELECT
  count(*) as total,
  count(*) FILTER (
    WHERE min_exec_time + max_exec_time = 0
  ) as minmax_exec_zero,
  count(*) FILTER (
    WHERE minmax_stats_since >= :'ref_ts'
  ) as minmax_ts_after_ref,
  count(*) FILTER (
    WHERE stats_since >= :'ref_ts'
  ) as stats_since_after_ref
FROM pg_stat_statements
WHERE query LIKE '%STMTTS%';

-- min/max reset keeps mean and standard deviation
SELECT pg_stat_statements_reset() IS NOT NULL AS t;
SELECT pg_sleep(0.001) AS "STMTTS_MINMAX";
SELECT pg_sleep(0.03) AS "STMTTS_MINMAX";
SELECT mean_exec_time AS mean_before, stddev_exec_time AS stddev_before
FROM pg_stat_statements
WHERE query = 'SELECT pg_sleep($1) AS "STMTTS_MINMAX"' \gset
SELECT pg_stat_statements_reset(0, 0, 0, true) IS NOT NULL AS t;
SELECT calls,
  min_exec_time + max_exec_time = 0 AS minmax_exec_zero,
  mean_exec_time = :mean_before AS mean_kept,
  stddev_exec_time = :stddev_before AS stddev_kept,
  stddev_exec_time > 0 AS stddev_nonzero
FROM pg_stat_statements
WHERE query = 'SELECT pg_sleep($1) AS "STMTTS_MINMAX"';
SELECT pg_sleep(0.001) AS "STMTTS_MINMAX";
SELECT calls,
  min_exec_time > 0 AS min_exec_set,
  max_exec_time >= min_exec_time AS max_exec_set,
  stddev_exec_time > 0 AS stddev_nonzero
FROM pg_stat_statements
WHERE query = 'SELECT pg_sleep($1) AS "STMTTS_MINMAX"';

-- Cleanup
SELECT pg_stat_statements_reset() IS NOT NULL AS t;
