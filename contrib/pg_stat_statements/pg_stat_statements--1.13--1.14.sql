/* contrib/pg_stat_statements/pg_stat_statements--1.13--1.14.sql */

-- complain if script is sourced in psql, rather than via ALTER EXTENSION
\echo Use "ALTER EXTENSION pg_stat_statements UPDATE TO '1.14'" to load this file. \quit

/* Update pg_stat_statements_info with entry and query-text storage counters */
ALTER EXTENSION pg_stat_statements DROP VIEW pg_stat_statements_info;
ALTER EXTENSION pg_stat_statements DROP FUNCTION pg_stat_statements_info();

DROP VIEW pg_stat_statements_info;
DROP FUNCTION pg_stat_statements_info();

CREATE FUNCTION pg_stat_statements_info(
    OUT entry_dealloc bigint,
    OUT entry_count bigint,
    OUT entry_dsa_size bigint,
    OUT qtext_dsa_size bigint,
    OUT qtext_count bigint,
    OUT qtext_total_len bigint,
    OUT stats_reset timestamp with time zone
)
RETURNS record
AS 'MODULE_PATHNAME'
LANGUAGE C STRICT VOLATILE PARALLEL SAFE;

CREATE VIEW pg_stat_statements_info AS
  SELECT * FROM pg_stat_statements_info();

GRANT SELECT ON pg_stat_statements_info TO PUBLIC;

/*
 * These functions can flush this backend's pending stats, which cannot be
 * done from a parallel worker.
 */
ALTER FUNCTION pg_stat_statements(boolean) PARALLEL RESTRICTED;
ALTER FUNCTION pg_stat_statements_reset(Oid, Oid, bigint, boolean) PARALLEL RESTRICTED;
