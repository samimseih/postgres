-- Use CREATE EXTENSION to load this file
\echo Use "CREATE EXTENSION query_counter" to load this file. \quit

CREATE FUNCTION query_counter(
    OUT userid oid,
    OUT dbid oid,
    OUT toplevel bool,
    OUT queryid bigint,
    OUT query text,
    OUT calls_started bigint,
    OUT calls_completed bigint,
    OUT generic_plan_calls bigint,
    OUT custom_plan_calls bigint
)
RETURNS SETOF record
AS 'MODULE_PATHNAME'
LANGUAGE C;

CREATE VIEW query_counter AS
  SELECT * FROM query_counter();

GRANT SELECT ON query_counter TO PUBLIC;
