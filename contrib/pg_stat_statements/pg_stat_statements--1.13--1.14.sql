/* contrib/pg_stat_statements/pg_stat_statements--1.13--1.14.sql */

-- complain if script is sourced in psql, rather than via ALTER EXTENSION
\echo Use "ALTER EXTENSION pg_stat_statements UPDATE TO '1.14'" to load this file. \quit

ALTER FUNCTION pg_stat_statements(boolean) PARALLEL RESTRICTED;
ALTER FUNCTION pg_stat_statements_reset(Oid, Oid, bigint, boolean) PARALLEL RESTRICTED;
