# Copyright (c) 2026, PostgreSQL Global Development Group

# Tests that pg_stat_statements entries are not affected by the per-database
# processing of the cumulative statistics system: DROP DATABASE and
# pg_stat_reset().

use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('main');
$node->init;
$node->append_conf('postgresql.conf',
	"shared_preload_libraries = 'pg_stat_statements'");
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION pg_stat_statements');
$node->safe_psql('postgres', 'CREATE DATABASE regress_pgss_dropdb');

my $dboid = $node->safe_psql('postgres',
	"SELECT oid FROM pg_database WHERE datname = 'regress_pgss_dropdb'");

$node->safe_psql(
	'regress_pgss_dropdb', q{
SELECT 1 AS dropdb_a;
SELECT 1 AS dropdb_a, 2 AS dropdb_b;
SELECT 1 AS dropdb_a, 2 AS dropdb_b, 3 AS dropdb_c;
});

my $entries_query =
  "SELECT count(*), count(query) FROM pg_stat_statements WHERE dbid = $dboid AND query LIKE '%dropdb_a%'";

is($node->safe_psql('postgres', $entries_query),
	'3|3', 'statements of the database are tracked');

# The texts retained before dropping the database.  The statements used
# around the measurement are themselves tracked, so run them once beforehand.
my $qtext_query = 'SELECT qtext_count FROM pg_stat_statements_info';
$node->safe_psql('postgres', "SET pg_stat_statements.track = 'none'");
$node->safe_psql('postgres', $qtext_query);
my $qtext_before = $node->safe_psql('postgres', $qtext_query);

# Don't track the DROP DATABASE itself, to keep the counter comparable.
$node->safe_psql(
	'postgres', q{
SET pg_stat_statements.track = 'none';
DROP DATABASE regress_pgss_dropdb;
});

is($node->safe_psql('postgres', $entries_query),
	'3|3', 'statements of a dropped database are kept, with their text');
is($node->safe_psql('postgres', $qtext_query),
	$qtext_before, 'query text accounting is unchanged by DROP DATABASE');

# pg_stat_reset() resets the statistics counters of the current database,
# not pg_stat_statements.
$node->safe_psql(
	'postgres', q{
SELECT 1 AS stat_reset_a, 2 AS stat_reset_b;
SELECT 1 AS stat_reset_a, 2 AS stat_reset_b;
});
$node->safe_psql('postgres', 'SELECT pg_stat_reset()');

is( $node->safe_psql(
		'postgres',
		"SELECT calls FROM pg_stat_statements WHERE query LIKE '%stat_reset_a%' AND query NOT LIKE '%pg_stat_statements%'"
	),
	'2',
	'pg_stat_reset() does not reset pg_stat_statements');

# The entries, including those of the dropped database, survive a restart.
$node->restart;

is($node->safe_psql('postgres', $entries_query),
	'3|3', 'statements of a dropped database are kept across restart');

$node->stop;

done_testing();
