# Copyright (c) 2023-2026, PostgreSQL Global Development Group

# Tests for checking that pg_stat_statements contents are preserved
# across restarts.

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

$node->safe_psql('postgres', 'CREATE TABLE t1 (a int)');
$node->safe_psql('postgres', 'SELECT a FROM t1');

is( $node->safe_psql(
		'postgres',
		"SELECT query FROM pg_stat_statements WHERE query NOT LIKE '%pg_stat_statements%' ORDER BY query"
	),
	"CREATE TABLE t1 (a int)\nSELECT a FROM t1",
	'pg_stat_statements populated');

$node->restart;

is( $node->safe_psql(
		'postgres',
		"SELECT query FROM pg_stat_statements WHERE query NOT LIKE '%pg_stat_statements%' ORDER BY query"
	),
	"CREATE TABLE t1 (a int)\nSELECT a FROM t1",
	'pg_stat_statements data kept across restart');

is( $node->safe_psql(
		'postgres',
		"SELECT calls FROM pg_stat_statements WHERE query = 'SELECT a FROM t1'"
	),
	'1',
	'pg_stat_statements counters kept across restart');

# Statements run in another session become visible once it flushes its
# pending stats.
my $bg = $node->background_psql('postgres');
$bg->query_safe('SELECT 1 AS cross_session');
$bg->query_safe('SELECT pg_stat_force_next_flush()');
ok( $node->poll_query_until(
		'postgres',
		"SELECT calls = 1 FROM pg_stat_statements WHERE query = 'SELECT \$1 AS cross_session'"
	),
	'pg_stat_statements shows statements from another session');
$bg->quit;

# Entries saved with their texts are not restored if .save is off at
# startup.
$node->stop;
$node->append_conf('postgresql.conf', "pg_stat_statements.save = false");
$node->start;

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query NOT LIKE '%pg_stat_statements%'"
	),
	'0',
	'pg_stat_statements data not restored with .save=false at startup');

# Entries are saved without their texts if .save is off at shutdown, and
# such entries are not restored.
$node->safe_psql('postgres', 'SELECT a FROM t1');
$node->stop;
$node->append_conf('postgresql.conf', "pg_stat_statements.save = true");
$node->start;

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query NOT LIKE '%pg_stat_statements%'"
	),
	'0',
	'pg_stat_statements data not kept across restart with .save=false at shutdown');

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query IS NULL"
	),
	'0',
	'pg_stat_statements textless entries not restored');

my $stats_reset = $node->safe_psql(
	'postgres',
	"SET pg_stat_statements.track = none;
	 SELECT pg_stat_statements_reset()::text");

$node->restart;

is( $node->safe_psql(
		'postgres',
		"SELECT stats_reset::text FROM pg_stat_statements_info"
	),
	$stats_reset,
	'pg_stat_statements_info stats_reset kept across restart');

# A corrupted stats file discards all entries, and the global stats.
$node->safe_psql('postgres', 'SELECT a FROM t1');
$node->stop;
my $statsfile = $node->data_dir . '/pg_stat/pgstat.stat';
truncate($statsfile, int((-s $statsfile) / 2))
  or die "could not truncate \"$statsfile\": $!";
$node->start;

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query NOT LIKE '%pg_stat_statements%'"
	),
	'0',
	'pg_stat_statements data discarded with a corrupted stats file');

isnt(
	$node->safe_psql(
		'postgres', "SELECT stats_reset::text FROM pg_stat_statements_info"),
	$stats_reset,
	'pg_stat_statements_info reset with a corrupted stats file');

$node->stop;

done_testing();
