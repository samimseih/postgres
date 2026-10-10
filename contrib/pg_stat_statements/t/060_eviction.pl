# Copyright (c) 2026, PostgreSQL Global Development Group

# Tests for the eviction of entries once more distinct statements than
# pg_stat_statements.max have been seen.

use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $max = 20;

my $node = PostgreSQL::Test::Cluster->new('main');
$node->init;
$node->append_conf(
	'postgresql.conf', qq{
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = $max
});
$node->start;

$node->safe_psql('postgres', 'CREATE EXTENSION pg_stat_statements');

# A statement that is executed often.  It runs in a transaction block in a
# session that stays connected while entries are evicted around it.
my $hot_calls = 50;
my $session = $node->background_psql('postgres');
$session->query_safe('BEGIN');
$session->query_safe('SELECT 1 + 1 AS evict_hot') for (1 .. $hot_calls);
$session->query_safe('COMMIT');
$session->query_safe('SELECT pg_stat_force_next_flush()');

# Many statements that are executed once each.
my $ncold = 200;
my $sql = '';
for my $i (1 .. $ncold)
{
	$sql .= 'SELECT ' . join(', ', (1) x $i) . " AS evict_cold;\n";
}
$node->safe_psql('postgres', $sql);

cmp_ok(
	$node->safe_psql(
		'postgres', 'SELECT entry_dealloc FROM pg_stat_statements_info'),
	'>', 0, 'entries were evicted');

# Eviction starts when the retained texts exceed the maximum by 10 percent.
# Leave some room for the statements issued by this test itself.
cmp_ok(
	$node->safe_psql(
		'postgres', 'SELECT qtext_count FROM pg_stat_statements_info'),
	'<=',
	$max * 110 / 100 + 3,
	'number of retained query texts is bounded');

cmp_ok(
	$node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query LIKE '%evict_cold'"
	),
	'<=',
	$max * 5 / 4,
	'statements executed once were evicted');

my $hot_query =
  "SELECT coalesce(sum(calls), 0) FROM pg_stat_statements WHERE query = 'SELECT \$1 + \$2 AS evict_hot'";

is($node->safe_psql('postgres', $hot_query),
	$hot_calls, 'frequently executed statement was not evicted');

# The session that ran the frequent statement can still add to it.
$session->query_safe('SELECT 1 + 1 AS evict_hot');
$session->query_safe('SELECT pg_stat_force_next_flush()');
is($node->safe_psql('postgres', $hot_query),
	$hot_calls + 1, 'frequently executed statement is still tracked');
$session->quit;

# Statements that were evicted are tracked again when they come back.
$node->safe_psql('postgres', "SELECT 1 AS evict_cold;\n" x 3);
is( $node->safe_psql(
		'postgres',
		"SELECT calls FROM pg_stat_statements WHERE query = 'SELECT \$1 AS evict_cold'"
	),
	'3',
	'evicted statement is tracked again');

# The counters of retained texts and entries must have followed all of the
# above, so that nothing is left once everything is reset.  Only the
# statements run since the reset may be counted.
$node->safe_psql('postgres', 'SELECT pg_stat_statements_reset()');
cmp_ok(
	$node->safe_psql(
		'postgres',
		'SELECT qtext_count FROM pg_stat_statements_info'),
	'<=', 2, 'no query text is counted after a reset');

# Statistics are written at shutdown and read back, which also verifies that
# no dropped entry was left behind.
$node->safe_psql('postgres', 'SELECT 1 + 1 AS evict_hot');
$node->restart;
is($node->safe_psql('postgres', $hot_query),
	'1', 'statistics survive a restart after evictions');

$node->stop;

my $small_max = 8;

my $small_node = PostgreSQL::Test::Cluster->new('small_max');
$small_node->init;
$small_node->append_conf(
	'postgresql.conf', qq{
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = $small_max
});
$small_node->start;

$small_node->safe_psql('postgres', 'CREATE EXTENSION pg_stat_statements');

# Eviction still starts when the 10 percent buffer rounds down to the maximum.
my $small_sql = '';
for my $i (1 .. 40)
{
	$small_sql .= 'SELECT ' . join(', ', (1) x $i) . " AS evict_small;\n";
}
$small_node->safe_psql('postgres', $small_sql);

cmp_ok(
	$small_node->safe_psql(
		'postgres', 'SELECT entry_dealloc FROM pg_stat_statements_info'),
	'>', 0, 'entries were evicted with a small maximum');

$small_node->safe_psql('postgres', "SELECT 1 AS evict_small_hot;\n" x 3);
is( $small_node->safe_psql(
		'postgres',
		"SELECT calls FROM pg_stat_statements WHERE query = 'SELECT \$1 AS evict_small_hot'"
	),
	'3',
	'new statements are tracked after small-maximum eviction');

$small_node->stop;

done_testing();
