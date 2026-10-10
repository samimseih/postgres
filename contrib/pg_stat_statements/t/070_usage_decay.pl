# Copyright (c) 2026, PostgreSQL Global Development Group

# Test that usage decays as entries are evicted: a statement that was executed
# many times but is no longer used eventually makes room for new statements,
# while a statement that keeps being executed is kept.

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

# The stale statement starts with enough usage to survive many eviction passes.
my $stale_calls = 30;
$node->safe_psql('postgres', "SELECT 1 + 1 AS decay_stale;\n" x $stale_calls);

# The active statement starts lower, but keeps getting executed as new
# statements are admitted.
my $active_calls = 10;
my $sql = "SELECT 1 - 1 AS decay_active;\n" x $active_calls;

# With max = 20 and a 10 percent buffer, eviction runs roughly once for every
# two new statements.
my $ncold = 700;
for my $i (1 .. $ncold)
{
	$sql .= 'SELECT ' . join(', ', (1) x $i) . " AS decay_cold;\n";
	if ($i % 10 == 0)
	{
		$sql .= "SELECT 1 - 1 AS decay_active;\n";
		$active_calls++;
	}
}
$node->safe_psql('postgres', $sql);

cmp_ok(
	$node->safe_psql(
		'postgres', 'SELECT entry_dealloc FROM pg_stat_statements_info'),
	'>=', 300, 'entries were evicted many times');

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query = 'SELECT \$1 + \$2 AS decay_stale'"
	),
	'0',
	'statement that is no longer executed was eventually evicted');

is( $node->safe_psql(
		'postgres',
		"SELECT calls FROM pg_stat_statements WHERE query = 'SELECT \$1 - \$2 AS decay_active'"
	),
	$active_calls,
	'statement that keeps being executed was not evicted');

$node->stop;

done_testing();
