# Copyright (c) 2026, PostgreSQL Global Development Group

# Test that the query texts restored from a statistics file are released
# when the file turns out to be corrupted and its contents are discarded.

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

# Track a number of distinct statements.
my $nstatements = 50;
my $sql = '';
for my $i (1 .. $nstatements)
{
	$sql .= 'SELECT ' . join(', ', (1) x $i) . " AS broken_file;\n";
}
$node->safe_psql('postgres', $sql);

my $qtext_query = 'SELECT qtext_count FROM pg_stat_statements_info';
cmp_ok($node->safe_psql('postgres', $qtext_query),
	'>=', $nstatements, 'query texts are retained');

$node->stop;

# Chop off the end of the statistics file.  pg_stat_statements entries are
# written last, so reading fails after most of them have been restored.
my $statsfile = $node->data_dir . '/pg_stat/pgstat.stat';
ok(-f $statsfile, 'statistics file exists after shutdown');
truncate($statsfile, (-s $statsfile) - 16)
  or die "could not truncate $statsfile: $!";

my $log_offset = -s $node->logfile;
$node->start;

ok( $node->log_contains('corrupted statistics file', $log_offset),
	'statistics file was found corrupted');

is( $node->safe_psql(
		'postgres',
		"SELECT count(*) FROM pg_stat_statements WHERE query LIKE '%broken_file%'"
	),
	'0',
	'no statement was restored');

# Only the statements executed since the restart may hold query text.
cmp_ok($node->safe_psql('postgres', $qtext_query),
	'<', 10, 'restored query texts were released');

$node->stop;

done_testing();
