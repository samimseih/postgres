# Copyright (c) 2026, PostgreSQL Global Development Group

# Tests for the pending statistics of pg_stat_statements, accumulated by a
# backend that finds the lock of an entry busy.  An injection point keeps
# the entry lock held by another backend.

use strict;
use warnings FATAL => 'all';
use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

if ($ENV{enable_injection_points} ne 'yes')
{
	plan skip_all => 'Injection points not supported by this build';
}

my $node = PostgreSQL::Test::Cluster->new('main');
$node->init;
$node->append_conf('postgresql.conf',
	"shared_preload_libraries = 'pg_stat_statements'");
$node->start;

if (!$node->check_extension('injection_points'))
{
	plan skip_all => 'Extension injection_points not installed';
}

$node->safe_psql('postgres', 'CREATE EXTENSION pg_stat_statements');
$node->safe_psql('postgres', 'CREATE EXTENSION injection_points');

my $point = 'pg-stat-statements-store-locked';
my $calls_query =
  q{SELECT calls FROM pg_stat_statements WHERE query = 'SELECT pg_sleep($1)'};

# Start a session that waits at the injection point while holding the
# entry lock of "SELECT pg_sleep($1)".  Statements setting it up are not
# tracked, so that only this one reaches the point.
sub block_entry
{
	my $session = $node->background_psql('postgres');

	$session->query_safe('SET pg_stat_statements.track = none');
	$session->query_safe('SELECT injection_points_set_local()');
	$session->query_safe(
		"SELECT injection_points_attach('$point', 'wait')");
	$session->query_safe('SET pg_stat_statements.track = top');
	$session->query_until(qr/blocking/,
		"\\echo blocking\nSELECT pg_sleep(0.01);\n");
	$node->poll_query_until('postgres',
		"SELECT count(*) > 0 FROM pg_stat_activity WHERE wait_event = '$point'"
	) or die "timed out waiting for the injection point";

	return $session;
}

sub release_entry
{
	my ($session) = @_;

	$node->safe_psql('postgres', "SELECT injection_points_wakeup('$point')");
	$session->quit;
}

# Executions finding the entry lock busy are counted once flushed, and
# pending min/max values gathered before a min/max reset are discarded.  The
# transaction block keeps the pending statistics until COMMIT, after the
# reset.
$node->safe_psql('postgres', 'SELECT pg_stat_statements_reset()');
my $blocker = block_entry();
my $session = $node->background_psql('postgres');

$session->query_safe('BEGIN');
$session->query_safe('SELECT pg_sleep(0.2)');
$session->query_safe('SELECT pg_sleep(0.2)');
release_entry($blocker);

$node->safe_psql('postgres',
	'SELECT pg_stat_statements_reset(0, 0, 0, true)');
$session->query_safe('SELECT pg_sleep(0.01)');
$session->query_safe('COMMIT');

is( $session->query_safe(
		q{SELECT calls, max_exec_time < 100 FROM pg_stat_statements WHERE query = 'SELECT pg_sleep($1)'}
	),
	'4|t',
	'pending statistics counted, and their min/max discarded after a reset');

# A backend keeps counting a statement after its entry was removed while
# the backend had pending statistics for it.
$node->safe_psql('postgres', 'SELECT pg_stat_statements_reset()');
$blocker = block_entry();

$session->query_safe('BEGIN');
$session->query_safe('SELECT pg_sleep(0.01)');
release_entry($blocker);

$node->safe_psql('postgres', 'SELECT pg_stat_statements_reset()');
$session->query_safe('SELECT 1');
$session->query_safe('COMMIT');

$session->query_safe('SELECT pg_sleep(0.01)');
$session->query_safe('SELECT pg_sleep(0.01)');

is($session->query_safe($calls_query),
	'2', 'statement counted again after its entry was removed');

$session->quit;
$node->stop;

done_testing();
