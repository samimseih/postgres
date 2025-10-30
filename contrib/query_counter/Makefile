EXTENSION = query_counter
MODULES = query_counter
DATA = query_counter--1.0.sql
PGFILEDESC = "query_counter - count query executions by queryId via pgstat_kind"

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
