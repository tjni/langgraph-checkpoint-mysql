.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-mysql:
	MYSQL_VERSION=$(MYSQL_VERSION) docker compose -f tests/compose-mysql.yml up -V --force-recreate --wait || ( \
    echo "Failed to start MySQL, printing logs..."; \
		docker compose -f tests/compose-mysql.yml logs; \
		exit 1 \
  )

stop-mysql:
	docker compose -f tests/compose-mysql.yml down --remove-orphans -v

MYSQL_VERSIONS ?= 8
test_mysql_version:
	@echo "Testing MySQL $(MYSQL_VERSION)"
	@MYSQL_VERSION=$(MYSQL_VERSION) make start-mysql
	@uv run pytest --ignore=langgraph-tests $(TEST) && \
	uv run pytest -n auto --dist worksteal langgraph-tests || ( \
	  EXIT_CODE=$$?; \
	  make stop-mysql; \
	  echo "Finished testing MySQL $(MYSQL_VERSION); Exit code: $$EXIT_CODE"; \
	  exit $$EXIT_CODE \
	)

test:
	@for version in $(MYSQL_VERSIONS); do \
		if ! make test_mysql_version MYSQL_VERSION=$$version; then \
			echo "Test failed for MySQL $$version"; \
			exit 1; \
		fi; \
	done
	@echo "All MySQL versions tested successfully"

start-mariadb:
	MARIADB_VERSION=$(MARIADB_VERSION) docker compose -f tests/compose-mariadb.yml up -V --force-recreate --wait || ( \
    echo "Failed to start MariaDB, printing logs..."; \
		docker compose -f tests/compose-mariadb.yml logs; \
		exit 1 \
  )

stop-mariadb:
	docker compose -f tests/compose-mariadb.yml down --remove-orphans -v

MARIADB_VERSIONS ?= 10
test_mariadb_version:
	@echo "Testing MariaDB $(MARIADB_VERSION)"
	@MARIADB_VERSION=$(MARIADB_VERSION) make start-mariadb
	@uv run pytest --ignore=langgraph-tests $(TEST) && \
	uv run pytest -n auto --dist worksteal langgraph-tests || ( \
	  EXIT_CODE=$$?; \
	  make stop-mariadb; \
	  echo "Finished testing MariaDB $(MARIADB_VERSION); Exit code: $$EXIT_CODE"; \
	  exit $$EXIT_CODE \
	)

test-mariadb:
	@for version in $(MARIADB_VERSIONS); do \
		if ! make test_mariadb_version MARIADB_VERSION=$$version; then \
			echo "Test failed for MariaDB $$version"; \
			exit 1; \
		fi; \
	done
	@echo "All MariaDB versions tested successfully"

TEST ?= .
test_watch:
	MYSQL_VERSION=$(MYSQL_VERSION) make start-mysql; \
	uv run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-mysql; \
	exit $$EXIT_CODE

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)
