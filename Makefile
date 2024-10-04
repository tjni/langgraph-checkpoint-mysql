.PHONY: test test_watch lint format

######################
# TESTING AND COVERAGE
######################

start-mysql:
	MYSQL_VERSION=${MYSQL_VERSION:-16} docker compose -f tests/compose-mysql.yml up -V --force-recreate --wait

stop-mysql:
	docker compose -f tests/compose-mysql.yml down

MYSQL_VERSIONS ?= 8
test_mysql_version:
	@echo "Testing MySQL $(MYSQL_VERSION)"
	@MYSQL_VERSION=$(MYSQL_VERSION) make start-mysql
	@poetry run pytest $(TEST)
	@EXIT_CODE=$$?; \
	make stop-mysql; \
	echo "Finished testing MySQL $(MYSQL_VERSION); Exit code: $$EXIT_CODE"; \
	exit $$EXIT_CODE

test:
	@for version in $(MYSQL_VERSIONS); do \
		if ! make test_mysql_version MYSQL_VERSION=$$version; then \
			echo "Test failed for MySQL $$version"; \
			exit 1; \
		fi; \
	done
	@echo "All MySQL versions tested successfully"

TEST ?= .
test_watch:
	POSTGRES_VERSION=${MYSQL_VERSION:-8} make start-mysql; \
	poetry run ptw $(TEST); \
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
	poetry run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)
