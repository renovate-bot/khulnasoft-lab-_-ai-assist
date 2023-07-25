MONITORING_NAMESPACE ?= monitoring

ROOT_DIR := $(shell pwd)
TESTS_DIR := ${ROOT_DIR}/tests
CODE_SUGGESTIONS_DIR := ${ROOT_DIR}/codesuggestions

LINT_WORKING_DIR ?= ${CODE_SUGGESTIONS_DIR}/suggestions

COMPOSE_FILES := -f docker-compose.dev.yaml
ifneq (,$(wildcard docker-compose.override.yaml))
COMPOSE_FILES += -f docker-compose.override.yaml
endif
COMPOSE := docker-compose $(COMPOSE_FILES)

.PHONY: develop-local
develop-local:
	$(COMPOSE) up --build --remove-orphans

.PHONY: test-local
test-local:
	$(COMPOSE) run -v "$(ROOT_DIR):/app" api bash -c 'poetry install --with test && poetry run pytest'

.PHONY: lint-local
lint-local:
	$(COMPOSE) run -v "$(ROOT_DIR):/app" api bash -c 'poetry install --with lint && poetry run flake8 codesuggestions'

clean:
	$(COMPOSE) rm -s -v -f

.PHONY: install-lint-deps
install-lint-deps:
	@echo "Installing lint dependencies..."
	@poetry install --with lint

.PHONY: black
black: install-lint-deps
	@echo "Running black format..."
	@poetry run black ${LINT_WORKING_DIR}

.PHONY: isort
isort: install-lint-deps
	@echo "Running isort format..."
	@poetry run isort ${LINT_WORKING_DIR}

.PHONY: format
format: black isort

.PHONY: lint
lint: flake8 check-black check-isort

.PHONY: flake8
flake8: install-lint-deps
	@echo "Running flake8..."
	@poetry run flake8 ${CODE_SUGGESTIONS_DIR}

.PHONY: check-black
check-black: install-lint-deps
	@echo "Running black check..."
	@poetry run black --check ${LINT_WORKING_DIR}

.PHONY: check-isort
check-isort: install-lint-deps
	@echo "Running isort check..."
	@poetry run isort --check-only ${LINT_WORKING_DIR}

.PHONY: install-test-deps
install-test-deps:
	@echo "Installing test dependencies..."
	@poetry install --with test
	@echo 'Building tree-sitter library...'
	@poetry run python scripts/build-tree-sitter-lib.py
	@mkdir -p lib
	@mv scripts/lib/*.so lib

.PHONY: test
test: LIB_DIR ?= ${ROOT_DIR}/lib
test: install-test-deps
	@echo "Running test..."
	@poetry run pytest

.PHONY: nuke
nuke: monitoring-teardown monitoring-nuke
	@echo "Delete namespace ${NAMESPACE}"
	@kubectl delete namespace ${NAMESPACE}

.PHONY: monitoring-setup
monitoring-setup:
	@helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	@kubectl create namespace ${MONITORING_NAMESPACE}

.PHONY: monitoring-nuke
monitoring-nuke:
	@helm repo remove prometheus-community
	@kubectl delete namespace ${MONITORING_NAMESPACE}

.PHONY: monitoring-deploy
monitoring-deploy:
	@helm install -n ${MONITORING_NAMESPACE} prometheus prometheus-community/kube-prometheus-stack

.PHONY: monitoring-teardown
monitoring-teardown:
	@helm uninstall -n ${MONITORING_NAMESPACE} prometheus

.PHONY: all test test clean
