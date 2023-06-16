MONITORING_NAMESPACE ?= monitoring

APPDIR := $(shell pwd)
COMPOSE := docker-compose -f docker-compose.dev.yaml

.PHONY: develop-local
develop-local:
	$(COMPOSE) up --build --remove-orphans

.PHONY: test-local
test-local:
	$(COMPOSE) run -v "$(APPDIR):/app" api bash -c 'poetry install --with test && poetry run pytest'

.PHONY: lint
lint:
	$(COMPOSE) run -v "$(APPDIR):/app" api bash -c 'poetry install --with lint && poetry run flake8 codesuggestions'

clean:
	$(COMPOSE) rm -s -v -f

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
