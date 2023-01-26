MONITORING_NAMESPACE ?= monitoring

develop-local:
	docker-compose -f docker-compose.dev.yaml up --build --remove-orphans

clean:
	docker-compose -f docker-compose.dev.yaml rm -s -v -f

.PHONY: all test test

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
