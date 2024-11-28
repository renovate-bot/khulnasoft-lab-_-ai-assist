from dependency_injector import providers

from ai_gateway.experimentation.base import ExperimentRegistry
from ai_gateway.experimentation.experiments import exp_truncate_suffix


def _experiments_provider() -> providers.List:
    return providers.List(
        exp_truncate_suffix.make_experiment(),
    )


def experiment_registry_provider() -> providers.Singleton:
    return providers.Singleton(
        ExperimentRegistry,
        experiments=_experiments_provider(),
    )
