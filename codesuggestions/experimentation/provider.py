from dependency_injector import providers

from codesuggestions.experimentation.base import Experiment, ExperimentRegistry
from codesuggestions.experimentation.experiments import exp_truncate_suffix


def _experiments_provider() -> list[Experiment]:
    return providers.List(
        exp_truncate_suffix.make_experiment(),
    )


def experiment_registry_provider() -> providers.Singleton:
    return providers.Singleton(
        ExperimentRegistry,
        experiments=_experiments_provider(),
    )
