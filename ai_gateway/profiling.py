import googlecloudprofiler

from ai_gateway.config import ProfilingConfig


def setup_profiling(profiling_config: ProfilingConfig, logger):
    if not profiling_config.enabled:
        return

    # Profiler initialization. It starts a daemon thread which continuously
    # collects and uploads profiles.
    try:
        googlecloudprofiler.start(
            service="model-gateway",
            service_version="1.0.0",  # TODO: fix this
            verbose=profiling_config.verbose,
            period_ms=profiling_config.period_ms,
        )
    except (ValueError, NotImplementedError) as exc:
        logger.error("failed to setup Google Cloud Profiler: %s", exc)
