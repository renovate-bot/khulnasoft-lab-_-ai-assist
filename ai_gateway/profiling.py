import os

import googlecloudprofiler

from ai_gateway.config import ConfigGoogleCloudProfiler


def setup_profiling(google_cloud_profiler: ConfigGoogleCloudProfiler, logger):
    if not google_cloud_profiler.enabled:
        return

    # Profiler initialization. It starts a daemon thread which continuously
    # collects and uploads profiles.
    try:
        googlecloudprofiler.start(
            service="ai-gateway",
            service_version=os.environ.get(
                "K_REVISION", "1.0.0"
            ),  # https://cloud.google.com/run/docs/container-contract#services-env-vars
            verbose=google_cloud_profiler.verbose,
            period_ms=google_cloud_profiler.period_ms,
        )
    except (ValueError, NotImplementedError) as exc:
        logger.error("failed to setup Google Cloud Profiler: %s", exc)
