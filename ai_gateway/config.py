import os
from typing import Annotated, Optional

from dotenv import find_dotenv
from pydantic import AliasChoices, BaseModel, Field, RootModel
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Config",
    "ConfigLogging",
    "ConfigFastApi",
    "ConfigAuth",
    "ConfigGoogleCloudProfiler",
    "FFlags",
    "FFlagsCodeSuggestions",
    "ConfigSnowplow",
    "ConfigInstrumentator",
    "ConfigVertexTextModel",
    "ConfigModelConcurrency",
    "ConfigCustomModels",
]

ENV_PREFIX = "AIGW"


class ConfigLogging(BaseModel):
    level: str = "INFO"
    format_json: bool = True
    to_file: Optional[str] = None


class ConfigSelfSignedJwt(BaseModel):
    signing_key: str = ""
    validation_key: str = ""


class ConfigFastApi(BaseModel):
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    metrics_host: str = "0.0.0.0"
    metrics_port: int = 8082
    uvicorn_logger: dict = {"version": 1, "disable_existing_loggers": False}
    docs_url: Optional[str] = None
    openapi_url: Optional[str] = None
    redoc_url: Optional[str] = None
    reload: bool = False


class ConfigAuth(BaseModel):
    bypass_external: bool = False
    bypass_external_with_header: bool = False


class ConfigGoogleCloudProfiler(BaseModel):
    enabled: bool = False
    verbose: int = 2
    period_ms: int = 10


class ConfigInstrumentator(BaseModel):
    thread_monitoring_enabled: bool = False
    thread_monitoring_interval: int = 60


class FFlagsCodeSuggestions(BaseModel):
    excl_post_proc: list[str] = []


class FFlags(BaseSettings):
    code_suggestions: Annotated[
        FFlagsCodeSuggestions, Field(default_factory=FFlagsCodeSuggestions)
    ] = FFlagsCodeSuggestions()


class ConfigSnowplow(BaseModel):
    enabled: bool = False
    endpoint: Optional[str] = None
    batch_size: Optional[int] = 10
    thread_count: Optional[int] = 1


class ConfigCustomModels(BaseModel):
    enabled: bool = False


def _build_location(default: str = "us-central1") -> str:
    """
    Reads the GCP region from the environment.
    Returns the default argument when not configured.
    """
    # pylint: disable=direct-environment-variable-reference
    return os.getenv("RUNWAY_REGION", default)
    # pylint: enable=direct-environment-variable-reference


def _build_endpoint() -> str:
    """
    Returns the default endpoint for Vertex AI.

    This code assumes that the Runway region (i.e. Cloud Run region) is the same as the Vertex AI region.
    To support other Cloud Run regions, this code will need to be updated to map to a nearby Vertex AI region instead.
    """
    return f"{_build_location()}-aiplatform.googleapis.com"


class ConfigGoogleCloudPlatform(BaseModel):
    project: str = ""
    service_account_json_key: str = ""


class ConfigVertexTextModel(ConfigGoogleCloudPlatform):
    location: str = Field(default_factory=_build_location)
    endpoint: str = Field(default_factory=_build_endpoint)


class ConfigVertexSearch(ConfigGoogleCloudPlatform):
    fallback_datastore_version: str = ""


class ConfigModelConcurrency(RootModel):
    root: dict[str, dict[str, int]] = {}

    def for_model(self, engine: str, name: str) -> Optional[int]:
        return self.root.get(engine, {}).get(name, None)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix=f"{ENV_PREFIX}_",
        protected_namespaces=(),
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gitlab_url: str = "https://gitlab.com"
    gitlab_api_url: str = "https://gitlab.com/api/v4/"
    customer_portal_url: str = "https://customers.gitlab.com"

    mock_model_responses: bool = Field(
        validation_alias=AliasChoices(
            f"{ENV_PREFIX.lower()}_mock_model_responses",
            f"{ENV_PREFIX.lower()}_use_fake_models",  # Backward compatibility with the GitLab QA tests
        ),
        default=False,
    )

    logging: Annotated[ConfigLogging, Field(default_factory=ConfigLogging)] = (
        ConfigLogging()
    )
    self_signed_jwt: Annotated[
        ConfigSelfSignedJwt, Field(default_factory=ConfigSelfSignedJwt)
    ] = ConfigSelfSignedJwt()
    fastapi: Annotated[ConfigFastApi, Field(default_factory=ConfigFastApi)] = (
        ConfigFastApi()
    )
    auth: Annotated[ConfigAuth, Field(default_factory=ConfigAuth)] = ConfigAuth()
    google_cloud_profiler: Annotated[
        ConfigGoogleCloudProfiler, Field(default_factory=ConfigGoogleCloudProfiler)
    ] = ConfigGoogleCloudProfiler()
    instrumentator: Annotated[
        ConfigInstrumentator, Field(default_factory=ConfigInstrumentator)
    ] = ConfigInstrumentator()
    f: Annotated[FFlags, Field(default_factory=FFlags)] = FFlags()
    snowplow: Annotated[ConfigSnowplow, Field(default_factory=ConfigSnowplow)] = (
        ConfigSnowplow()
    )
    google_cloud_platform: Annotated[
        ConfigGoogleCloudPlatform, Field(default_factory=ConfigGoogleCloudPlatform)
    ] = ConfigGoogleCloudPlatform()
    custom_models: Annotated[
        ConfigCustomModels, Field(default_factory=ConfigCustomModels)
    ] = ConfigCustomModels()
    vertex_text_model: Annotated[
        ConfigVertexTextModel, Field(default_factory=ConfigVertexTextModel)
    ] = ConfigVertexTextModel()
    vertex_search: Annotated[
        ConfigVertexSearch, Field(default_factory=ConfigVertexSearch)
    ] = ConfigVertexSearch()
    model_engine_concurrency_limits: Annotated[
        ConfigModelConcurrency, Field(default_factory=ConfigModelConcurrency)
    ] = ConfigModelConcurrency()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._apply_global_configs(
            parent=self.google_cloud_platform,
            children=[self.vertex_text_model, self.vertex_search],
        )

    def _apply_global_configs(self, parent: BaseModel, children: list[BaseModel]):
        """Set a parent config to child configs if the field value is not specified"""
        for field in parent.model_fields_set:
            parent_value = getattr(parent, field)

            if not parent_value:
                continue

            for child in children:
                if field in child.model_fields_set:
                    continue

                setattr(child, field, parent_value)
