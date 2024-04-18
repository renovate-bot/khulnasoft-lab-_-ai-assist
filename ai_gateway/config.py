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
]

ENV_PREFIX = "AIGW"


class ConfigLogging(BaseModel):
    level: str = "INFO"
    format_json: bool = True
    to_file: Optional[str] = None


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


class ConfigVertexTextModel(BaseModel):
    project: str = "unreview-poc-390200e5"
    location: str = "us-central1"
    endpoint: str = "us-central1-aiplatform.googleapis.com"
    json_key: str = ""


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
    vertex_text_model: Annotated[
        ConfigVertexTextModel, Field(default_factory=ConfigVertexTextModel)
    ] = ConfigVertexTextModel()
    model_engine_concurrency_limits: Annotated[
        ConfigModelConcurrency, Field(default_factory=ConfigModelConcurrency)
    ] = ConfigModelConcurrency()
