# flake8: noqa

from ai_gateway import api, cloud_connector, container, experimentation, main, models
from ai_gateway.cloud_connector import CloudConnectorConfig
from ai_gateway.config import *

# Set a default service name
CloudConnectorConfig.set_service_name("gitlab-ai-gateway")
