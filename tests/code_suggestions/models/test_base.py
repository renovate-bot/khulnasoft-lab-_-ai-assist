from unittest import mock

from ai_gateway.models import ModelMetadata
from ai_gateway.models.base_text import TextGenModelBase


class TestTextGenBaseModel:
    class TestClass(TextGenModelBase):
        @property
        def metadata(self):
            return ModelMetadata(engine="vertex", name="code-gecko@002")

        async def generate(self, **kwargs):
            pass

    @mock.patch("ai_gateway.models.base.config.model_engine_concurrency_limits")
    def test_instrumentator(self, mock_config):
        mock_config.for_model.return_value = 7

        model = TestTextGenBaseModel.TestClass()
        instrumentator = model.instrumentator

        mock_config.for_model.assert_called_with(engine="vertex", name="code-gecko@002")
        assert instrumentator.concurrency_limit == 7
