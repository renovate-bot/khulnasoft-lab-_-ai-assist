from pathlib import Path
from unittest.mock import Mock

import pytest

from ai_gateway.models import TextGenBaseModel

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def tpl_codegen_dir() -> Path:
    assets_dir = Path(__file__).parent / "ai_gateway" / "_assets"
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture
def text_gen_base_model():
    model = Mock(spec=TextGenBaseModel)
    model.MAX_MODEL_LEN = 1_000
    model.UPPER_BOUND_MODEL_CHARS = model.MAX_MODEL_LEN * 5
    return model
