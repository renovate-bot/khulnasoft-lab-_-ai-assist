from pathlib import Path

import pytest
from unittest.mock import Mock

from codesuggestions.models import TextGenBaseModel


@pytest.fixture
def tpl_codegen_dir() -> Path:
    assets_dir = Path(__file__).parent / "codesuggestions" / "_assets"
    tpl_dir = assets_dir / "tpl"
    return tpl_dir / "codegen"


@pytest.fixture
def text_gen_base_model():
    model = Mock(spec=TextGenBaseModel)
    model.MAX_MODEL_LEN = 1_000
    return model
