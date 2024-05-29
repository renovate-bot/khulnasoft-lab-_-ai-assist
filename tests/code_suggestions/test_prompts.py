import pytest

from ai_gateway.prompts import PromptTemplate, PromptTemplateFewShot


@pytest.mark.parametrize(
    "lang, code",
    [
        ("python", "random code"),
        ("python", ""),
    ],
)
def test_prompt_template(tpl_codegen_dir, lang, code):
    filepath = tpl_codegen_dir / "base.tpl"
    with open(filepath, "r") as f:
        tpl_raw = f.read()

    tpl = PromptTemplate.from_local_file(filepath)
    prompt = tpl.apply(lang=lang, code=code)

    assert tpl_raw == tpl.raw
    assert lang in prompt
    assert code in prompt


@pytest.mark.parametrize(
    "lang,content,example",
    [
        ("python", "random_prompt", "def hello_world"),
        ("python", "random_prompt", "s = 'hello world'"),
    ],
)
def test_prompt_template_few_shot(tpl_codegen_dir, lang, content, example):
    tpl_filepath = tpl_codegen_dir / "completion.tpl"
    ex_tpl_filepath = tpl_codegen_dir / "base.tpl"

    example_tpl = PromptTemplate.from_local_file(ex_tpl_filepath)
    tpl = PromptTemplateFewShot.from_local_file(
        tpl_filepath,
        [{"lang": lang, "code": example}],
        example_tpl,
    )
    prompt = tpl.apply(lang=lang, prompt=content)

    assert lang in prompt
    assert example in prompt
    assert content in prompt
