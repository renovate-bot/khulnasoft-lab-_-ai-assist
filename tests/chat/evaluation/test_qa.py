from unittest.mock import Mock

import pytest
from langchain.chat_models import ChatAnthropic
from langchain.schema import OutputParserException

from ai_gateway.chat.evaluation import (
    QAChainDataInput,
    QAChainDataOutput,
    qa_context_eval_chain,
    qa_eval_chain,
)
from ai_gateway.chat.evaluation.chains.qa import (
    ContextQAChainDataInput,
    QAChainFormatOutput,
)


@pytest.fixture
def model() -> Mock:
    _model = Mock(spec=ChatAnthropic)
    _model.config_specs = []

    return _model


class TestQAEvalChain:
    @pytest.mark.parametrize(
        ("input_variables", "model_output", "expected_output"),
        [
            (
                QAChainDataInput(
                    question="What is the capital of France?",
                    answer_actual="Paris",
                    answer_expected="Paris",
                ),
                QAChainFormatOutput(explanation="this is explanation", grade="CORRECT"),
                QAChainDataOutput(is_correct=True, explanation="this is explanation"),
            ),
            (
                QAChainDataInput(
                    question="What is the capital of France?",
                    answer_actual="Don't know",
                    answer_expected="Paris",
                ),
                QAChainFormatOutput(
                    explanation="this is explanation", grade="INCORRECT"
                ),
                QAChainDataOutput(is_correct=False, explanation="this is explanation"),
            ),
        ],
    )
    def test_ok(
        self,
        model: ChatAnthropic,
        input_variables: QAChainDataInput,
        model_output: QAChainFormatOutput,
        expected_output: QAChainDataOutput,
    ):
        model.invoke = Mock(return_value=f"```{model_output.json()}```")

        chain = qa_eval_chain(model)
        actual = chain.invoke(input_variables.dict())

        assert actual == expected_output

    def test_schema_error(self, model: ChatAnthropic):
        model.invoke = Mock(return_value="random output")

        input_variables = QAChainDataInput(
            question="What is the capital of Paris?",
            answer_actual="Paris",
            answer_expected="Paris",
        )

        chain = qa_eval_chain(model)

        with pytest.raises(OutputParserException):
            _ = chain.invoke(input_variables.dict())


class TestQAContextEvalChain:
    @pytest.mark.parametrize(
        ("input_variables", "model_output", "expected_output"),
        [
            (
                ContextQAChainDataInput(
                    question="What is the capital of France?",
                    answer_actual="Paris",
                    context="Context about the capital of France - Paris",
                ),
                QAChainFormatOutput(explanation="this is explanation", grade="CORRECT"),
                QAChainDataOutput(is_correct=True, explanation="this is explanation"),
            ),
            (
                ContextQAChainDataInput(
                    question="What is the capital of France?",
                    answer_actual="Don't know",
                    context="Context about the capital of France - Paris",
                ),
                QAChainFormatOutput(
                    explanation="this is explanation", grade="INCORRECT"
                ),
                QAChainDataOutput(is_correct=False, explanation="this is explanation"),
            ),
        ],
    )
    def test_ok(
        self,
        model: ChatAnthropic,
        input_variables: QAChainDataInput,
        model_output: QAChainFormatOutput,
        expected_output: QAChainDataOutput,
    ):
        model.invoke = Mock(return_value=f"```{model_output.json()}```")

        chain = qa_context_eval_chain(model)
        actual = chain.invoke(input_variables.dict())

        assert actual == expected_output

    def test_schema_error(self, model: ChatAnthropic):
        model.invoke = Mock(return_value="random output")

        input_variables = ContextQAChainDataInput(
            question="What is the capital of Paris?",
            answer_actual="Paris",
            context="Context about the capital of France - Paris",
        )

        chain = qa_context_eval_chain(model)

        with pytest.raises(OutputParserException):
            _ = chain.invoke(input_variables.dict())
