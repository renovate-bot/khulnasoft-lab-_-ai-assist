from typing import Optional

from langchain.chat_models import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable, RunnableLambda
from pydantic import BaseModel, Field

__all__ = [
    "PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_CONTEXT",
    "QAChainDataInput",
    "ContextQAChainDataInput",
    "QAChainDataOutput",
    "qa_eval_chain",
    "qa_context_eval_chain",
]


PROMPT_TEMPLATE = """
You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer,
and are asked to score the student answer as either CORRECT or INCORRECT.

Input format:
```
{{
  "question": "question here",
  "student_answer": "student's answer here",
  "true_answer": "true answer here"
}}
```

{output_format}

Grade the student answers based ONLY on their factual accuracy.
Ignore differences in punctuation and phrasing between the student answer and true answer.
It is OK if the student answer contains more information than the true answer,
as long as it does not contain any conflicting statements.

Begin!

```
{{
  "question": "{question}",
  "student_answer": "{answer_actual}",
  "true_answer": "{answer_expected}"
}}
```
""".strip(
    "\n"
)

PROMPT_TEMPLATE_CONTEXT = """
You are a teacher grading a quiz.
You are given a question, the context the question is about, and the student's answer.
You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Input Format:
```
{{
  "question": "question here",
  "context": "context the question is about here",
  "student_answer": "student's answer here"
}}
```

{output_format}

Grade the student answers based ONLY on their factual accuracy.
Ignore differences in punctuation and phrasing between the student answer and true answer.
It is OK if the student answer contains more information than the true answer,
as long as it does not contain any conflicting statements.

Begin!

```
{{
  "question": "{question}",
  "context": "{context}",
  "student_answer": "{answer_actual}"
}}
```
""".strip(
    "\n"
)


class QAChainFormatOutput(BaseModel):
    explanation: Optional[str] = Field(
        description="Step-by-step explanation on why a particular grade has been awarded"
    )
    grade: str = Field(description="grade, i.e., CORRECT or INCORRECT")


class QAChainDataInput(BaseModel):
    question: str
    answer_actual: str
    answer_expected: str


class QAChainDataOutput(BaseModel):
    is_correct: bool
    explanation: Optional[str]

    @classmethod
    def from_parser_output(cls, data: QAChainFormatOutput):
        return cls(
            is_correct=True if data.grade == "CORRECT" else False,
            explanation=data.explanation,
        )


def qa_eval_chain(model: ChatAnthropic) -> Runnable[dict, QAChainDataOutput]:
    parser = PydanticOutputParser(pydantic_object=QAChainFormatOutput)
    prompt = PromptTemplate(
        input_variables=list(QAChainDataInput.__fields__.keys()),
        template=PROMPT_TEMPLATE,
        partial_variables={"output_format": parser.get_format_instructions()},
    )

    model.temperature = 0

    chain = (
        prompt | model | parser | RunnableLambda(QAChainDataOutput.from_parser_output)
    )

    return chain


class ContextQAChainDataInput(BaseModel):
    question: str
    context: str
    answer_actual: str


def qa_context_eval_chain(model: ChatAnthropic) -> Runnable[dict, QAChainDataOutput]:
    parser = PydanticOutputParser(pydantic_object=QAChainFormatOutput)
    prompt = PromptTemplate(
        input_variables=list(ContextQAChainDataInput.__fields__.keys()),
        template=PROMPT_TEMPLATE_CONTEXT,
        partial_variables={"output_format": parser.get_format_instructions()},
    )

    model.temperature = 0

    chain = (
        prompt | model | parser | RunnableLambda(QAChainDataOutput.from_parser_output)
    )

    return chain
