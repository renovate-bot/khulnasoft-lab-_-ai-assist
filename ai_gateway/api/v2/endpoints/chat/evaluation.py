import os

from dotenv import load_dotenv
from fastapi import APIRouter
from langchain.chat_models import ChatAnthropic
from langserve import add_routes

from ai_gateway.chat.evaluation import (
    QAChainDataInput,
    QAChainDataOutput,
    qa_context_eval_chain,
    qa_eval_chain,
)

__all__ = ["router"]

from ai_gateway.chat.evaluation.chains.qa import ContextQAChainDataInput

# TODO: Move model instantiation to deps.py or to a similar module
load_dotenv()
model_anthropic_claude: ChatAnthropic = ChatAnthropic(
    model="claude-2", anthropic_api_key=os.environ["ANTHROPIC_API_KEY"]
)

router = APIRouter(
    tags=["chat evaluation"],
)


add_routes(
    router,
    qa_eval_chain(model_anthropic_claude),
    path="/qa",
    input_type=QAChainDataInput,
    output_type=QAChainDataOutput,
)


add_routes(
    router,
    qa_context_eval_chain(model_anthropic_claude),
    path="/qa_context",
    input_type=ContextQAChainDataInput,
    output_type=QAChainDataOutput,
)
