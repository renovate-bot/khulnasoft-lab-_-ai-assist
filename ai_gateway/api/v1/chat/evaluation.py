import os

import structlog
from dotenv import load_dotenv
from fastapi import APIRouter
from langchain.chat_models import ChatAnthropic
from langserve import add_routes

from ai_gateway.chat.evaluation import (
    ContextQAChainDataInput,
    QAChainDataInput,
    QAChainDataOutput,
    qa_context_eval_chain,
    qa_eval_chain,
)

__all__ = ["create_router"]

# TODO: Move model instantiation to deps.py or to a similar module
load_dotenv()

log = structlog.stdlib.get_logger("chat_evaluation")


def create_router() -> APIRouter:
    router = APIRouter(tags=["chat evaluation"])
    if not os.environ.get("ANTHROPIC_API_KEY", None):
        log.warn(
            "Env ANTHROPIC_API_KEY is not set. "
            "All chat evaluation endpoints will be disabled."
        )
        return router

    model_anthropic_claude: ChatAnthropic = ChatAnthropic(model="claude-2.0")

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

    return router
