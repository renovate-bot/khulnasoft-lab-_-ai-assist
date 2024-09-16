import re
from typing import Any, AsyncIterator, Optional, Sequence

import structlog
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from ai_gateway.chat.agents.typing import (
    AdditionalContext,
    AgentFinalAnswer,
    AgentStep,
    AgentToolAction,
    AgentUnknownAction,
    Context,
    CurrentFile,
    TypeAgentEvent,
)
from ai_gateway.chat.tools.base import BaseTool
from ai_gateway.feature_flags import FeatureFlag, is_feature_enabled
from ai_gateway.models.base_chat import Message, Role
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.typing import ModelMetadata

__all__ = [
    "ReActAgentInputs",
    "ReActPlainTextParser",
    "ReActAgent",
]

log = structlog.stdlib.get_logger("react")


class ReActAgentInputs(BaseModel):
    question: str
    chat_history: list[Message] | list[str] | str
    agent_scratchpad: list[AgentStep]
    context: Optional[Context] = None
    current_file: Optional[CurrentFile] = None
    model_metadata: Optional[ModelMetadata] = None
    additional_context: Optional[list[AdditionalContext]] = None
    unavailable_resources: Optional[list[str]] = [
        "Merge Requests, Pipelines, Vulnerabilities"
    ]
    tools: Optional[list[BaseTool]] = None


class ReActInputParser(Runnable[ReActAgentInputs, dict]):
    def invoke(
        self, input: ReActAgentInputs, config: Optional[RunnableConfig] = None
    ) -> dict:
        final_inputs = {
            "additional_context": input.additional_context,
            "context_type": "",
            "context_content": "",
            "question": input.question,
            "agent_scratchpad": agent_scratchpad_plain_text_renderer(
                input.agent_scratchpad
            ),
            "current_file": input.current_file,
            "unavailable_resources": input.unavailable_resources,
            "tools": input.tools,
        }

        if isinstance(input.chat_history, list) and any(
            isinstance(m, Message) for m in input.chat_history
        ):
            pass  # no-op
        else:
            # Legacy support for chat history as a string
            final_inputs.update(
                {"chat_history": chat_history_plain_text_renderer(input.chat_history)}
            )

        if context := input.context:
            final_inputs.update(
                {"context_type": context.type, "context_content": context.content}
            )

        if is_feature_enabled(FeatureFlag.EXPANDED_AI_LOGGING):
            log.info("ReActInputParser", source=__name__, final_inputs=final_inputs)

        return final_inputs


def chat_history_plain_text_renderer(chat_history: list | str) -> str:
    if isinstance(chat_history, list):
        return "\n".join(chat_history)

    return chat_history


def agent_scratchpad_plain_text_renderer(
    scratchpad: list[AgentStep],
) -> str:
    tpl = (
        "Thought: {thought}\n"
        "Action: {action}\n"
        "Action Input: {action_input}\n"
        "Observation: {observation}"
    )

    steps = [
        tpl.format(
            thought=pad.action.thought,
            action=pad.action.tool,
            action_input=pad.action.tool_input,
            observation=pad.observation,
        )
        for pad in scratchpad
        if isinstance(pad.action, AgentToolAction)
    ]

    return "\n".join(steps)


class ReActPlainTextParser(BaseCumulativeTransformOutputParser):
    re_thought = re.compile(
        r"<message>Thought:\s*([\s\S]*?)\s*(?:Action|Final Answer):"
    )
    re_action = re.compile(r"Action:\s*([\s\S]*?)\s*Action", re.DOTALL)
    re_action_input = re.compile(r"Action Input:\s*([\s\S]*?)\s*</message>")
    re_final_answer = re.compile(r"Final Answer:\s*([\s\S]*?)\s*</message>")

    def _parse_final_answer(self, message: str) -> Optional[AgentFinalAnswer]:
        if match_answer := self.re_final_answer.search(message):
            match_thought = self.re_thought.search(message)

            return AgentFinalAnswer(
                thought=match_thought.group(1) if match_thought else "",
                text=match_answer.group(1),
            )

        return None

    def _parse_agent_action(self, message: str) -> Optional[AgentToolAction]:
        match_action = self.re_action.search(message)
        match_action_input = self.re_action_input.search(message)
        match_thought = self.re_thought.search(message)

        if match_action and match_action_input:
            return AgentToolAction(
                tool=match_action.group(1),
                tool_input=match_action_input.group(1),
                thought=match_thought.group(1) if match_thought else "",
            )

        return None

    def _parse(self, text: str) -> TypeAgentEvent:
        wrapped_text = f"<message>Thought: {text}</message>"

        event: Optional[TypeAgentEvent] = None
        if final_answer := self._parse_final_answer(wrapped_text):
            event = final_answer
        elif agent_action := self._parse_agent_action(wrapped_text):
            event = agent_action
        else:
            event = AgentUnknownAction(text=text)

        return event

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Optional[TypeAgentEvent]:
        event = None
        text = result[0].text.strip()

        try:
            event = self._parse(text)
        except ValueError as e:
            if not partial:
                msg = f"Invalid output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

        return event

    def parse(self, text: str) -> Optional[TypeAgentEvent]:
        return self.parse_result([Generation(text=text)])


class ReActAgent(Prompt[ReActAgentInputs, TypeAgentEvent]):
    @staticmethod
    def _build_chain(
        chain: Runnable[ReActAgentInputs, TypeAgentEvent]
    ) -> Runnable[ReActAgentInputs, TypeAgentEvent]:
        return ReActInputParser() | chain | ReActPlainTextParser()

    @classmethod
    def build_messages(
        cls,
        prompt_template: dict[str, str],
        chat_history: list[Message] | list[str] | str,
        **kwargs,
    ) -> Sequence[MessageLikeRepresentation]:
        messages = []

        if "system" in prompt_template:
            messages.append(("system", prompt_template["system"]))

        # NOTE: You MUST encapsulate arbitrary inputs into HumanMessage or AIMessage.
        # Do NOT use other types (e.g. tuple).
        # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/604
        if isinstance(chat_history, list) and all(
            isinstance(m, Message) for m in chat_history
        ):
            for m in chat_history:
                if m.role is Role.USER:
                    messages.append(HumanMessage(m.content))
                elif m.role is Role.ASSISTANT:
                    messages.append(AIMessage(m.content))
                else:
                    raise ValueError("Unsupported message")

        if "user" in prompt_template:
            messages.append(("user", prompt_template["user"]))

        if "assistant" in prompt_template:
            messages.append(("assistant", prompt_template["assistant"]))

        return messages

    async def astream(
        self,
        input: ReActAgentInputs,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[TypeAgentEvent]:
        events = []
        astream = super().astream(input, config=config, **kwargs)
        len_final_answer = 0

        async for event in astream:
            if is_feature_enabled(FeatureFlag.EXPANDED_AI_LOGGING):
                log.info("Response streaming", source=__name__, streamed_event=event)

            if isinstance(event, AgentFinalAnswer) and len(event.text) > 0:
                yield AgentFinalAnswer(
                    text=event.text[len_final_answer:],
                )

                len_final_answer = len(event.text)

            events.append(event)

        if any(isinstance(e, AgentFinalAnswer) for e in events):
            pass  # no-op
        elif any(isinstance(e, AgentToolAction) for e in events):
            yield events[-1]
        elif isinstance(events[-1], AgentUnknownAction):
            yield events[-1]
