import re
from typing import Any, AsyncIterator, Optional, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from ai_gateway.chat.agents.typing import (
    AdditionalContext,
    AgentFinalAnswer,
    AgentStep,
    AgentToolAction,
    Context,
    CurrentFile,
)
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.typing import ModelMetadata

__all__ = [
    "ReActAgentInputs",
    "ReActAgentToolAction",
    "ReActAgentFinalAnswer",
    "TypeReActAgentAction",
    "ReActPlainTextParser",
    "ReActAgent",
]


class ReActAgentToolAction(AgentToolAction):
    pass


class ReActAgentFinalAnswer(AgentFinalAnswer):
    pass


TypeReActAgentAction = ReActAgentToolAction | ReActAgentFinalAnswer


class ReActAgentInputs(BaseModel):
    question: str
    chat_history: str | list[str]
    agent_scratchpad: list[AgentStep[TypeReActAgentAction]]
    context: Optional[Context] = None
    current_file: Optional[CurrentFile] = None
    model_metadata: Optional[ModelMetadata] = None
    additional_context: Optional[list[AdditionalContext]] = None


class ReActInputParser(Runnable[ReActAgentInputs, dict]):
    def invoke(
        self, input: ReActAgentInputs, config: Optional[RunnableConfig] = None
    ) -> dict:
        return {
            "chat_history": chat_history_plain_text_renderer(input.chat_history),
            "question": input.question,
            "agent_scratchpad": agent_scratchpad_plain_text_renderer(
                input.agent_scratchpad
            ),
            "current_file": input.current_file,
        }


def chat_history_plain_text_renderer(chat_history: list | str) -> str:
    if isinstance(chat_history, list):
        return "\n".join(chat_history)

    return chat_history


def agent_scratchpad_plain_text_renderer(
    scratchpad: list[AgentStep[TypeReActAgentAction]],
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
        if isinstance(pad.action, ReActAgentToolAction)
    ]

    return "\n".join(steps)


class ReActPlainTextParser(BaseCumulativeTransformOutputParser):
    re_thought = re.compile(
        r"<message>Thought:\s*([\s\S]*?)\s*(?:Action|Final Answer):"
    )
    re_action = re.compile(r"Action:\s*([\s\S]*?)\s*Action", re.DOTALL)
    re_action_input = re.compile(r"Action Input:\s*([\s\S]*?)\s*</message>")
    re_final_answer = re.compile(r"Final Answer:\s*([\s\S]*?)\s*</message>")

    def _parse_final_answer(self, message: str) -> Optional[ReActAgentFinalAnswer]:
        if match_answer := self.re_final_answer.search(message):
            match_thought = self.re_thought.search(message)

            return ReActAgentFinalAnswer(
                thought=match_thought.group(1) if match_thought else "",
                text=match_answer.group(1),
            )

        return None

    def _parse_agent_action(self, message: str) -> Optional[ReActAgentToolAction]:
        match_action = self.re_action.search(message)
        match_action_input = self.re_action_input.search(message)
        match_thought = self.re_thought.search(message)

        if match_action and match_action_input:
            return ReActAgentToolAction(
                tool=match_action.group(1),
                tool_input=match_action_input.group(1),
                thought=match_thought.group(1) if match_thought else "",
            )

        return None

    def _parse(self, text: str) -> TypeReActAgentAction:
        text = f"Thought: {text}"
        wrapped_text = f"<message>{text}</message>"

        message: Optional[TypeReActAgentAction] = None
        if final_answer := self._parse_final_answer(wrapped_text):
            message = final_answer
        elif agent_action := self._parse_agent_action(wrapped_text):
            message = agent_action

        if message is None:
            raise ValueError("incorrect `TypeReActAgentAction` schema output")

        return message

    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Optional[TypeReActAgentAction]:
        action = None
        text = result[0].text.strip()

        try:
            action = self._parse(text)
        except ValueError as e:
            if not partial:
                msg = f"Invalid output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

        return action

    def parse(self, text: str) -> Optional[TypeReActAgentAction]:
        return self.parse_result([Generation(text=text)])


class ReActAgent(Prompt[ReActAgentInputs, TypeReActAgentAction]):
    class _StreamState(TypedDict):
        tool_action: Optional[ReActAgentToolAction]
        len_final_answer: int
        len_thought: int

    @staticmethod
    def _build_chain(
        chain: Runnable[ReActAgentInputs, TypeReActAgentAction]
    ) -> Runnable[ReActAgentInputs, TypeReActAgentAction]:
        return ReActInputParser() | chain | ReActPlainTextParser()

    async def astream(
        self,
        input: ReActAgentInputs,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[TypeReActAgentAction]:
        state = ReActAgent._StreamState(
            tool_action=None,
            len_final_answer=0,
            len_thought=0,
        )

        astream = super().astream(input, config=config, **kwargs)

        async for action in astream:
            if isinstance(action, ReActAgentToolAction):
                state["tool_action"] = action
            elif (
                action
                and isinstance(action, ReActAgentFinalAnswer)
                and len(action.text) > 0
            ):
                yield ReActAgentFinalAnswer(
                    text=action.text[state["len_final_answer"] :],
                )

                state["len_final_answer"] = len(action.text)

        if tool_action := state.get("tool_action", None):
            yield tool_action
