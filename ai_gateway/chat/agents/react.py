import re
from typing import Any, AsyncIterator, Callable, Optional, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ai_gateway.agents.base import Agent
from ai_gateway.chat.agents.base import (
    AgentFinalAnswer,
    AgentStep,
    AgentToolAction,
    BaseParser,
    BaseSingleActionAgent,
)
from ai_gateway.chat.agents.utils import convert_prompt_to_messages
from ai_gateway.chat.tools import BaseTool
from ai_gateway.chat.typing import Context
from ai_gateway.models import Message, TextGenModelChunk

__all__ = [
    "TypeReActAgentAction",
    "ReActAgentInputs",
    "ReActAgentMessage",
    "ReActAgentToolAction",
    "ReActAgentFinalAnswer",
    "ReActPlainTextParser",
    "chat_history_plain_text_renderer",
    "agent_scratchpad_plain_text_renderer",
    "ReActAgent",
]


class ReActAgentInputs(BaseModel):
    question: str
    chat_history: str | list[str]
    context: Optional[Context] = None


class ReActAgentMessage(BaseModel):
    thought: str


class ReActAgentToolAction(AgentToolAction, ReActAgentMessage):
    pass


class ReActAgentFinalAnswer(AgentFinalAnswer, ReActAgentMessage):
    pass


TypeReActAgentAction = ReActAgentToolAction | ReActAgentFinalAnswer


class ReActPlainTextParser(BaseParser):
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

    def parse(self, text: str) -> TypeReActAgentAction:
        text = f"Thought: {text}"
        message = f"<message>{text}</message>"

        if final_answer := self._parse_final_answer(message):
            message = final_answer
        elif agent_action := self._parse_agent_action(message):
            message = agent_action
        else:
            raise ValueError("parser error")

        message.log = text

        return message


def chat_history_plain_text_renderer(inputs: ReActAgentInputs) -> str:
    if isinstance(inputs.chat_history, list):
        return "\n".join(inputs.chat_history)

    return inputs.chat_history


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


class _StreamState(TypedDict):
    tool_action: AgentToolAction
    len_final_answer: int
    len_log: int
    len_thought: int


class ReActAgent(BaseSingleActionAgent):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    # TODO: Validate whether the agent's prompts have all the required placeholders specified in `ReActAgentInputs`.
    agent: Agent
    tools: Sequence[BaseTool]
    inputs: ReActAgentInputs
    parser: BaseParser = Field(default_factory=ReActPlainTextParser)
    render_chat_history: Callable[[ReActAgentInputs], str] = (
        chat_history_plain_text_renderer
    )
    render_agent_scratchpad: Callable[[list[AgentStep]], str] = (
        agent_scratchpad_plain_text_renderer
    )
    model_kwargs: dict[str, Any] = {"stop_sequences": ["Observation:"]}

    async def invoke(
        self, *, inputs: ReActAgentInputs, **kwargs: Any
    ) -> TypeReActAgentAction:
        messages = self._convert_prompt_to_messages(inputs)

        model_kwargs = {**self.model_kwargs, **kwargs}
        response = await self.agent.model.generate(
            messages, stream=False, **model_kwargs
        )
        parsed_action = self.parser.parse(response.text)

        return parsed_action

    async def stream(
        self, *, inputs: ReActAgentInputs, **kwargs
    ) -> AsyncIterator[TypeReActAgentAction]:
        messages = self._convert_prompt_to_messages(inputs)

        model_kwargs = {**self.model_kwargs, **kwargs}
        actions_stream = await self.agent.model.generate(
            messages, stream=True, **model_kwargs
        )

        state = _StreamState(
            tool_action=None,
            len_final_answer=0,
            len_log=0,
            len_thought=0,
        )

        async for action in self._parse_stream(actions_stream):
            if isinstance(action, AgentToolAction):
                state["tool_action"] = action
            elif isinstance(action, ReActAgentFinalAnswer) and len(action.text) > 0:
                yield ReActAgentFinalAnswer(
                    thought=action.thought[state["len_thought"] :],
                    text=action.text[state["len_final_answer"] :],
                    log=action.log[state["len_log"] :],
                )

                state["len_thought"] = len(action.thought)
                state["len_final_answer"] = len(action.text)
                state["len_log"] = len(action.log)

        if tool_action := state.get("tool_action", None):
            yield tool_action

    async def _parse_stream(
        self, stream: AsyncIterator[TextGenModelChunk]
    ) -> AsyncIterator[TypeReActAgentAction]:
        text = ""
        async for chunk in stream:
            try:
                text += chunk.text
                yield self.parser.parse(text)
            except ValueError:
                pass

    def _convert_prompt_to_messages(self, inputs: ReActAgentInputs) -> list[Message]:
        return convert_prompt_to_messages(
            self.agent,
            tools=self.tools,
            context_type=self.inputs.context.type if self.inputs.context else None,
            question=inputs.question,
            chat_history=self.render_chat_history(inputs),
            agent_scratchpad=self.render_agent_scratchpad(self.agent_scratchpad),
            context_content=inputs.context.content if inputs.context else "",
        )
