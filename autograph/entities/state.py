import operator
from typing import Annotated, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

__all__ = [
    "WorkflowState",
    "Action",
    "Cost",
]


class Action(TypedDict):
    actor: str
    contents: str
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, str]]
    tool_result: Optional[str]
    time: str


class Cost(TypedDict):
    input_tokens: int
    output_tokens: int
    llm_calls: int


def reduce_cost_state(
    current: Dict[str, Cost], new: Tuple[str, Cost] | None
) -> Dict[str, Cost]:
    # reducers can be called multiple times by the LangGraph framework. One MUST asure
    # that fully new object is returned from reducer function. If mutation happens instead
    # results might be broken!!!!!!
    reduced = {
        k: Cost(
            llm_calls=v["llm_calls"],
            input_tokens=v["input_tokens"],
            output_tokens=v["output_tokens"],
        )
        for k, v in current.items()
    }

    if new is None:
        return reduced

    model_name, model_cost = new
    if model_name in reduced:
        reduced[model_name]["llm_calls"] += model_cost["llm_calls"]
        reduced[model_name]["input_tokens"] += model_cost["input_tokens"]
        reduced[model_name]["output_tokens"] += model_cost["output_tokens"]
    else:
        reduced[model_name] = model_cost

    return reduced


class WorkflowState(TypedDict):
    goal: str
    plan: List[str]  # Follow up will add full implementation
    previous_step_summary: Optional[str]
    messages: List[BaseMessage]
    # ------Presentation Layer------
    actions: Annotated[Sequence[Action], operator.add]
    costs: Annotated[Dict[str, Cost], reduce_cost_state]  # Costs by model
    # agent_status: str
    # workflow_status: str
    # human_question: Optional[str]
