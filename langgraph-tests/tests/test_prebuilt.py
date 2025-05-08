from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import (
    create_react_agent,
)
from langgraph.prebuilt.chat_agent_executor import (
    StructuredResponse,
)
from tests.any_str import AnyStr
from tests.conftest import (
    ALL_CHECKPOINTERS_ASYNC,
    ALL_CHECKPOINTERS_SYNC,
    awith_checkpointer,
)
from tests.messages import _AnyIdHumanMessage

pytestmark = pytest.mark.anyio

REACT_TOOL_CALL_VERSIONS = ["v1", "v2"]


class FakeToolCallingModel(BaseChatModel):
    tool_calls: Optional[list[list[ToolCall]]] = None
    structured_response: Optional[StructuredResponse] = None
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_string = "-".join([m.content for m in messages])
        tool_calls = (
            self.tool_calls[self.index % len(self.tool_calls)]
            if self.tool_calls
            else []
        )
        message = AIMessage(
            content=messages_string, id=str(self.index), tool_calls=tool_calls.copy()
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def with_structured_output(
        self, schema: Type[BaseModel]
    ) -> Runnable[LanguageModelInput, StructuredResponse]:
        if self.structured_response is None:
            raise ValueError("Structured response is not set")

        return RunnableLambda(lambda x: self.structured_response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            raise ValueError("Must provide at least one tool")

        tool_dicts = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise TypeError(
                    "Only BaseTool is supported by FakeToolCallingModel.bind_tools"
                )

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts)


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
@pytest.mark.parametrize("version", REACT_TOOL_CALL_VERSIONS)
def test_no_prompt(
    request: pytest.FixtureRequest, checkpointer_name: str, version: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        "checkpointer_" + checkpointer_name
    )
    model = FakeToolCallingModel()

    agent = create_react_agent(
        model,
        [],
        checkpointer=checkpointer,
        version=version,
    )
    inputs = [HumanMessage("hi?")]
    thread = {"configurable": {"thread_id": "123"}}
    response = agent.invoke({"messages": inputs}, thread, debug=True)
    expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
    assert response == expected_response

    if checkpointer:
        saved = checkpointer.get_tuple(thread)
        assert saved is not None
        assert saved.checkpoint["channel_values"] == {
            "messages": [
                _AnyIdHumanMessage(content="hi?"),
                AIMessage(content="hi?", id="0"),
            ],
            "agent": "agent",
        }
        assert saved.metadata == {
            "checkpoint_id": AnyStr(),
            "checkpoint_ns": "",
            "parents": {},
            "source": "loop",
            "writes": {"agent": {"messages": [AIMessage(content="hi?", id="0")]}},
            "step": 1,
            "thread_id": "123",
        }
        assert saved.pending_writes == []


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_ASYNC)
async def test_no_prompt_async(checkpointer_name: str) -> None:
    async with awith_checkpointer(checkpointer_name) as checkpointer:
        model = FakeToolCallingModel()

        agent = create_react_agent(model, [], checkpointer=checkpointer)
        inputs = [HumanMessage("hi?")]
        thread = {"configurable": {"thread_id": "123"}}
        response = await agent.ainvoke({"messages": inputs}, thread, debug=True)
        expected_response = {"messages": inputs + [AIMessage(content="hi?", id="0")]}
        assert response == expected_response

        if checkpointer:
            saved = await checkpointer.aget_tuple(thread)
            assert saved is not None
            assert saved.checkpoint["channel_values"] == {
                "messages": [
                    _AnyIdHumanMessage(content="hi?"),
                    AIMessage(content="hi?", id="0"),
                ],
                "agent": "agent",
            }
            assert saved.metadata == {
                "checkpoint_id": AnyStr(),
                "checkpoint_ns": "",
                "parents": {},
                "source": "loop",
                "writes": {"agent": {"messages": [AIMessage(content="hi?", id="0")]}},
                "step": 1,
                "thread_id": "123",
            }
            assert saved.pending_writes == []
