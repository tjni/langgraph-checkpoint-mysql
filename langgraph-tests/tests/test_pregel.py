import json
import operator
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from typing import (
    Annotated,
    Any,
    Dict,
    Iterator,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)

import httpx
import pytest
from langchain_core.runnables import (
    RunnableConfig,
    RunnableMap,
    RunnablePick,
)
from pydantic import BaseModel
from pytest_mock import MockerFixture
from syrupy import SnapshotAssertion

from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.context import Context
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.constants import (
    CONFIG_KEY_NODE_FINISHED,
    ERROR,
    FF_SEND_V2,
    PULL,
    PUSH,
    START,
)
from langgraph.errors import MultipleSubgraphsError, NodeInterrupt
from langgraph.graph import END, Graph, StateGraph
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.managed.shared_value import SharedValue
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import Channel, Pregel, StateSnapshot
from langgraph.pregel.retry import RetryPolicy
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import (
    Command,
    Interrupt,
    PregelTask,
    Send,
    StreamWriter,
    interrupt,
)
from tests.any_str import AnyDict, AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.conftest import (
    ALL_CHECKPOINTERS_SYNC,
    ALL_STORES_SYNC,
    SHOULD_CHECK_SNAPSHOTS,
)
from tests.fake_tracer import FakeTracer
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)


# define these objects to avoid importing langchain_core.agents
# and therefore avoid relying on core Pydantic version
class AgentAction(BaseModel):
    tool: str
    tool_input: Union[str, dict]
    log: str
    type: Literal["AgentAction"] = "AgentAction"

    model_config = {
        "json_schema_extra": {
            "description": (
                """Represents a request to execute an action by an agent.

The action consists of the name of the tool to execute and the input to pass
to the tool. The log is used to pass along extra information about the action."""
            )
        }
    }


class AgentFinish(BaseModel):
    """Final return value of an ActionAgent.

    Agents return an AgentFinish when they have reached a stopping condition.
    """

    return_values: dict
    log: str
    type: Literal["AgentFinish"] = "AgentFinish"
    model_config = {
        "json_schema_extra": {
            "description": (
                """Final return value of an ActionAgent.

Agents return an AgentFinish when they have reached a stopping condition."""
            )
        }
    }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_two_processes_in_out_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    one = Channel.subscribe_to("input") | add_one | Channel.write_to("inbox")
    two = Channel.subscribe_to("inbox") | add_one | Channel.write_to("output")

    app = Pregel(
        nodes={"one": one, "two": two},
        channels={
            "inbox": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=checkpointer,
        interrupt_after_nodes=["one"],
    )
    thread1 = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}

    # start execution, stop at inbox
    assert app.invoke(2, thread1) is None

    # inbox == 3
    checkpoint = checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 3

    # resume execution, finish
    assert app.invoke(None, thread1) == 4

    # start execution again, stop at inbox
    assert app.invoke(20, thread1) is None

    # inbox == 21
    checkpoint = checkpointer.get(thread1)
    assert checkpoint is not None
    assert checkpoint["channel_values"]["inbox"] == 21

    # send a new value in, interrupting the previous execution
    assert app.invoke(3, thread1) is None
    assert app.invoke(None, thread1) == 5

    # start execution again, stopping at inbox
    assert app.invoke(20, thread2) is None

    # inbox == 21
    snapshot = app.get_state(thread2)
    assert snapshot.values["inbox"] == 21
    assert snapshot.next == ("two",)

    # update the state, resume
    app.update_state(thread2, 25, as_node="one")
    assert app.invoke(None, thread2) == 26

    # no pending tasks
    snapshot = app.get_state(thread2)
    assert snapshot.next == ()

    # list history
    history = [c for c in app.get_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values={"inbox": 4, "output": 5, "input": 3},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 6,
                "writes": {"two": 5},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values={"inbox": 4, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 5}),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
                "writes": {"one": None},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 3},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 4}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": 4,
                "writes": {"input": 3},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values={"inbox": 21, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two")),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"one": None},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 20},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 21}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": 2,
                "writes": {"input": 20},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "output": 4, "input": 2},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"two": 4},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values={"inbox": 3, "input": 2},
            tasks=(PregelTask(AnyStr(), "two", (PULL, "two"), result={"output": 4}),),
            next=("two",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": {"one": None},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[7].config,
        ),
        StateSnapshot(
            values={"input": 2},
            tasks=(PregelTask(AnyStr(), "one", (PULL, "one"), result={"inbox": 3}),),
            next=("one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": -1,
                "writes": {"input": 2},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # re-running from any previous checkpoint should re-run nodes
    assert [c for c in app.stream(None, history[0].config, stream_mode="updates")] == []
    assert [c for c in app.stream(None, history[1].config, stream_mode="updates")] == [
        {"two": {"output": 5}},
    ]
    assert [c for c in app.stream(None, history[2].config, stream_mode="updates")] == [
        {"one": {"inbox": 4}},
        {"__interrupt__": ()},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_fork_always_re_runs_nodes(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    add_one = mocker.Mock(side_effect=lambda _: 1)

    builder = StateGraph(Annotated[int, operator.add])
    builder.add_node("add_one", add_one)
    builder.add_edge(START, "add_one")
    builder.add_conditional_edges("add_one", lambda cnt: "add_one" if cnt < 6 else END)
    graph = builder.compile(checkpointer=checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}

    # start execution, stop at inbox
    assert [*graph.stream(1, thread1, stream_mode=["values", "updates"])] == [
        ("values", 1),
        ("updates", {"add_one": 1}),
        ("values", 2),
        ("updates", {"add_one": 1}),
        ("values", 3),
        ("updates", {"add_one": 1}),
        ("values", 4),
        ("updates", {"add_one": 1}),
        ("values", 5),
        ("updates", {"add_one": 1}),
        ("values", 6),
    ]

    # list history
    history = [c for c in graph.get_state_history(thread1)]
    assert history == [
        StateSnapshot(
            values=6,
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 5,
                "writes": {"add_one": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[1].config,
        ),
        StateSnapshot(
            values=5,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 4,
                "writes": {"add_one": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[2].config,
        ),
        StateSnapshot(
            values=4,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 3,
                "writes": {"add_one": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[3].config,
        ),
        StateSnapshot(
            values=3,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 2,
                "writes": {"add_one": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[4].config,
        ),
        StateSnapshot(
            values=2,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 1,
                "writes": {"add_one": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[5].config,
        ),
        StateSnapshot(
            values=1,
            tasks=(PregelTask(AnyStr(), "add_one", (PULL, "add_one"), result=1),),
            next=("add_one",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "step": 0,
                "writes": None,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=history[6].config,
        ),
        StateSnapshot(
            values=0,
            tasks=(PregelTask(AnyStr(), "__start__", (PULL, "__start__"), result=1),),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "step": -1,
                "writes": {"__start__": 1},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]

    # forking from any previous checkpoint should re-run nodes
    assert [
        c for c in graph.stream(None, history[0].config, stream_mode="updates")
    ] == []
    assert [
        c for c in graph.stream(None, history[1].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
    ]
    assert [
        c for c in graph.stream(None, history[2].config, stream_mode="updates")
    ] == [
        {"add_one": 1},
        {"add_one": 1},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_run_from_checkpoint_id_retains_previous_writes(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class MyState(TypedDict):
        myval: Annotated[int, operator.add]
        otherval: bool

    class Anode:
        def __init__(self):
            self.switch = False

        def __call__(self, state: MyState):
            self.switch = not self.switch
            return {"myval": 2 if self.switch else 1, "otherval": self.switch}

    builder = StateGraph(MyState)
    thenode = Anode()  # Fun.
    builder.add_node("node_one", thenode)
    builder.add_node("node_two", thenode)
    builder.add_edge(START, "node_one")

    def _getedge(src: str):
        swap = "node_one" if src == "node_two" else "node_two"

        def _edge(st: MyState) -> Literal["__end__", "node_one", "node_two"]:
            if st["myval"] > 3:
                return END
            if st["otherval"]:
                return swap
            return src

        return _edge

    builder.add_conditional_edges("node_one", _getedge("node_one"))
    builder.add_conditional_edges("node_two", _getedge("node_two"))
    graph = builder.compile(checkpointer=checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = graph.invoke({"myval": 1}, thread1)
    assert result["myval"] == 4
    history = [c for c in graph.get_state_history(thread1)]

    assert len(history) == 4
    assert history[-1].values == {"myval": 0}
    assert history[0].values == {"myval": 4, "otherval": False}

    second_run_config = {
        **thread1,
        "configurable": {
            **thread1["configurable"],
            "checkpoint_id": history[1].config["configurable"]["checkpoint_id"],
        },
    }
    second_result = graph.invoke(None, second_run_config)
    assert second_result == {"myval": 5, "otherval": True}

    new_history = [
        c
        for c in graph.get_state_history(
            {"configurable": {"thread_id": str(thread_id), "checkpoint_ns": ""}}
        )
    ]

    assert len(new_history) == len(history) + 1
    for original, new in zip(history, new_history[1:]):
        assert original.values == new.values
        assert original.next == new.next
        assert original.metadata["step"] == new.metadata["step"]

    def _get_tasks(hist: list, start: int):
        return [h.tasks for h in hist[start:]]

    assert _get_tasks(new_history, 1) == _get_tasks(history, 0)


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_checkpoint_two(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )
    add_one = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])
    errored_once = False

    def raise_if_above_10(input: int) -> int:
        nonlocal errored_once
        if input > 4:
            if errored_once:
                pass
            else:
                errored_once = True
                raise ConnectionError("I will be retried")
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        Channel.subscribe_to(["input"]).join(["total"])
        | add_one
        | Channel.write_to("output", "total")
        | raise_if_above_10
    )

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=checkpointer,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint_tup = checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated, error is recorded
    checkpoint_tup = checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    assert checkpoint_tup.pending_writes == [
        (AnyStr(), ERROR, "ValueError('Input is too large')")
    ]
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = checkpointer.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_pending_writes_resume(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self, sleep: float, rtn: Union[Dict, Exception]) -> None:
            self.sleep = sleep
            self.rtn = rtn
            self.reset()

        def __call__(self, input: State) -> Any:
            self.calls += 1
            time.sleep(self.sleep)
            if isinstance(self.rtn, Exception):
                raise self.rtn
            else:
                return self.rtn

        def reset(self):
            self.calls = 0

    one = AwhileMaker(0.1, {"value": 2})
    two = AwhileMaker(0.3, ConnectionError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node("two", two, retry=RetryPolicy(max_attempts=2))
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    graph = builder.compile(checkpointer=checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke({"value": 1}, thread1)

    # both nodes should have been called once
    assert one.calls == 1
    assert two.calls == 2  # two attempts

    # latest checkpoint should be before nodes "one", "two"
    # but we should have applied the write from "one"
    state = graph.get_state(thread1)
    assert state is not None
    assert state.values == {"value": 3}
    assert state.next == ("two",)
    assert state.tasks == (
        PregelTask(AnyStr(), "one", (PULL, "one"), result={"value": 2}),
        PregelTask(AnyStr(), "two", (PULL, "two"), 'ConnectionError("I\'m not good")'),
    )
    assert state.metadata == {
        "parents": {},
        "source": "loop",
        "step": 0,
        "writes": None,
        "thread_id": "1",
    }
    # get_state with checkpoint_id should not apply any pending writes
    state = graph.get_state(state.config)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    # should contain pending write of "one"
    checkpoint = checkpointer.get_tuple(thread1)
    assert checkpoint is not None
    # should contain error from "two"
    expected_writes = [
        (AnyStr(), "one", "one"),
        (AnyStr(), "value", 2),
        (AnyStr(), ERROR, 'ConnectionError("I\'m not good")'),
    ]
    assert len(checkpoint.pending_writes) == 3
    assert all(w in expected_writes for w in checkpoint.pending_writes)
    # both non-error pending writes come from same task
    non_error_writes = [w for w in checkpoint.pending_writes if w[1] != ERROR]
    assert non_error_writes[0][0] == non_error_writes[1][0]
    # error write is from the other task
    error_write = next(w for w in checkpoint.pending_writes if w[1] == ERROR)
    assert error_write[0] != non_error_writes[0][0]

    # resume execution
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke(None, thread1)

    # node "one" succeeded previously, so shouldn't be called again
    assert one.calls == 1
    # node "two" should have been called once again
    assert two.calls == 4  # two attempts before + two attempts now

    # confirm no new checkpoints saved
    state_two = graph.get_state(thread1)
    assert state_two.metadata == state.metadata

    # resume execution, without exception
    two.rtn = {"value": 3}
    # both the pending write and the new write were applied, 1 + 2 + 3 = 6
    assert graph.invoke(None, thread1) == {"value": 6}

    # check all final checkpoints
    checkpoints = [c for c in checkpointer.list(thread1)]
    # we should have 3
    assert len(checkpoints) == 3
    # the last one not too interesting for this test
    assert checkpoints[0] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {
                "one": {
                    "start:one": AnyVersion(),
                },
                "two": {
                    "start:two": AnyVersion(),
                },
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
                "__interrupt__": {
                    "value": AnyVersion(),
                    "__start__": AnyVersion(),
                    "start:one": AnyVersion(),
                    "start:two": AnyVersion(),
                },
            },
            "channel_versions": {
                "one": AnyVersion(),
                "two": AnyVersion(),
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "start:one": AnyVersion(),
                "start:two": AnyVersion(),
            },
            "channel_values": {"one": "one", "two": "two", "value": 6},
        },
        metadata={
            "parents": {},
            "step": 1,
            "source": "loop",
            "writes": {"one": {"value": 2}, "two": {"value": 3}},
            "thread_id": "1",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoints[1].config["configurable"]["checkpoint_id"],
            }
        },
        pending_writes=[],
    )
    # the previous one we assert that pending writes contains both
    # - original error
    # - successful writes from resuming after preventing error
    assert checkpoints[1] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "start:one": AnyVersion(),
                "start:two": AnyVersion(),
            },
            "channel_values": {
                "value": 1,
                "start:one": "__start__",
                "start:two": "__start__",
            },
        },
        metadata={
            "parents": {},
            "step": 0,
            "source": "loop",
            "writes": None,
            "thread_id": "1",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoints[2].config["configurable"]["checkpoint_id"],
            }
        },
        pending_writes=UnsortedSequence(
            (AnyStr(), "one", "one"),
            (AnyStr(), "value", 2),
            (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
            (AnyStr(), "two", "two"),
            (AnyStr(), "value", 3),
        ),
    )
    assert checkpoints[2] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 1,
            "id": AnyStr(),
            "ts": AnyStr(),
            "pending_sends": [],
            "versions_seen": {"__input__": {}},
            "channel_versions": {
                "__start__": AnyVersion(),
            },
            "channel_values": {"__start__": {"value": 1}},
        },
        metadata={
            "parents": {},
            "step": -1,
            "source": "input",
            "writes": {"__start__": {"value": 1}},
            "thread_id": "1",
        },
        parent_config=None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 1),
            (AnyStr(), "start:one", "__start__"),
            (AnyStr(), "start:two", "__start__"),
        ),
    )


@pytest.mark.repeat(20)
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_send_dedupe_on_resume(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    if not FF_SEND_V2:
        pytest.skip("Send deduplication is only available in Send V2")
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class InterruptOnce:
        ticks: int = 0

        def __call__(self, state):
            self.ticks += 1
            if self.ticks == 1:
                raise NodeInterrupt("Bahh")
            return ["|".join(("flaky", str(state)))]

    class Node:
        def __init__(self, name: str):
            self.name = name
            self.ticks = 0
            setattr(self, "__name__", name)

        def __call__(self, state):
            self.ticks += 1
            update = (
                [self.name]
                if isinstance(state, list)
                else ["|".join((self.name, str(state)))]
            )
            if isinstance(state, Command):
                return replace(state, update=update)
            else:
                return update

    def send_for_fun(state):
        return [
            Send("2", Command(goto=Send("2", 3))),
            Send("2", Command(goto=Send("flaky", 4))),
            "3.1",
        ]

    def route_to_three(state) -> Literal["3"]:
        return "3"

    builder = StateGraph(Annotated[list, operator.add])
    builder.add_node(Node("1"))
    builder.add_node(Node("2"))
    builder.add_node(Node("3"))
    builder.add_node(Node("3.1"))
    builder.add_node("flaky", InterruptOnce())
    builder.add_edge(START, "1")
    builder.add_conditional_edges("1", send_for_fun)
    builder.add_conditional_edges("2", route_to_three)

    graph = builder.compile(checkpointer=checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke(["0"], thread1, debug=1) == [
        "0",
        "1",
        "2|Command(goto=Send(node='2', arg=3))",
        "2|Command(goto=Send(node='flaky', arg=4))",
        "2|3",
    ]
    assert builder.nodes["2"].runnable.func.ticks == 3
    assert builder.nodes["flaky"].runnable.func.ticks == 1
    # check state
    state = graph.get_state(thread1)
    assert state.next == ("flaky",)
    # check history
    history = [c for c in graph.get_state_history(thread1)]
    assert len(history) == 2
    # resume execution
    assert graph.invoke(None, thread1, debug=1) == [
        "0",
        "1",
        "2|Command(goto=Send(node='2', arg=3))",
        "2|Command(goto=Send(node='flaky', arg=4))",
        "2|3",
        "flaky|4",
        "3",
        "3.1",
    ]
    # node "2" doesn't get called again, as we recover writes saved before
    assert builder.nodes["2"].runnable.func.ticks == 3
    # node "flaky" gets called again, as it was interrupted
    assert builder.nodes["flaky"].runnable.func.ticks == 2
    # check state
    state = graph.get_state(thread1)
    assert state.next == ()
    # check history
    history = [c for c in graph.get_state_history(thread1)]
    assert (
        history[1]
        == [
            StateSnapshot(
                values=[
                    "0",
                    "1",
                    "2|Command(goto=Send(node='2', arg=3))",
                    "2|Command(goto=Send(node='flaky', arg=4))",
                    "2|3",
                    "flaky|4",
                    "3",
                    "3.1",
                ],
                next=(),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": {"3": ["3"], "3.1": ["3.1"]},
                    "thread_id": "1",
                    "step": 2,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(),
            ),
            StateSnapshot(
                values=[
                    "0",
                    "1",
                    "2|Command(goto=Send(node='2', arg=3))",
                    "2|Command(goto=Send(node='flaky', arg=4))",
                    "2|3",
                    "flaky|4",
                ],
                next=("3", "3.1"),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": {
                        "1": ["1"],
                        "2": [
                            ["2|Command(goto=Send(node='2', arg=3))"],
                            ["2|Command(goto=Send(node='flaky', arg=4))"],
                            ["2|3"],
                        ],
                        "flaky": ["flaky|4"],
                    },
                    "thread_id": "1",
                    "step": 1,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="3",
                        path=("__pregel_pull", "3"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["3"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="3.1",
                        path=("__pregel_pull", "3.1"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["3.1"],
                    ),
                ),
            ),
            StateSnapshot(
                values=["0"],
                next=("1", "2", "2", "2", "flaky"),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "loop",
                    "writes": None,
                    "thread_id": "1",
                    "step": 0,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="1",
                        path=("__pregel_pull", "1"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["1"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=(
                            "__pregel_push",
                            ("__pregel_pull", "1"),
                            2,
                            AnyStr(),
                        ),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|Command(goto=Send(node='2', arg=3))"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=(
                            "__pregel_push",
                            ("__pregel_pull", "1"),
                            3,
                            AnyStr(),
                        ),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|Command(goto=Send(node='flaky', arg=4))"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="2",
                        path=(
                            "__pregel_push",
                            (
                                "__pregel_push",
                                ("__pregel_pull", "1"),
                                2,
                                AnyStr(),
                            ),
                            2,
                            AnyStr(),
                        ),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["2|3"],
                    ),
                    PregelTask(
                        id=AnyStr(),
                        name="flaky",
                        path=(
                            "__pregel_push",
                            (
                                "__pregel_push",
                                ("__pregel_pull", "1"),
                                3,
                                AnyStr(),
                            ),
                            2,
                            AnyStr(),
                        ),
                        error=None,
                        interrupts=(Interrupt(value="Bahh", when="during"),),
                        state=None,
                        result=["flaky|4"],
                    ),
                ),
            ),
            StateSnapshot(
                values=[],
                next=("__start__",),
                config={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    }
                },
                metadata={
                    "source": "input",
                    "writes": {"__start__": ["0"]},
                    "thread_id": "1",
                    "step": -1,
                    "parents": {},
                },
                created_at=AnyStr(),
                parent_config=None,
                tasks=(
                    PregelTask(
                        id=AnyStr(),
                        name="__start__",
                        path=("__pregel_pull", "__start__"),
                        error=None,
                        interrupts=(),
                        state=None,
                        result=["0"],
                    ),
                ),
            ),
        ][1]
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_send_react_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    def agent(state):
        return {"messages": ai_message}

    def route(state):
        if isinstance(state["messages"][-1], AIMessage):
            return [
                Send(call["name"], call) for call in state["messages"][-1].tool_calls
            ]

    foo_called = 0

    def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route)
    graph = builder.compile()

    assert graph.invoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # interrupt-update-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    if not FF_SEND_V2:
        return

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 0,
            "source": "loop",
            "writes": None,
            "parents": {},
            "thread_id": "2",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        id="ai1",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    )
                },
            ),
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", ("__pregel_pull", "agent"), 2, AnyStr()),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # remove the tool call, clearing the pending task
    graph.update_state(
        thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
    )

    # tool call no longer in pending tasks
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="Bye now",
                    tool_calls=[],
                ),
            ]
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "update",
            "writes": {
                "agent": {
                    "messages": _AnyIdAIMessage(
                        content="Bye now",
                        tool_calls=[],
                    )
                }
            },
            "parents": {},
            "thread_id": "2",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(),
    )

    # tool call not executed
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(content="Bye now"),
        ]
    }
    assert foo_called == 0

    # interrupt-update-resume flow, creating new Send in update call
    foo_called = 0
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "3"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 0,
            "source": "loop",
            "writes": None,
            "parents": {},
            "thread_id": "3",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        "",
                        id="ai1",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    )
                },
            ),
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", ("__pregel_pull", "agent"), 2, AnyStr()),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # replace the tool call, should clear previous send, create new one
    graph.update_state(
        thread1,
        {
            "messages": AIMessage(
                "",
                id=ai_message.id,
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [4, 5, 6]},
                        "id": "tool1",
                        "type": "tool_call",
                    }
                ],
            )
        },
    )

    # prev tool call no longer in pending tasks, new tool call is
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [4, 5, 6]},
                            "id": "tool1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "update",
            "writes": {
                "agent": {
                    "messages": _AnyIdAIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [4, 5, 6]},
                                "id": "tool1",
                                "type": "tool_call",
                            }
                        ],
                    )
                }
            },
            "parents": {},
            "thread_id": "3",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "3",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", (), 0, AnyStr()),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # prev tool call not executed, new tool call is
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            AIMessage(
                "",
                id="ai1",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [4, 5, 6]},
                        "id": "tool1",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(content="{'hi': [4, 5, 6]}", tool_call_id="tool1"),
        ]
    }
    assert foo_called == 1


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_send_react_interrupt_control(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    ai_message = AIMessage(
        "",
        id="ai1",
        tool_calls=[ToolCall(name="foo", args={"hi": [1, 2, 3]}, id=AnyStr())],
    )

    def agent(state) -> Command[Literal["foo"]]:
        return Command(
            update={"messages": ai_message},
            goto=[Send(call["name"], call) for call in ai_message.tool_calls],
        )

    foo_called = 0

    def foo(call: ToolCall):
        nonlocal foo_called
        foo_called += 1
        return {"messages": ToolMessage(str(call["args"]), tool_call_id=call["id"])}

    builder = StateGraph(MessagesState)
    builder.add_node(agent)
    builder.add_node(foo)
    builder.add_edge(START, "agent")
    graph = builder.compile()
    assert graph.get_graph().draw_mermaid() == snapshot

    assert graph.invoke({"messages": [HumanMessage("hello")]}) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    # simple interrupt-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "1"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="{'hi': [1, 2, 3]}",
                tool_call_id=AnyStr(),
            ),
        ]
    }
    assert foo_called == 1

    if not FF_SEND_V2:
        return

    # interrupt-update-resume flow
    foo_called = 0
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["foo"])
    thread1 = {"configurable": {"thread_id": "2"}}
    assert graph.invoke({"messages": [HumanMessage("hello")]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "foo",
                        "args": {"hi": [1, 2, 3]},
                        "id": "",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    }
    assert foo_called == 0

    # get state should show the pending task
    state = graph.get_state(thread1)
    assert state == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "foo",
                            "args": {"hi": [1, 2, 3]},
                            "id": "",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        },
        next=("foo",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 0,
            "source": "loop",
            "writes": None,
            "parents": {},
            "thread_id": "2",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        id="ai1",
                        tool_calls=[
                            {
                                "name": "foo",
                                "args": {"hi": [1, 2, 3]},
                                "id": "",
                                "type": "tool_call",
                            }
                        ],
                    )
                },
            ),
            PregelTask(
                id=AnyStr(),
                name="foo",
                path=("__pregel_push", ("__pregel_pull", "agent"), 2, AnyStr()),
                error=None,
                interrupts=(),
                state=None,
                result=None,
            ),
        ),
    )

    # remove the tool call, clearing the pending task
    graph.update_state(
        thread1, {"messages": AIMessage("Bye now", id=ai_message.id, tool_calls=[])}
    )

    # tool call no longer in pending tasks
    assert graph.get_state(thread1) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="hello"),
                _AnyIdAIMessage(
                    content="Bye now",
                    tool_calls=[],
                ),
            ]
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "step": 1,
            "source": "update",
            "writes": {
                "agent": {
                    "messages": _AnyIdAIMessage(
                        content="Bye now",
                        tool_calls=[],
                    )
                }
            },
            "parents": {},
            "thread_id": "2",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(),
    )

    # tool call not executed
    assert graph.invoke(None, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            _AnyIdAIMessage(content="Bye now"),
        ]
    }
    assert foo_called == 0

    # interrupt-update-resume flow, creating new Send in update call

    # TODO add here test with invoke(Command())


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_checkpoint_three(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    adder = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        Channel.subscribe_to(["input"]).join(["total"])
        | adder
        | Channel.write_to("output", "total")
        | raise_if_above_10
    )

    app = Pregel(
        nodes={"one": one},
        channels={
            "total": BinaryOperatorAggregate(int, operator.add),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
        checkpointer=checkpointer,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, thread_1, debug=1) == 2
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert state.next == ()
    assert (
        state.config["configurable"]["checkpoint_id"]
        == checkpointer.get(thread_1)["id"]
    )
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, thread_1) == 5
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == checkpointer.get(thread_1)["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, thread_1)
    # checkpoint is updated with new input
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert app.invoke(2, thread_1) == 9
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, thread_2, debug=True) == 5
    state = app.get_state({"configurable": {"thread_id": "1"}})
    assert state is not None
    assert state.values.get("total") == 16
    assert state.next == (), "checkpoint of other thread not touched"
    state = app.get_state(thread_2)
    assert state is not None
    assert state.values.get("total") == 5
    assert state.next == ()

    assert len(list(app.get_state_history(thread_1, limit=1))) == 1
    # list all checkpoints for thread 1
    thread_1_history = [c for c in app.get_state_history(thread_1)]
    # there are 7 checkpoints
    assert len(thread_1_history) == 7
    assert Counter(c.metadata["source"] for c in thread_1_history) == {
        "input": 4,
        "loop": 3,
    }
    # sorted descending
    assert (
        thread_1_history[0].config["configurable"]["checkpoint_id"]
        > thread_1_history[1].config["configurable"]["checkpoint_id"]
    )
    # cursor pagination
    cursored = list(
        app.get_state_history(thread_1, limit=1, before=thread_1_history[0].config)
    )
    assert len(cursored) == 1
    assert cursored[0].config == thread_1_history[1].config
    # the last checkpoint
    assert thread_1_history[0].values["total"] == 16
    # the first "loop" checkpoint
    assert thread_1_history[-2].values["total"] == 2
    # can get each checkpoint using aget with config
    assert (
        checkpointer.get(thread_1_history[0].config)["id"]
        == thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    assert (
        checkpointer.get(thread_1_history[1].config)["id"]
        == thread_1_history[1].config["configurable"]["checkpoint_id"]
    )

    thread_1_next_config = app.update_state(thread_1_history[1].config, 10)
    # update creates a new checkpoint
    assert (
        thread_1_next_config["configurable"]["checkpoint_id"]
        > thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    # update makes new checkpoint child of the previous one
    assert (
        app.get_state(thread_1_next_config).parent_config == thread_1_history[1].config
    )
    # 1 more checkpoint in history
    assert len(list(app.get_state_history(thread_1))) == 8
    assert Counter(c.metadata["source"] for c in app.get_state_history(thread_1)) == {
        "update": 1,
        "input": 4,
        "loop": 3,
    }
    # the latest checkpoint is the updated one
    assert app.get_state(thread_1) == app.get_state(thread_1_next_config)


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_invoke_join_then_call_other_pregel(
    mocker: MockerFixture, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": Channel.subscribe_to("input") | add_one | Channel.write_to("output")
        },
        channels={
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    one = (
        Channel.subscribe_to("input")
        | add_10_each
        | Channel.write_to("inbox_one").map()
    )
    two = (
        Channel.subscribe_to("inbox_one")
        | inner_app.map()
        | sorted
        | Channel.write_to("outbox_one")
    )
    chain_three = Channel.subscribe_to("outbox_one") | sum | Channel.write_to("output")

    app = Pregel(
        nodes={
            "one": one,
            "two": two,
            "chain_three": chain_three,
        },
        channels={
            "inbox_one": Topic(int),
            "outbox_one": LastValue(int),
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    for _ in range(10):
        assert app.invoke([2, 3]) == 27

    with ThreadPoolExecutor() as executor:
        assert [*executor.map(app.invoke, [[2, 3]] * 10)] == [27] * 10

    # add checkpointer
    app.checkpointer = checkpointer
    # subgraph is called twice in the same node, through .map(), so raises
    with pytest.raises(MultipleSubgraphsError):
        app.invoke([2, 3], {"configurable": {"thread_id": "1"}})

    # set inner graph checkpointer NeverCheckpoint
    inner_app.checkpointer = False
    # subgraph still called twice, but checkpointing for inner graph is disabled
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_conditional_graph(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools import tool

    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    # Assemble the tools
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    # Construct the agent
    prompt = PromptTemplate.from_template("Hello!")

    llm = FakeStreamingListLLM(
        responses=[
            "tool:search_api:query",
            "tool:search_api:another",
            "finish:answer",
        ]
    )

    def agent_parser(input: str) -> Union[AgentAction, AgentFinish]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return AgentFinish(return_values={"answer": answer}, log=input)
        else:
            _, tool_name, tool_input = input.split(":")
            return AgentAction(tool=tool_name, tool_input=tool_input, log=input)

    agent = RunnablePassthrough.assign(agent_outcome=prompt | llm | agent_parser)

    # Define tool execution logic
    def execute_tools(data: dict) -> dict:
        data = data.copy()
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        if data.get("intermediate_steps") is None:
            data["intermediate_steps"] = []
        else:
            data["intermediate_steps"] = data["intermediate_steps"].copy()
        data["intermediate_steps"].append([agent_action, observation])
        return data

    # Define decision-making logic
    def should_continue(data: dict) -> str:
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = Graph()

    workflow.add_node("agent", agent)
    workflow.add_node(
        "tools",
        execute_tools,
        metadata={"parents": {}, "version": 2, "variant": "b"},
    )

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_graph().draw_mermaid() == snapshot
        assert json.dumps(app.get_graph(xray=True).to_json(), indent=2) == snapshot
        assert app.get_graph(xray=True).draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"input": "what is weather in sf"}) == {
        "input": "what is weather in sf",
        "intermediate_steps": [
            [
                AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:query",
                ),
                "result for query",
            ],
            [
                AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
                "result for another",
            ],
        ],
        "agent_outcome": AgentFinish(
            return_values={"answer": "answer"}, log="finish:answer"
        ),
    }

    assert [c for c in app.stream({"input": "what is weather in sf"})] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    if SHOULD_CHECK_SNAPSHOTS:
        assert app_w_interrupt.get_graph().to_json() == snapshot
        assert app_w_interrupt.get_graph().draw_mermaid() == snapshot

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                },
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )
    assert (
        app_w_interrupt.checkpointer.get_tuple(config).config["configurable"][
            "checkpoint_id"
        ]
        is not None
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "input": "what is weather in sf",
                },
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    app_w_interrupt.update_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "input": "what is weather in sf",
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "input": "what is weather in sf",
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                ),
                "input": "what is weather in sf",
            },
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    app_w_interrupt.update_state(
        config,
        {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                ),
            },
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {
                "agent": {
                    "input": "what is weather in sf",
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    ),
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test re-invoke to continue with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "3"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        }
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {
                "agent": {
                    "agent": {
                        "input": "what is weather in sf",
                        "agent_outcome": AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                    }
                }
            },
            "thread_id": "3",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            },
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {
            "tools": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
            }
        },
        {
            "agent": {
                "input": "what is weather in sf",
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ],
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="another",
                            log="tool:search_api:another",
                        ),
                        "result for another",
                    ],
                ],
                "agent_outcome": AgentFinish(
                    return_values={"answer": "answer"}, log="finish:answer"
                ),
            }
        },
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_conditional_state_graph(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from langchain_core.language_models.fake import FakeStreamingListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import tool

    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )
    setup = mocker.Mock()
    teardown = mocker.Mock()

    @contextmanager
    def assert_ctx_once() -> Iterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @contextmanager
    def make_httpx_client() -> Iterator[httpx.Client]:
        setup()
        with httpx.Client() as client:
            try:
                yield client
            finally:
                teardown()

    class AgentState(TypedDict, total=False):
        input: Annotated[str, UntrackedValue]
        agent_outcome: Optional[Union[AgentAction, AgentFinish]]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        session: Annotated[httpx.Client, Context(make_httpx_client)]

    class ToolState(TypedDict, total=False):
        agent_outcome: Union[AgentAction, AgentFinish]
        session: Annotated[httpx.Client, Context(make_httpx_client)]

    # Assemble the tools
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    # Construct the agent
    prompt = PromptTemplate.from_template("Hello!")

    llm = FakeStreamingListLLM(
        responses=[
            "tool:search_api:query",
            "tool:search_api:another",
            "finish:answer",
        ]
    )

    def agent_parser(input: str) -> dict[str, Union[AgentAction, AgentFinish]]:
        if input.startswith("finish"):
            _, answer = input.split(":")
            return {
                "agent_outcome": AgentFinish(
                    return_values={"answer": answer}, log=input
                )
            }
        else:
            _, tool_name, tool_input = input.split(":")
            return {
                "agent_outcome": AgentAction(
                    tool=tool_name, tool_input=tool_input, log=input
                )
            }

    agent = prompt | llm | agent_parser

    # Define tool execution logic
    def execute_tools(data: ToolState) -> dict:
        # check session in data
        assert isinstance(data["session"], httpx.Client)
        assert "input" not in data
        assert "intermediate_steps" not in data
        # execute the tool
        agent_action: AgentAction = data.pop("agent_outcome")
        observation = {t.name: t for t in tools}[agent_action.tool].invoke(
            agent_action.tool_input
        )
        return {"intermediate_steps": [[agent_action, observation]]}

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        # check session in data
        assert isinstance(data["session"], httpx.Client)
        # Logic to decide whether to continue in the loop or exit
        if isinstance(data["agent_outcome"], AgentFinish):
            return "exit"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", execute_tools, input=ToolState)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )

    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    with assert_ctx_once():
        assert app.invoke({"input": "what is weather in sf"}) == {
            "input": "what is weather in sf",
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ],
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                    "result for another",
                ],
            ],
            "agent_outcome": AgentFinish(
                return_values={"answer": "answer"}, log="finish:answer"
            ),
        }

    with assert_ctx_once():
        assert [*app.stream({"input": "what is weather in sf"})] == [
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        ]
                    ],
                }
            },
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                }
            },
            {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="another",
                                log="tool:search_api:another",
                            ),
                            "result for another",
                        ],
                    ],
                }
            },
            {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "answer"}, log="finish:answer"
                    ),
                }
            },
        ]

    # test state get/update methods with interrupt_after

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    with assert_ctx_once():
        assert [
            c
            for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
        ] == [
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            {"__interrupt__": ()},
        ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    with assert_ctx_once():
        app_w_interrupt.update_state(
            config,
            {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="query",
                    log="tool:search_api:a different query",
                )
            },
        )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    )
                },
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:a different query",
                            ),
                            "result for query",
                        ]
                    ],
                }
            },
            {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="another",
                        log="tool:search_api:another",
                    ),
                }
            },
            {"__interrupt__": ()},
        ]

    with assert_ctx_once():
        app_w_interrupt.update_state(
            config,
            {
                "agent_outcome": AgentFinish(
                    return_values={"answer": "a really nice answer"},
                    log="finish:a really nice answer",
                )
            },
        )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    )
                }
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test state get/update methods with interrupt_before

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
        debug=True,
    )
    config = {"configurable": {"thread_id": "2"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api",
                tool_input="query",
                log="tool:search_api:a different query",
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    )
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:a different query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    app_w_interrupt.update_state(
        config,
        {
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            )
        },
    )

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentFinish(
                return_values={"answer": "a really nice answer"},
                log="finish:a really nice answer",
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:a different query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {
                "agent": {
                    "agent_outcome": AgentFinish(
                        return_values={"answer": "a really nice answer"},
                        log="finish:a really nice answer",
                    )
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # test w interrupt before all
    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before="*",
        debug=True,
    )
    config = {"configurable": {"thread_id": "3"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "3",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            "thread_id": "3",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        ]
                    ],
                }
            },
            "thread_id": "3",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    # test w interrupt after all
    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after="*",
    )
    config = {"configurable": {"thread_id": "4"}}
    llm.i = 0  # reset the llm

    assert [
        c for c in app_w_interrupt.stream({"input": "what is weather in sf"}, config)
    ] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api", tool_input="query", log="tool:search_api:query"
                ),
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [],
        },
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": {
                    "agent_outcome": AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                }
            },
            "thread_id": "4",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "intermediate_steps": [
                    [
                        AgentAction(
                            tool="search_api",
                            tool_input="query",
                            log="tool:search_api:query",
                        ),
                        "result for query",
                    ]
                ],
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "agent_outcome": AgentAction(
                tool="search_api", tool_input="query", log="tool:search_api:query"
            ),
            "intermediate_steps": [
                [
                    AgentAction(
                        tool="search_api",
                        tool_input="query",
                        log="tool:search_api:query",
                    ),
                    "result for query",
                ]
            ],
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "intermediate_steps": [
                        [
                            AgentAction(
                                tool="search_api",
                                tool_input="query",
                                log="tool:search_api:query",
                            ),
                            "result for query",
                        ]
                    ],
                }
            },
            "thread_id": "4",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "agent": {
                "agent_outcome": AgentAction(
                    tool="search_api",
                    tool_input="another",
                    log="tool:search_api:another",
                ),
            }
        },
        {"__interrupt__": ()},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_state_graph_packets(
    request: pytest.FixtureRequest, checkpointer_name: str, mocker: MockerFixture
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolCall,
        ToolMessage,
    )
    from langchain_core.tools import tool

    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        session: Annotated[httpx.Client, Context(httpx.Client)]

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]
    tools_by_name = {t.name: t for t in tools}

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            AIMessage(id="ai3", content="answer"),
        ]
    )

    def agent(data: AgentState) -> AgentState:
        assert isinstance(data["session"], httpx.Client)
        return {
            "messages": model.invoke(data["messages"]),
            "something_extra": "hi there",
        }

    # Define decision-making logic
    def should_continue(data: AgentState) -> str:
        assert isinstance(data["session"], httpx.Client)
        assert (
            data["something_extra"] == "hi there"
        ), "nodes can pass extra data to their cond edges, which isn't saved in state"
        # Logic to decide whether to continue in the loop or exit
        if tool_calls := data["messages"][-1].tool_calls:
            return [Send("tools", tool_call) for tool_call in tool_calls]
        else:
            return END

    def tools_node(input: ToolCall, config: RunnableConfig) -> AgentState:
        time.sleep(input["args"].get("idx", 0) / 10)
        output = tools_by_name[input["name"]].invoke(input["args"], config)
        return {
            "messages": ToolMessage(
                content=output, name=input["name"], tool_call_id=input["id"]
            )
        }

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tools_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges("agent", should_continue)

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert app.invoke({"messages": HumanMessage(content="what is weather in sf")}) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[
                    {
                        "id": "tool_call234",
                        "name": "search_api",
                        "args": {"query": "another", "idx": 0},
                    },
                    {
                        "id": "tool_call567",
                        "name": "search_api",
                        "args": {"query": "a third one", "idx": 1},
                    },
                ],
            ),
            _AnyIdToolMessage(
                content="result for another",
                name="search_api",
                tool_call_id="tool_call234",
            ),
            _AnyIdToolMessage(
                content="result for a third one",
                name="search_api",
                tool_call_id="tool_call567",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    }

    assert [
        c
        for c in app.stream(
            {"messages": [HumanMessage(content="what is weather in sf")]}
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            },
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            }
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call234",
                )
            },
        },
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a third one",
                    name="search_api",
                    tool_call_id="tool_call567",
                ),
            },
        },
        {"agent": {"messages": AIMessage(content="answer", id="ai3")}},
    ]

    # interrupt after agent

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            {"messages": HumanMessage(content="what is weather in sf")}, config
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            }
        },
        {"__interrupt__": ()},
    ]

    if not FF_SEND_V2:
        return

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        content="",
                        additional_kwargs={},
                        response_metadata={},
                        id="ai1",
                        tool_calls=[
                            {
                                "name": "search_api",
                                "args": {"query": "query"},
                                "id": "tool_call123",
                                "type": "tool_call",
                            }
                        ],
                    )
                },
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 2, AnyStr())
            ),
        ),
        next=("tools",),
        config=(app_w_interrupt.checkpointer.get_tuple(config)).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = (app_w_interrupt.get_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    app_w_interrupt.update_state(
        config, {"messages": last_message, "something_extra": "hi there"}
    )

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, (), 0, AnyStr())),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "messages": AIMessage(
                        id="ai1",
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "a different query"},
                            },
                        ],
                    ),
                    "something_extra": "hi there",
                }
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            },
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        "",
                        id="ai2",
                        tool_calls=[
                            {
                                "name": "search_api",
                                "args": {"query": "another", "idx": 0},
                                "id": "tool_call234",
                                "type": "tool_call",
                            },
                            {
                                "name": "search_api",
                                "args": {"query": "a third one", "idx": 1},
                                "id": "tool_call567",
                                "type": "tool_call",
                            },
                        ],
                    )
                },
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 2, AnyStr())
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 3, AnyStr())
            ),
        ),
        next=("tools", "tools"),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "messages": _AnyIdToolMessage(
                        content="result for a different query",
                        name="search_api",
                        tool_call_id="tool_call123",
                    ),
                },
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "messages": AIMessage(content="answer", id="ai2"),
            "something_extra": "hi there",
        },
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 3,
            "writes": {
                "agent": {
                    "messages": AIMessage(content="answer", id="ai2"),
                    "something_extra": "hi there",
                }
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # interrupt before tools

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0

    assert [
        c
        for c in app_w_interrupt.stream(
            {"messages": HumanMessage(content="what is weather in sf")}, config
        )
    ] == [
        {
            "agent": {
                "messages": AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                )
            }
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        "",
                        id="ai1",
                        tool_calls=[
                            {
                                "name": "search_api",
                                "args": {"query": "query"},
                                "id": "tool_call123",
                                "type": "tool_call",
                            }
                        ],
                    )
                },
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 2, AnyStr())
            ),
        ),
        next=("tools",),
        config=(app_w_interrupt.checkpointer.get_tuple(config)).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = (app_w_interrupt.get_state(config)).values["messages"][-1]
    last_message.tool_calls[0]["args"]["query"] = "a different query"
    app_w_interrupt.update_state(
        config, {"messages": last_message, "something_extra": "hi there"}
    )

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
            ]
        },
        tasks=(PregelTask(AnyStr(), "tools", (PUSH, (), 0, AnyStr())),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {
                "agent": {
                    "messages": AIMessage(
                        id="ai1",
                        content="",
                        tool_calls=[
                            {
                                "id": "tool_call123",
                                "name": "search_api",
                                "args": {"query": "a different query"},
                            },
                        ],
                    ),
                    "something_extra": "hi there",
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": {
                "messages": _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            }
        },
        {
            "agent": {
                "messages": AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                )
            },
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(
                    id="ai2",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call234",
                            "name": "search_api",
                            "args": {"query": "another", "idx": 0},
                        },
                        {
                            "id": "tool_call567",
                            "name": "search_api",
                            "args": {"query": "a third one", "idx": 1},
                        },
                    ],
                ),
            ]
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="agent",
                path=("__pregel_pull", "agent"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "messages": AIMessage(
                        "",
                        id="ai2",
                        tool_calls=[
                            {
                                "name": "search_api",
                                "args": {"query": "another", "idx": 0},
                                "id": "tool_call234",
                                "type": "tool_call",
                            },
                            {
                                "name": "search_api",
                                "args": {"query": "a third one", "idx": 1},
                                "id": "tool_call567",
                                "type": "tool_call",
                            },
                        ],
                    )
                },
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 2, AnyStr())
            ),
            PregelTask(
                AnyStr(), "tools", (PUSH, ("__pregel_pull", "agent"), 3, AnyStr())
            ),
        ),
        next=("tools", "tools"),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {
                "tools": {
                    "messages": _AnyIdToolMessage(
                        content="result for a different query",
                        name="search_api",
                        tool_call_id="tool_call123",
                    ),
                },
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        {
            "messages": AIMessage(content="answer", id="ai2"),
            "something_extra": "hi there",
        },
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    id="ai1",
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        },
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
            ]
        },
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=(app_w_interrupt.checkpointer.get_tuple(config)).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 3,
            "writes": {
                "agent": {
                    "messages": AIMessage(content="answer", id="ai2"),
                    "something_extra": "hi there",
                }
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_message_graph(
    snapshot: SnapshotAssertion,
    deterministic_uuids: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Define a new graph
    workflow = MessageGraph()

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert json.dumps(app.get_input_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_output_schema().model_json_schema()) == snapshot
        assert json.dumps(app.get_graph().to_json(), indent=2) == snapshot
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
        _AnyIdHumanMessage(
            content="what is weather in sf",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "search_api",
                    "args": {"query": "query"},
                }
            ],
            id="ai1",  # respects ids passed in
        ),
        _AnyIdToolMessage(
            content="result for query",
            name="search_api",
            tool_call_id="tool_call123",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call456",
                    "name": "search_api",
                    "args": {"query": "another"},
                }
            ],
            id="ai2",
        ),
        _AnyIdToolMessage(
            content="result for another",
            name="search_api",
            tool_call_id="tool_call456",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream(("human", "what is weather in sf"), config)
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [c for c in app_w_interrupt.stream("what is weather in sf", config)] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 6,
            "writes": {"tools": UnsortedSequence("ai", "an extra message")},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_root_graph(
    deterministic_uuids: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from copy import deepcopy

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool

    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    class FakeFuntionChatModel(FakeMessagesListChatModel):
        def bind_functions(self, functions: list):
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            response = deepcopy(self.responses[self.i])
            if self.i < len(self.responses) - 1:
                self.i += 1
            else:
                self.i = 0
            generation = ChatGeneration(message=response)
            return ChatResult(generations=[generation])

    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    model = FakeFuntionChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
            AIMessage(content="answer", id="ai3"),
        ]
    )

    # Define the function that determines whether to continue or not
    def should_continue(messages):
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    class State(TypedDict):
        __root__: Annotated[list[BaseMessage], add_messages]

    # Define a new graph
    workflow = StateGraph(State)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    assert app.invoke(HumanMessage(content="what is weather in sf")) == [
        _AnyIdHumanMessage(
            content="what is weather in sf",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call123",
                    "name": "search_api",
                    "args": {"query": "query"},
                }
            ],
            id="ai1",  # respects ids passed in
        ),
        _AnyIdToolMessage(
            content="result for query",
            name="search_api",
            tool_call_id="tool_call123",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tool_call456",
                    "name": "search_api",
                    "args": {"query": "another"},
                }
            ],
            id="ai2",
        ),
        _AnyIdToolMessage(
            content="result for another",
            name="search_api",
            tool_call_id="tool_call456",
        ),
        AIMessage(content="answer", id="ai3"),
    ]

    assert [*app.stream([HumanMessage(content="what is weather in sf")])] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for query",
                    name="search_api",
                    tool_call_id="tool_call123",
                    id="00000000-0000-4000-8000-000000000033",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {
            "tools": [
                ToolMessage(
                    content="result for another",
                    name="search_api",
                    tool_call_id="tool_call456",
                    id="00000000-0000-4000-8000-000000000041",
                )
            ]
        },
        {"agent": AIMessage(content="answer", id="ai3")},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["agent"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream(("human", "what is weather in sf"), config)
    ] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    next_config = app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=next_config,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),  # replace existing message
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
            "thread_id": "1",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "2"}}
    model.i = 0  # reset the llm

    assert [c for c in app_w_interrupt.stream("what is weather in sf", config)] == [
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    }
                ],
                id="ai1",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # modify ai message
    last_message = app_w_interrupt.get_state(config).values[-1]
    last_message.tool_calls[0]["args"] = {"query": "a different query"}
    app_w_interrupt.update_state(config, last_message)

    # message was replaced instead of appended
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 2,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                    id="ai1",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {
            "tools": [
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                )
            ]
        },
        {
            "agent": AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            )
        },
        {"__interrupt__": ()},
    ]

    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call456",
                        "name": "search_api",
                        "args": {"query": "another"},
                    }
                ],
                id="ai2",
            ),
        ],
        tasks=(PregelTask(AnyStr(), "tools", (PULL, "tools")),),
        next=("tools",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 4,
            "writes": {
                "agent": AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tool_call456",
                            "name": "search_api",
                            "args": {"query": "another"},
                        }
                    ],
                    id="ai2",
                )
            },
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    app_w_interrupt.update_state(
        config,
        AIMessage(content="answer", id="ai2"),
    )

    # replaces message even if object identity is different, as long as id is the same
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
        ],
        tasks=(),
        next=(),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 5,
            "writes": {"agent": AIMessage(content="answer", id="ai2")},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # add an extra message as if it came from "tools" node
    app_w_interrupt.update_state(config, ("ai", "an extra message"), as_node="tools")

    # extra message is coerced BaseMessge and appended
    # now the next node is "agent" per the graph edges
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values=[
            _AnyIdHumanMessage(content="what is weather in sf"),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "a different query"},
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
                id=AnyStr(),
            ),
            AIMessage(content="answer", id="ai2"),
            _AnyIdAIMessage(content="an extra message"),
        ],
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 6,
            "writes": {"tools": UnsortedSequence("ai", "an extra message")},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # create new graph with one more state key, reuse previous thread history

    def simple_add(left, right):
        if not isinstance(right, list):
            right = [right]
        return left + right

    class MoreState(TypedDict):
        __root__: Annotated[list[BaseMessage], simple_add]
        something_else: str

    # Define a new graph
    new_workflow = StateGraph(MoreState)
    new_workflow.add_node(
        "agent", RunnableMap(__root__=RunnablePick("__root__") | model)
    )
    new_workflow.add_node(
        "tools", RunnableMap(__root__=RunnablePick("__root__") | ToolNode(tools))
    )
    new_workflow.set_entry_point("agent")
    new_workflow.add_conditional_edges(
        "agent",
        RunnablePick("__root__") | should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )
    new_workflow.add_edge("tools", "agent")
    new_app = new_workflow.compile(checkpointer=checkpointer)
    model.i = 0  # reset the llm

    # previous state is converted to new schema
    assert new_app.get_state(config) == StateSnapshot(
        values={
            "__root__": [
                _AnyIdHumanMessage(content="what is weather in sf"),
                AIMessage(
                    content="",
                    id="ai1",
                    tool_calls=[
                        {
                            "id": "tool_call123",
                            "name": "search_api",
                            "args": {"query": "a different query"},
                        }
                    ],
                ),
                _AnyIdToolMessage(
                    content="result for a different query",
                    name="search_api",
                    tool_call_id="tool_call123",
                ),
                AIMessage(content="answer", id="ai2"),
                _AnyIdAIMessage(content="an extra message"),
            ]
        },
        tasks=(PregelTask(AnyStr(), "agent", (PULL, "agent")),),
        next=("agent",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 6,
            "writes": {"tools": UnsortedSequence("ai", "an extra message")},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    # new input is merged to old state
    assert new_app.invoke(
        {
            "__root__": [HumanMessage(content="what is weather in la")],
            "something_else": "value",
        },
        config,
        interrupt_before=["agent"],
    ) == {
        "__root__": [
            HumanMessage(
                content="what is weather in sf",
                id="00000000-0000-4000-8000-000000000070",
            ),
            AIMessage(
                content="",
                id="ai1",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "a different query"},
                        "id": "tool_call123",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for a different query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            AIMessage(content="answer", id="ai2"),
            AIMessage(
                content="an extra message", id="00000000-0000-4000-8000-000000000092"
            ),
            HumanMessage(content="what is weather in la"),
        ],
        "something_else": "value",
    }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_dynamic_interrupt(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_node_count = 0

    def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry=RetryPolicy())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    resumable=True,
                    ns=[AnyStr("tool_two:")],
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_two": {"my_key": " my answer"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
        "my_key": "value ⛰️",
        "market": "DE",
    }
    assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
            "thread_id": "1",
        },
    ]
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:")],
                    ),
                ),
            ),
        ),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )
    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None, as_node=END)
    # interrupt and next tasks are cleared
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=(),
        tasks=(),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {},
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )


@pytest.mark.skipif(not FF_SEND_V2, reason="send v2 is not enabled")
@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_copy_checkpoint(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    def tool_one(s: State) -> State:
        return {"my_key": " one"}

    tool_two_node_count = 0

    def tool_two_node(s: State) -> State:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    def start(state: State) -> list[Union[Send, str]]:
        return ["tool_two", Send("tool_one", state)]

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", tool_two_node, retry=RetryPolicy())
    tool_two_graph.add_node("tool_one", tool_one)
    tool_two_graph.set_conditional_entry_point(start)
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value one",
        "market": "DE",
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value one"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value one all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "tool_one": {"my_key": " one"},
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    resumable=True,
                    ns=[AnyStr("tool_two:")],
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_two": {"my_key": " my answer"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
        "my_key": "value ⛰️ one",
        "market": "DE",
    }
    assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {"tool_one": {"my_key": " one"}},
            "thread_id": "1",
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
            "thread_id": "1",
        },
    ]
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:")],
                    ),
                ),
            ),
        ),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": {"tool_one": {"my_key": " one"}},
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )
    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None)
    # interrupt is cleared, next task is kept
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ one", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(),
            ),
        ),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {},
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_dynamic_interrupt_subgraph(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class SubgraphState(TypedDict):
        my_key: str
        market: str

    tool_two_node_count = 0

    def tool_two_node(s: SubgraphState) -> SubgraphState:
        nonlocal tool_two_node_count
        tool_two_node_count += 1
        if s["market"] == "DE":
            answer = interrupt("Just because...")
        else:
            answer = " all good"
        return {"my_key": answer}

    subgraph = StateGraph(SubgraphState)
    subgraph.add_node("do", tool_two_node, retry=RetryPolicy())
    subgraph.add_edge(START, "do")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two", subgraph.compile())
    tool_two_graph.add_edge(START, "tool_two")
    tool_two = tool_two_graph.compile()

    tracer = FakeTracer()
    assert tool_two.invoke(
        {"my_key": "value", "market": "DE"}, {"callbacks": [tracer]}
    ) == {
        "my_key": "value",
        "market": "DE",
    }
    assert tool_two_node_count == 1, "interrupts aren't retried"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.end_time is not None
    assert run.error is None
    assert run.outputs == {"market": "DE", "my_key": "value"}

    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value all good",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(checkpointer=checkpointer)

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    # flow: interrupt -> resume with answer
    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert [
        c for c in tool_two.stream({"my_key": "value ⛰️", "market": "DE"}, thread2)
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="Just because...",
                    resumable=True,
                    ns=[AnyStr("tool_two:"), AnyStr("do:")],
                ),
            )
        },
    ]
    # resume with answer
    assert [c for c in tool_two.stream(Command(resume=" my answer"), thread2)] == [
        {"tool_two": {"my_key": " my answer", "market": "DE"}},
    ]

    # flow: interrupt -> clear tasks
    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
        "my_key": "value ⛰️",
        "market": "DE",
    }
    assert [
        c.metadata
        for c in tool_two.checkpointer.list(
            {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
        )
    ] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
            "thread_id": "1",
        },
    ]
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=("tool_two",),
        tasks=(
            PregelTask(
                AnyStr(),
                "tool_two",
                (PULL, "tool_two"),
                interrupts=(
                    Interrupt(
                        value="Just because...",
                        resumable=True,
                        ns=[AnyStr("tool_two:"), AnyStr("do:")],
                    ),
                ),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("tool_two:"),
                    }
                },
            ),
        ),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "thread_id": "1",
        },
        parent_config=[
            *tool_two.checkpointer.list(
                {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}, limit=2
            )
        ][-1].config,
    )
    # clear the interrupt and next tasks
    tool_two.update_state(thread1, None, as_node=END)
    # interrupt and next tasks are cleared
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        next=(),
        tasks=(),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {},
            "thread_id": "1",
        },
        parent_config=[
            *tool_two.checkpointer.list(
                {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}, limit=2
            )
        ][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_start_branch_then(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str
        shared: Annotated[dict[str, dict[str, Any]], SharedValue.on("assistant_id")]

    def assert_shared_value(data: State, config: RunnableConfig) -> State:
        assert "shared" in data
        if thread_id := config["configurable"].get("thread_id"):
            if thread_id == "1":
                # this is the first thread, so should not see a value
                assert data["shared"] == {}
                return {"shared": {"1": {"hello": "world"}}}
            elif thread_id == "2":
                # this should get value saved by thread 1
                assert data["shared"] == {"1": {"hello": "world"}}
            elif thread_id == "3":
                # this is a different assistant, so should not see previous value
                assert data["shared"] == {}
        return {}

    def tool_two_slow(data: State, config: RunnableConfig) -> State:
        return {"my_key": " slow", **assert_shared_value(data, config)}

    def tool_two_fast(data: State, config: RunnableConfig) -> State:
        return {"my_key": " fast", **assert_shared_value(data, config)}

    tool_two_graph = StateGraph(State)
    tool_two_graph.add_node("tool_two_slow", tool_two_slow)
    tool_two_graph.add_node("tool_two_fast", tool_two_fast)
    tool_two_graph.set_conditional_entry_point(
        lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast", then=END
    )
    tool_two = tool_two_graph.compile()
    assert tool_two.get_graph().draw_mermaid() == snapshot

    assert tool_two.invoke({"my_key": "value", "market": "DE"}) == {
        "my_key": "value slow",
        "market": "DE",
    }
    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value fast",
        "market": "US",
    }

    tool_two = tool_two_graph.compile(
        store=InMemoryStore(),
        checkpointer=checkpointer,
        interrupt_before=["tool_two_fast", "tool_two_slow"],
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "1", "assistant_id": "a"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value ⛰️", "market": "DE"}, thread1) == {
        "my_key": "value ⛰️",
        "market": "DE",
    }
    assert [c.metadata for c in tool_two.checkpointer.list(thread1)] == [
        {
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "assistant_id": "a",
            "thread_id": "1",
        },
        {
            "parents": {},
            "source": "input",
            "step": -1,
            "writes": {"__start__": {"my_key": "value ⛰️", "market": "DE"}},
            "assistant_id": "a",
            "thread_id": "1",
        },
    ]
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
        next=("tool_two_slow",),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "assistant_id": "a",
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread1, debug=1) == {
        "my_key": "value ⛰️ slow",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value ⛰️ slow", "market": "DE"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"tool_two_slow": {"my_key": " slow"}},
            "assistant_id": "a",
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )

    thread2 = {"configurable": {"thread_id": "2", "assistant_id": "a"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
        next=("tool_two_fast",),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "assistant_id": "a",
            "thread_id": "2",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread2, debug=1) == {
        "my_key": "value fast",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value fast", "market": "US"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"tool_two_fast": {"my_key": " fast"}},
            "assistant_id": "a",
            "thread_id": "2",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )

    thread3 = {"configurable": {"thread_id": "3", "assistant_id": "b"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "US"}, thread3) == {
        "my_key": "value",
        "market": "US",
    }
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "value", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
        next=("tool_two_fast",),
        config=tool_two.checkpointer.get_tuple(thread3).config,
        created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 0,
            "writes": None,
            "assistant_id": "b",
            "thread_id": "3",
        },
        parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
    )
    # update state
    tool_two.update_state(thread3, {"my_key": "key"})  # appends to my_key
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "valuekey", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
        next=("tool_two_fast",),
        config=tool_two.checkpointer.get_tuple(thread3).config,
        created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 1,
            "writes": {START: {"my_key": "key"}},
            "assistant_id": "b",
            "thread_id": "3",
        },
        parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread3, debug=1) == {
        "my_key": "valuekey fast",
        "market": "US",
    }
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "valuekey fast", "market": "US"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread3).config,
        created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {"tool_two_fast": {"my_key": " fast"}},
            "assistant_id": "b",
            "thread_id": "3",
        },
        parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_branch_then(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]
        market: str

    tool_two_graph = StateGraph(State)
    tool_two_graph.set_entry_point("prepare")
    tool_two_graph.set_finish_point("finish")
    tool_two_graph.add_conditional_edges(
        source="prepare",
        path=lambda s: "tool_two_slow" if s["market"] == "DE" else "tool_two_fast",
        then="finish",
    )
    tool_two_graph.add_node("prepare", lambda s: {"my_key": " prepared"})
    tool_two_graph.add_node("tool_two_slow", lambda s: {"my_key": " slow"})
    tool_two_graph.add_node("tool_two_fast", lambda s: {"my_key": " fast"})
    tool_two_graph.add_node("finish", lambda s: {"my_key": " finished"})
    tool_two = tool_two_graph.compile()
    assert tool_two.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert tool_two.get_graph().draw_mermaid() == snapshot

    assert tool_two.invoke({"my_key": "value", "market": "DE"}, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert tool_two.invoke({"my_key": "value", "market": "US"}) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }

    # test stream_mode=debug
    tool_two = tool_two_graph.compile(checkpointer=checkpointer)
    thread10 = {"configurable": {"thread_id": "10"}}

    res = [
        *tool_two.stream(
            {"my_key": "value", "market": "DE"}, thread10, stream_mode="debug"
        )
    ]

    assert res == [
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": -1,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {"my_key": ""},
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                    "writes": {"__start__": {"my_key": "value", "market": "DE"}},
                    "thread_id": "10",
                },
                "parent_config": None,
                "next": ["__start__"],
                "tasks": [
                    {
                        "id": AnyStr(),
                        "name": "__start__",
                        "interrupts": (),
                        "state": None,
                    }
                ],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 0,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                    "writes": None,
                    "thread_id": "10",
                },
                "parent_config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "next": ["prepare"],
                "tasks": [
                    {"id": AnyStr(), "name": "prepare", "interrupts": (), "state": None}
                ],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "prepare",
                "input": {"my_key": "value", "market": "DE"},
                "triggers": ["start:prepare"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "prepare",
                "result": [("my_key", " prepared")],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value prepared",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 1,
                    "writes": {"prepare": {"my_key": " prepared"}},
                    "thread_id": "10",
                },
                "parent_config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "next": ["tool_two_slow"],
                "tasks": [
                    {
                        "id": AnyStr(),
                        "name": "tool_two_slow",
                        "interrupts": (),
                        "state": None,
                    }
                ],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "tool_two_slow",
                "input": {"my_key": "value prepared", "market": "DE"},
                "triggers": ["branch:prepare:condition:tool_two_slow"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "tool_two_slow",
                "result": [("my_key", " slow")],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value prepared slow",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 2,
                    "writes": {"tool_two_slow": {"my_key": " slow"}},
                    "thread_id": "10",
                },
                "parent_config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "next": ["finish"],
                "tasks": [
                    {"id": AnyStr(), "name": "finish", "interrupts": (), "state": None}
                ],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "finish",
                "input": {"my_key": "value prepared slow", "market": "DE"},
                "triggers": ["branch:prepare:condition::then"],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "finish",
                "result": [("my_key", " finished")],
                "error": None,
                "interrupts": [],
            },
        },
        {
            "type": "checkpoint",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "values": {
                    "my_key": "value prepared slow finished",
                    "market": "DE",
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 3,
                    "writes": {"finish": {"my_key": " finished"}},
                    "thread_id": "10",
                },
                "parent_config": {
                    "tags": [],
                    "metadata": {"thread_id": "10"},
                    "callbacks": None,
                    "recursion_limit": 25,
                    "configurable": {
                        "thread_id": "10",
                        "checkpoint_ns": "",
                        "checkpoint_id": AnyStr(),
                    },
                },
                "next": [],
                "tasks": [],
            },
        },
    ]

    tool_two = tool_two_graph.compile(
        checkpointer=checkpointer, interrupt_before=["tool_two_fast", "tool_two_slow"]
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "1"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
        "my_key": "value prepared",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
        next=("tool_two_slow",),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread1, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
            "thread_id": "1",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )

    thread2 = {"configurable": {"thread_id": "2"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value prepared",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
        next=("tool_two_fast",),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
            "thread_id": "2",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread2, debug=1) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared fast finished", "market": "US"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
            "thread_id": "2",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )

    tool_two = tool_two_graph.compile(
        checkpointer=checkpointer, interrupt_before=["finish"]
    )

    thread1 = {"configurable": {"thread_id": "11"}}

    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
        "my_key": "value prepared slow",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={
            "my_key": "value prepared slow",
            "market": "DE",
        },
        tasks=(PregelTask(AnyStr(), "finish", (PULL, "finish")),),
        next=("finish",),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 2,
            "writes": {"tool_two_slow": {"my_key": " slow"}},
            "thread_id": "11",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )

    # update state
    tool_two.update_state(thread1, {"my_key": "er"})
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={
            "my_key": "value prepared slower",
            "market": "DE",
        },
        tasks=(PregelTask(AnyStr(), "finish", (PULL, "finish")),),
        next=("finish",),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 3,
            "writes": {"tool_two_slow": {"my_key": "er"}},
            "thread_id": "11",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )

    tool_two = tool_two_graph.compile(
        checkpointer=checkpointer, interrupt_after=["prepare"]
    )

    # missing thread_id
    with pytest.raises(ValueError, match="thread_id"):
        tool_two.invoke({"my_key": "value", "market": "DE"})

    thread1 = {"configurable": {"thread_id": "21"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "DE"}, thread1) == {
        "my_key": "value prepared",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
        next=("tool_two_slow",),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
            "thread_id": "21",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread1, debug=1) == {
        "my_key": "value prepared slow finished",
        "market": "DE",
    }
    assert tool_two.get_state(thread1) == StateSnapshot(
        values={"my_key": "value prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread1).config,
        created_at=tool_two.checkpointer.get_tuple(thread1).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
            "thread_id": "21",
        },
        parent_config=[*tool_two.checkpointer.list(thread1, limit=2)][-1].config,
    )

    thread2 = {"configurable": {"thread_id": "22"}}
    # stop when about to enter node
    assert tool_two.invoke({"my_key": "value", "market": "US"}, thread2) == {
        "my_key": "value prepared",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared", "market": "US"},
        tasks=(PregelTask(AnyStr(), "tool_two_fast", (PULL, "tool_two_fast")),),
        next=("tool_two_fast",),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
            "thread_id": "22",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread2, debug=1) == {
        "my_key": "value prepared fast finished",
        "market": "US",
    }
    assert tool_two.get_state(thread2) == StateSnapshot(
        values={"my_key": "value prepared fast finished", "market": "US"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread2).config,
        created_at=tool_two.checkpointer.get_tuple(thread2).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
            "thread_id": "22",
        },
        parent_config=[*tool_two.checkpointer.list(thread2, limit=2)][-1].config,
    )

    thread3 = {"configurable": {"thread_id": "23"}}
    # update an empty thread before first run
    uconfig = tool_two.update_state(thread3, {"my_key": "key", "market": "DE"})
    # check current state
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "key", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "prepare", (PULL, "prepare")),),
        next=("prepare",),
        config=uconfig,
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 0,
            "writes": {START: {"my_key": "key", "market": "DE"}},
            "thread_id": "23",
        },
        parent_config=None,
    )
    # run from this point
    assert tool_two.invoke(None, thread3) == {
        "my_key": "key prepared",
        "market": "DE",
    }
    # get state after first node
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "key prepared", "market": "DE"},
        tasks=(PregelTask(AnyStr(), "tool_two_slow", (PULL, "tool_two_slow")),),
        next=("tool_two_slow",),
        config=tool_two.checkpointer.get_tuple(thread3).config,
        created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 1,
            "writes": {"prepare": {"my_key": " prepared"}},
            "thread_id": "23",
        },
        parent_config=uconfig,
    )
    # resume, for same result as above
    assert tool_two.invoke(None, thread3, debug=1) == {
        "my_key": "key prepared slow finished",
        "market": "DE",
    }
    assert tool_two.get_state(thread3) == StateSnapshot(
        values={"my_key": "key prepared slow finished", "market": "DE"},
        tasks=(),
        next=(),
        config=tool_two.checkpointer.get_tuple(thread3).config,
        created_at=tool_two.checkpointer.get_tuple(thread3).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "loop",
            "step": 3,
            "writes": {"finish": {"my_key": " finished"}},
            "thread_id": "23",
        },
        parent_config=[*tool_two.checkpointer.list(thread3, limit=2)][-1].config,
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    workflow = StateGraph(State)

    @workflow.add_node
    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)  # to ensure stream order
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow.add_node(analyzer_one)
    workflow.add_node(retriever_one)
    workflow.add_node(retriever_two)
    workflow.add_node(qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["qa"],
    )
    config = {"configurable": {"thread_id": "2"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    app_w_interrupt.update_state(config, {"docs": ["doc5"]})
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        },
        tasks=(PregelTask(AnyStr(), "qa", (PULL, "qa")),),
        next=("qa",),
        config=app_w_interrupt.checkpointer.get_tuple(config).config,
        created_at=app_w_interrupt.checkpointer.get_tuple(config).checkpoint["ts"],
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
            "writes": {"retriever_one": {"docs": ["doc5"]}},
            "thread_id": "2",
        },
        parent_config=[*app_w_interrupt.checkpointer.list(config, limit=2)][-1].config,
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    snapshot: SnapshotAssertion, request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        return {"query": f'analyzed: {data["query"]}'}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    def rewrite_query_then(data: State) -> Literal["retriever_two"]:
        return "retriever_two"

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges("rewrite_query", rewrite_query_then)
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot

    assert app.invoke({"query": "what is weather in sf"}, debug=True) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic1(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from pydantic.v1 import BaseModel, ValidationError

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    setup = mocker.Mock()
    teardown = mocker.Mock()

    @contextmanager
    def assert_ctx_once() -> Iterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @contextmanager
    def make_httpx_client() -> Iterator[httpx.Client]:
        setup()
        with httpx.Client() as client:
            try:
                yield client
            finally:
                teardown()

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class InnerObject(BaseModel):
        yo: int

    class State(BaseModel):
        class Config:
            arbitrary_types_allowed = True

        query: str
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.Client, Context(make_httpx_client)]

    class Input(BaseModel):
        query: str
        inner: InnerObject

    class Output(BaseModel):
        answer: str
        docs: list[str]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    def analyzer_one(data: State) -> State:
        return StateUpdate(query=f"analyzed: {data.query}")

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State, input=Input, output=Output)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges(
        "rewrite_query", decider, {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
    assert app.get_input_jsonschema() == snapshot
    assert app.get_output_jsonschema() == snapshot

    with pytest.raises(ValidationError), assert_ctx_once():
        app.invoke({"query": {}})

    with assert_ctx_once():
        assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        }

    with assert_ctx_once():
        assert [
            *app.stream({"query": "what is weather in sf", "inner": {"yo": 1}})
        ] == [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    with assert_ctx_once():
        assert [
            c
            for c in app_w_interrupt.stream(
                {"query": "what is weather in sf", "inner": {"yo": 1}}, config
            )
        ] == [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"__interrupt__": ()},
        ]

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    with assert_ctx_once():
        assert app_w_interrupt.update_state(
            config, {"docs": ["doc5"]}, as_node="rewrite_query"
        ) == {
            "configurable": {
                "thread_id": "1",
                "checkpoint_id": AnyStr(),
                "checkpoint_ns": "",
            }
        }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    snapshot: SnapshotAssertion,
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    checkpointer_name: str,
) -> None:
    from pydantic import BaseModel, ConfigDict, ValidationError

    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    setup = mocker.Mock()
    teardown = mocker.Mock()

    @contextmanager
    def assert_ctx_once() -> Iterator[None]:
        assert setup.call_count == 0
        assert teardown.call_count == 0
        try:
            yield
        finally:
            assert setup.call_count == 1
            assert teardown.call_count == 1
            setup.reset_mock()
            teardown.reset_mock()

    @contextmanager
    def make_httpx_client() -> Iterator[httpx.Client]:
        setup()
        with httpx.Client() as client:
            try:
                yield client
            finally:
                teardown()

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class InnerObject(BaseModel):
        yo: int

    class State(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        query: str
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]
        client: Annotated[httpx.Client, Context(make_httpx_client)]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    class Input(BaseModel):
        query: str
        inner: InnerObject

    class Output(BaseModel):
        answer: str
        docs: list[str]

    def rewrite_query(data: State) -> State:
        return {"query": f"query: {data.query}"}

    def analyzer_one(data: State) -> State:
        return StateUpdate(query=f"analyzed: {data.query}")

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.1)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State, input=Input, output=Output)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_conditional_edges(
        "rewrite_query", decider, {"retriever_two": "retriever_two"}
    )
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    if SHOULD_CHECK_SNAPSHOTS:
        assert app.get_graph().draw_mermaid(with_styles=False) == snapshot
        assert app.get_input_schema().model_json_schema() == snapshot
        assert app.get_output_schema().model_json_schema() == snapshot

    with pytest.raises(ValidationError), assert_ctx_once():
        app.invoke({"query": {}})

    with assert_ctx_once():
        assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
            "docs": ["doc1", "doc2", "doc3", "doc4"],
            "answer": "doc1,doc2,doc3,doc4",
        }

    with assert_ctx_once():
        assert [
            *app.stream({"query": "what is weather in sf", "inner": {"yo": 1}})
        ] == [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    with assert_ctx_once():
        assert [
            c
            for c in app_w_interrupt.stream(
                {"query": "what is weather in sf", "inner": {"yo": 1}}, config
            )
        ] == [
            {"rewrite_query": {"query": "query: what is weather in sf"}},
            {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
            {"retriever_two": {"docs": ["doc3", "doc4"]}},
            {"retriever_one": {"docs": ["doc1", "doc2"]}},
            {"__interrupt__": ()},
        ]

    with assert_ctx_once():
        assert [c for c in app_w_interrupt.stream(None, config)] == [
            {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
        ]

    with assert_ctx_once():
        assert app_w_interrupt.update_state(
            config, {"docs": ["doc5"]}, as_node="rewrite_query"
        ) == {
            "configurable": {
                "thread_id": "1",
                "checkpoint_id": AnyStr(),
                "checkpoint_ns": "",
            }
        }


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer: BaseCheckpointSaver = request.getfixturevalue(
        f"checkpointer_{checkpointer_name}"
    )

    def sorted_add(
        x: list[str], y: Union[list[str], list[tuple[str, str]]]
    ) -> list[str]:
        if isinstance(y[0], tuple):
            for rem, _ in y:
                x.remove(rem)
            y = [t[1] for t in y]
        return sorted(operator.add(x, y))

    class State(TypedDict, total=False):
        query: str
        answer: str
        docs: Annotated[list[str], sorted_add]

    def rewrite_query(data: State) -> State:
        return {"query": f'query: {data["query"]}'}

    def analyzer_one(data: State) -> State:
        time.sleep(0.1)
        return {"query": f'analyzed: {data["query"]}'}

    def retriever_one(data: State) -> State:
        return {"docs": ["doc1", "doc2"]}

    def retriever_two(data: State) -> State:
        time.sleep(0.2)
        return {"docs": ["doc3", "doc4"]}

    def qa(data: State) -> State:
        return {"answer": ",".join(data["docs"])}

    workflow = StateGraph(State)

    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("analyzer_one", analyzer_one)
    workflow.add_node("retriever_one", retriever_one)
    workflow.add_node("retriever_two", retriever_two)
    workflow.add_node("qa", qa)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "analyzer_one")
    workflow.add_edge("analyzer_one", "retriever_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    workflow.set_finish_point("qa")

    # silly edge, to make sure having been triggered before doesn't break
    # semantics of named barrier (== waiting edges)
    workflow.add_edge("rewrite_query", "qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"qa": {"answer": ""}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"qa": {"answer": ""}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_stream_subgraphs_during_execution(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        time.sleep(0.5)
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        time.sleep(0.2)
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile())
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge(START, "inner")
    graph.add_edge(START, "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.add_edge("outer_2", END)

    app = graph.compile(checkpointer=checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in app.stream({"my_key": ""}, config, subgraphs=True):
        chunks.append((round(time.perf_counter() - start, 1), c))
    for idx in range(len(chunks)):
        elapsed, c = chunks[idx]
        chunks[idx] = (round(elapsed - chunks[0][0], 1), c)

    assert chunks == [
        # arrives before "inner" finishes
        (
            FloatBetween(0.0, 0.1),
            (
                (AnyStr("inner:"),),
                {"inner_1": {"my_key": "got here", "my_other_key": ""}},
            ),
        ),
        (FloatBetween(0.2, 0.3), ((), {"outer_1": {"my_key": " and parallel"}})),
        (
            FloatBetween(0.5, 0.8),
            (
                (AnyStr("inner:"),),
                {"inner_2": {"my_key": " and there", "my_other_key": "got here"}},
            ),
        ),
        (FloatBetween(0.5, 0.8), ((), {"inner": {"my_key": "got here and there"}})),
        (FloatBetween(0.5, 0.8), ((), {"outer_2": {"my_key": " and back again"}})),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_stream_buffering_single_node(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(state: State, writer: StreamWriter):
        writer("Before sleep")
        time.sleep(0.2)
        writer("After sleep")
        return {"my_key": "got here"}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    graph = builder.compile(checkpointer=checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in graph.stream({"my_key": ""}, config, stream_mode="custom"):
        chunks.append((round(time.perf_counter() - start, 1), c))

    assert chunks == [
        (FloatBetween(0.0, 0.1), "Before sleep"),
        (FloatBetween(0.2, 0.3), "After sleep"),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_nested_graph_interrupts_parallel(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        time.sleep(0.1)
        return {"my_key": "got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {
            "my_key": " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def outer_1(state: State):
        return {"my_key": " and parallel"}

    def outer_2(state: State):
        return {"my_key": " and back again"}

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(interrupt_before=["inner_2"]))
    graph.add_node("outer_1", outer_1)
    graph.add_node("outer_2", outer_2)

    graph.add_edge(START, "inner")
    graph.add_edge(START, "outer_1")
    graph.add_edge(["inner", "outer_1"], "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": ""}, config, debug=True) == {
        "my_key": " and parallel",
    }

    assert app.invoke(None, config, debug=True) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [*app.stream({"my_key": ""}, config, subgraphs=True)] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        ((AnyStr("inner:"),), {"inner_1": {"my_key": "got here", "my_other_key": ""}}),
        ((), {"__interrupt__": ()}),
    ]
    assert [*app.stream(None, config)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""}
    ]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [*app.stream({"my_key": ""}, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_doubly_nested_graph_interrupts(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: str

    class ChildState(TypedDict):
        my_key: str

    class GrandChildState(TypedDict):
        my_key: str

    def grandchild_1(state: ChildState):
        return {"my_key": state["my_key"] + " here"}

    def grandchild_2(state: ChildState):
        return {
            "my_key": state["my_key"] + " and there",
        }

    grandchild = StateGraph(GrandChildState)
    grandchild.add_node("grandchild_1", grandchild_1)
    grandchild.add_node("grandchild_2", grandchild_2)
    grandchild.add_edge("grandchild_1", "grandchild_2")
    grandchild.set_entry_point("grandchild_1")
    grandchild.set_finish_point("grandchild_2")

    child = StateGraph(ChildState)
    child.add_node(
        "child_1",
        grandchild.compile(interrupt_before=["grandchild_2"]),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile())
    graph.add_node("parent_2", parent_2)
    graph.set_entry_point("parent_1")
    graph.add_edge("parent_1", "child")
    graph.add_edge("child", "parent_2")
    graph.set_finish_point("parent_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": "my value"}, config, debug=True) == {
        "my_key": "hi my value",
    }

    assert app.invoke(None, config, debug=True) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    nodes: list[str] = []
    config = {
        "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
    }
    assert [*app.stream({"my_key": "my value"}, config)] == [
        {"parent_1": {"my_key": "hi my value"}},
        {"__interrupt__": ()},
    ]
    assert nodes == ["parent_1", "grandchild_1"]
    assert [*app.stream(None, config)] == [
        {"child": {"my_key": "hi my value here and there"}},
        {"parent_2": {"my_key": "hi my value here and there and back again"}},
    ]
    assert nodes == [
        "parent_1",
        "grandchild_1",
        "grandchild_2",
        "child_1",
        "child",
        "parent_2",
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [*app.stream({"my_key": "my value"}, config, stream_mode="values")] == [
        {"my_key": "my value"},
        {"my_key": "hi my value"},
    ]
    assert [*app.stream(None, config, stream_mode="values")] == [
        {"my_key": "hi my value"},
        {"my_key": "hi my value here and there"},
        {"my_key": "hi my value here and there and back again"},
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_nested_graph_state(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class InnerState(TypedDict):
        my_key: str
        my_other_key: str

    def inner_1(state: InnerState):
        return {
            "my_key": state["my_key"] + " here",
            "my_other_key": state["my_key"],
        }

    def inner_2(state: InnerState):
        return {
            "my_key": state["my_key"] + " and there",
            "my_other_key": state["my_key"],
        }

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str
        other_parent_key: str

    def outer_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def outer_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("outer_1", outer_1)
    graph.add_node(
        "inner",
        inner.compile(interrupt_before=["inner_2"]),
    )
    graph.add_node("outer_2", outer_2)
    graph.set_entry_point("outer_1")
    graph.add_edge("outer_1", "inner")
    graph.add_edge("inner", "outer_2")
    graph.set_finish_point("outer_2")

    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    app.invoke({"my_key": "my value"}, config, debug=True)
    # test state w/ nested subgraph state (right after interrupt)
    # first get_state without subgraph state
    assert app.get_state(config) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                (PULL, "inner"),
                state={"configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}},
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {"outer_1": {"my_key": "hi my value"}},
            "step": 1,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    # now, get_state with subgraphs state
    assert app.get_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "inner",
                (PULL, "inner"),
                state=StateSnapshot(
                    values={
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    },
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            "inner_2",
                            (PULL, "inner_2"),
                        ),
                    ),
                    next=("inner_2",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {
                            "": AnyStr(),
                        },
                        "source": "loop",
                        "writes": {
                            "inner_1": {
                                "my_key": "hi my value here",
                                "my_other_key": "hi my value",
                            }
                        },
                        "step": 1,
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("inner:"),
                        "langgraph_node": "inner",
                        "langgraph_path": [PULL, "inner"],
                        "langgraph_step": 2,
                        "langgraph_triggers": ["outer_1"],
                        "langgraph_checkpoint_ns": AnyStr("inner:"),
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                ),
            ),
        ),
        next=("inner",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {"outer_1": {"my_key": "hi my value"}},
            "step": 1,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    # get_state_history returns outer graph checkpoints
    history = list(app.get_state_history(config))
    assert history == [
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    (PULL, "inner"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("inner:"),
                        }
                    },
                ),
            ),
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "outer_1",
                    (PULL, "outer_1"),
                    result={"my_key": "hi my value"},
                ),
            ),
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": None,
                "step": 0,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "__start__",
                    (PULL, "__start__"),
                    result={"my_key": "my value"},
                ),
            ),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"__start__": {"my_key": "my value"}},
                "step": -1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    # get_state_history for a subgraph returns its checkpoints
    child_history = [*app.get_state_history(history[0].tasks[0].state)]
    assert child_history == [
        StateSnapshot(
            values={"my_key": "hi my value here", "my_other_key": "hi my value"},
            next=("inner_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {
                    "inner_1": {
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    }
                },
                "step": 1,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("inner:"),
                "langgraph_node": "inner",
                "langgraph_path": [PULL, "inner"],
                "langgraph_step": 2,
                "langgraph_triggers": ["outer_1"],
                "langgraph_checkpoint_ns": AnyStr("inner:"),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            tasks=(PregelTask(AnyStr(), "inner_2", (PULL, "inner_2")),),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("inner_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("inner:"),
                "langgraph_node": "inner",
                "langgraph_path": [PULL, "inner"],
                "langgraph_step": 2,
                "langgraph_triggers": ["outer_1"],
                "langgraph_checkpoint_ns": AnyStr("inner:"),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner_1",
                    (PULL, "inner_1"),
                    result={
                        "my_key": "hi my value here",
                        "my_other_key": "hi my value",
                    },
                ),
            ),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("inner:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"__start__": {"my_key": "hi my value"}},
                "step": -1,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("inner:"),
                "langgraph_node": "inner",
                "langgraph_path": [PULL, "inner"],
                "langgraph_step": 2,
                "langgraph_triggers": ["outer_1"],
                "langgraph_checkpoint_ns": AnyStr("inner:"),
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    AnyStr(),
                    "__start__",
                    (PULL, "__start__"),
                    result={"my_key": "hi my value"},
                ),
            ),
        ),
    ]

    # resume
    app.invoke(None, config, debug=True)
    # test state w/ nested subgraph state (after resuming from interrupt)
    assert app.get_state(config) == StateSnapshot(
        values={"my_key": "hi my value here and there and back again"},
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {
                "outer_2": {"my_key": "hi my value here and there and back again"}
            },
            "step": 3,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    # test full history at the end
    actual_history = list(app.get_state_history(config))
    expected_history = [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {
                    "outer_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "outer_2",
                    (PULL, "outer_2"),
                    result={"my_key": "hi my value here and there and back again"},
                ),
            ),
            next=("outer_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {"inner": {"my_key": "hi my value here and there"}},
                "step": 2,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "inner",
                    (PULL, "inner"),
                    state={
                        "configurable": {"thread_id": "1", "checkpoint_ns": AnyStr()}
                    },
                    result={"my_key": "hi my value here and there"},
                ),
            ),
            next=("inner",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {"outer_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "outer_1",
                    (PULL, "outer_1"),
                    result={"my_key": "hi my value"},
                ),
            ),
            next=("outer_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": None,
                "step": 0,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "__start__",
                    (PULL, "__start__"),
                    result={"my_key": "my value"},
                ),
            ),
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"__start__": {"my_key": "my value"}},
                "step": -1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert actual_history == expected_history
    # test looking up parent state by checkpoint ID
    for actual_snapshot, expected_snapshot in zip(actual_history, expected_history):
        assert app.get_state(actual_snapshot.config) == expected_snapshot


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_doubly_nested_graph_state(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class State(TypedDict):
        my_key: str

    class ChildState(TypedDict):
        my_key: str

    class GrandChildState(TypedDict):
        my_key: str

    def grandchild_1(state: ChildState):
        return {"my_key": state["my_key"] + " here"}

    def grandchild_2(state: ChildState):
        return {
            "my_key": state["my_key"] + " and there",
        }

    grandchild = StateGraph(GrandChildState)
    grandchild.add_node("grandchild_1", grandchild_1)
    grandchild.add_node("grandchild_2", grandchild_2)
    grandchild.add_edge("grandchild_1", "grandchild_2")
    grandchild.set_entry_point("grandchild_1")
    grandchild.set_finish_point("grandchild_2")

    child = StateGraph(ChildState)
    child.add_node(
        "child_1",
        grandchild.compile(interrupt_before=["grandchild_2"]),
    )
    child.set_entry_point("child_1")
    child.set_finish_point("child_1")

    def parent_1(state: State):
        return {"my_key": "hi " + state["my_key"]}

    def parent_2(state: State):
        return {"my_key": state["my_key"] + " and back again"}

    graph = StateGraph(State)
    graph.add_node("parent_1", parent_1)
    graph.add_node("child", child.compile())
    graph.add_node("parent_2", parent_2)
    graph.set_entry_point("parent_1")
    graph.add_edge("parent_1", "child")
    graph.add_edge("child", "parent_2")
    graph.set_finish_point("parent_2")

    app = graph.compile(checkpointer=checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert [c for c in app.stream({"my_key": "my value"}, config, subgraphs=True)] == [
        ((), {"parent_1": {"my_key": "hi my value"}}),
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_1": {"my_key": "hi my value here"}},
        ),
        ((), {"__interrupt__": ()}),
    ]
    # get state without subgraphs
    outer_state = app.get_state(config)
    assert outer_state == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                (PULL, "child"),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child"),
                    }
                },
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {"parent_1": {"my_key": "hi my value"}},
            "step": 1,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    child_state = app.get_state(outer_state.tasks[0].state)
    assert (
        child_state.tasks[0]
        == StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "child_1",
                    (PULL, "child_1"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr(),
                        }
                    },
                ),
            ),
            next=("child_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {"": AnyStr()},
                "source": "loop",
                "writes": None,
                "step": 0,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                }
            },
        ).tasks[0]
    )
    grandchild_state = app.get_state(child_state.tasks[0].state)
    assert grandchild_state == StateSnapshot(
        values={"my_key": "hi my value here"},
        tasks=(
            PregelTask(
                AnyStr(),
                "grandchild_2",
                (PULL, "grandchild_2"),
            ),
        ),
        next=("grandchild_2",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr(),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                        AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "parents": AnyDict(
                {
                    "": AnyStr(),
                    AnyStr("child:"): AnyStr(),
                }
            ),
            "source": "loop",
            "writes": {"grandchild_1": {"my_key": "hi my value here"}},
            "step": 1,
            "thread_id": "1",
            "checkpoint_ns": AnyStr("child:"),
            "langgraph_checkpoint_ns": AnyStr("child:"),
            "langgraph_node": "child_1",
            "langgraph_path": [PULL, AnyStr("child_1")],
            "langgraph_step": 1,
            "langgraph_triggers": [AnyStr("start:child_1")],
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr(),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                        AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                    }
                ),
            }
        },
    )
    # get state with subgraphs
    assert app.get_state(config, subgraphs=True) == StateSnapshot(
        values={"my_key": "hi my value"},
        tasks=(
            PregelTask(
                AnyStr(),
                "child",
                (PULL, "child"),
                state=StateSnapshot(
                    values={"my_key": "hi my value"},
                    tasks=(
                        PregelTask(
                            AnyStr(),
                            "child_1",
                            (PULL, "child_1"),
                            state=StateSnapshot(
                                values={"my_key": "hi my value here"},
                                tasks=(
                                    PregelTask(
                                        AnyStr(),
                                        "grandchild_2",
                                        (PULL, "grandchild_2"),
                                    ),
                                ),
                                next=("grandchild_2",),
                                config={
                                    "configurable": {
                                        "thread_id": "1",
                                        "checkpoint_ns": AnyStr(),
                                        "checkpoint_id": AnyStr(),
                                        "checkpoint_map": AnyDict(
                                            {
                                                "": AnyStr(),
                                                AnyStr("child:"): AnyStr(),
                                                AnyStr(
                                                    re.compile(r"child:.+|child1:")
                                                ): AnyStr(),
                                            }
                                        ),
                                    }
                                },
                                metadata={
                                    "parents": AnyDict(
                                        {
                                            "": AnyStr(),
                                            AnyStr("child:"): AnyStr(),
                                        }
                                    ),
                                    "source": "loop",
                                    "writes": {
                                        "grandchild_1": {"my_key": "hi my value here"}
                                    },
                                    "step": 1,
                                    "thread_id": "1",
                                    "checkpoint_ns": AnyStr("child:"),
                                    "langgraph_checkpoint_ns": AnyStr("child:"),
                                    "langgraph_node": "child_1",
                                    "langgraph_path": [
                                        PULL,
                                        AnyStr("child_1"),
                                    ],
                                    "langgraph_step": 1,
                                    "langgraph_triggers": [AnyStr("start:child_1")],
                                },
                                created_at=AnyStr(),
                                parent_config={
                                    "configurable": {
                                        "thread_id": "1",
                                        "checkpoint_ns": AnyStr(),
                                        "checkpoint_id": AnyStr(),
                                        "checkpoint_map": AnyDict(
                                            {
                                                "": AnyStr(),
                                                AnyStr("child:"): AnyStr(),
                                                AnyStr(
                                                    re.compile(r"child:.+|child1:")
                                                ): AnyStr(),
                                            }
                                        ),
                                    }
                                },
                            ),
                        ),
                    ),
                    next=("child_1",),
                    config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                    metadata={
                        "parents": {"": AnyStr()},
                        "source": "loop",
                        "writes": None,
                        "step": 0,
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("child:"),
                        "langgraph_node": "child",
                        "langgraph_path": [PULL, AnyStr("child")],
                        "langgraph_step": 2,
                        "langgraph_triggers": [AnyStr("parent_1")],
                        "langgraph_checkpoint_ns": AnyStr("child:"),
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {"": AnyStr(), AnyStr("child:"): AnyStr()}
                            ),
                        }
                    },
                ),
            ),
        ),
        next=("child",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {"parent_1": {"my_key": "hi my value"}},
            "step": 1,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    # # resume
    assert [c for c in app.stream(None, config, subgraphs=True)] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_2": {"my_key": "hi my value here and there"}},
        ),
        ((AnyStr("child:"),), {"child_1": {"my_key": "hi my value here and there"}}),
        ((), {"child": {"my_key": "hi my value here and there"}}),
        ((), {"parent_2": {"my_key": "hi my value here and there and back again"}}),
    ]
    # get state with and without subgraphs
    assert (
        app.get_state(config)
        == app.get_state(config, subgraphs=True)
        == StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {
                    "parent_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        )
    )
    # get outer graph history
    outer_history = list(app.get_state_history(config))
    assert outer_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there and back again"},
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {
                    "parent_2": {"my_key": "hi my value here and there and back again"}
                },
                "step": 3,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=("parent_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"child": {"my_key": "hi my value here and there"}},
                "step": 2,
                "parents": {},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="parent_2",
                    path=(PULL, "parent_2"),
                    result={"my_key": "hi my value here and there and back again"},
                ),
            ),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            tasks=(
                PregelTask(
                    AnyStr(),
                    "child",
                    (PULL, "child"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child"),
                        }
                    },
                    result={"my_key": "hi my value here and there"},
                ),
            ),
            next=("child",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {"parent_1": {"my_key": "hi my value"}},
                "step": 1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"my_key": "my value"},
            next=("parent_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": {},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="parent_1",
                    path=(PULL, "parent_1"),
                    result={"my_key": "hi my value"},
                ),
            ),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "source": "input",
                "writes": {"__start__": {"my_key": "my value"}},
                "step": -1,
                "parents": {},
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=(PULL, "__start__"),
                    result={"my_key": "my value"},
                ),
            ),
        ),
    ]
    # get child graph history
    child_history = list(app.get_state_history(outer_history[2].tasks[0].state))
    assert child_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"child_1": {"my_key": "hi my value here and there"}},
                "step": 1,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child",
                "langgraph_path": [PULL, AnyStr("child")],
                "langgraph_step": 2,
                "langgraph_triggers": [AnyStr("parent_1")],
                "langgraph_checkpoint_ns": AnyStr("child:"),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            tasks=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("child_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child",
                "langgraph_path": [PULL, AnyStr("child")],
                "langgraph_step": 2,
                "langgraph_triggers": [AnyStr("parent_1")],
                "langgraph_checkpoint_ns": AnyStr("child:"),
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="child_1",
                    path=(PULL, "child_1"),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("child:"),
                        }
                    },
                    result={"my_key": "hi my value here and there"},
                ),
            ),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr("child:"),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {"": AnyStr(), AnyStr("child:"): AnyStr()}
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"__start__": {"my_key": "hi my value"}},
                "step": -1,
                "parents": {"": AnyStr()},
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child",
                "langgraph_path": [PULL, AnyStr("child")],
                "langgraph_step": 2,
                "langgraph_triggers": [AnyStr("parent_1")],
                "langgraph_checkpoint_ns": AnyStr("child:"),
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=(PULL, "__start__"),
                    result={"my_key": "hi my value"},
                ),
            ),
        ),
    ]
    # get grandchild graph history
    grandchild_history = list(app.get_state_history(child_history[1].tasks[0].state))
    assert grandchild_history == [
        StateSnapshot(
            values={"my_key": "hi my value here and there"},
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"grandchild_2": {"my_key": "hi my value here and there"}},
                "step": 2,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child_1",
                "langgraph_path": [
                    PULL,
                    AnyStr("child_1"),
                ],
                "langgraph_step": 1,
                "langgraph_triggers": [AnyStr("start:child_1")],
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            tasks=(),
        ),
        StateSnapshot(
            values={"my_key": "hi my value here"},
            next=("grandchild_2",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": {"grandchild_1": {"my_key": "hi my value here"}},
                "step": 1,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child_1",
                "langgraph_path": [
                    PULL,
                    AnyStr("child_1"),
                ],
                "langgraph_step": 1,
                "langgraph_triggers": [AnyStr("start:child_1")],
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="grandchild_2",
                    path=(PULL, "grandchild_2"),
                    result={"my_key": "hi my value here and there"},
                ),
            ),
        ),
        StateSnapshot(
            values={"my_key": "hi my value"},
            next=("grandchild_1",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "loop",
                "writes": None,
                "step": 0,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child_1",
                "langgraph_path": [
                    PULL,
                    AnyStr("child_1"),
                ],
                "langgraph_step": 1,
                "langgraph_triggers": [AnyStr("start:child_1")],
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="grandchild_1",
                    path=(PULL, "grandchild_1"),
                    result={"my_key": "hi my value here"},
                ),
            ),
        ),
        StateSnapshot(
            values={},
            next=("__start__",),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": AnyStr(),
                    "checkpoint_id": AnyStr(),
                    "checkpoint_map": AnyDict(
                        {
                            "": AnyStr(),
                            AnyStr("child:"): AnyStr(),
                            AnyStr(re.compile(r"child:.+|child1:")): AnyStr(),
                        }
                    ),
                }
            },
            metadata={
                "source": "input",
                "writes": {"__start__": {"my_key": "hi my value"}},
                "step": -1,
                "parents": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("child:"): AnyStr(),
                    }
                ),
                "thread_id": "1",
                "checkpoint_ns": AnyStr("child:"),
                "langgraph_checkpoint_ns": AnyStr("child:"),
                "langgraph_node": "child_1",
                "langgraph_path": [
                    PULL,
                    AnyStr("child_1"),
                ],
                "langgraph_step": 1,
                "langgraph_triggers": [AnyStr("start:child_1")],
            },
            created_at=AnyStr(),
            parent_config=None,
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=(PULL, "__start__"),
                    result={"my_key": "hi my value"},
                ),
            ),
        ),
    ]

    # replay grandchild checkpoint
    assert [
        c for c in app.stream(None, grandchild_history[2].config, subgraphs=True)
    ] == [
        (
            (AnyStr("child:"), AnyStr("child_1:")),
            {"grandchild_1": {"my_key": "hi my value here"}},
        ),
        ((), {"__interrupt__": ()}),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_send_to_nested_graphs(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    class OverallState(TypedDict):
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]

    def continue_to_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    class JokeState(TypedDict):
        subject: str

    def edit(state: JokeState):
        subject = state["subject"]
        return {"subject": f"{subject} - hohoho"}

    # subgraph
    subgraph = StateGraph(JokeState, output=OverallState)
    subgraph.add_node("edit", edit)
    subgraph.add_node(
        "generate", lambda state: {"jokes": [f"Joke about {state['subject']}"]}
    )
    subgraph.set_entry_point("edit")
    subgraph.add_edge("edit", "generate")
    subgraph.set_finish_point("generate")

    # parent graph
    builder = StateGraph(OverallState)
    builder.add_node(
        "generate_joke",
        subgraph.compile(interrupt_before=["generate"]),
    )
    builder.add_conditional_edges(START, continue_to_jokes)
    builder.add_edge("generate_joke", END)

    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    tracer = FakeTracer()

    # invoke and pause at nested interrupt
    assert graph.invoke(
        {"subjects": ["cats", "dogs"]}, config={**config, "callbacks": [tracer]}
    ) == {
        "subjects": ["cats", "dogs"],
        "jokes": [],
    }
    assert len(tracer.runs) == 1, "Should produce exactly 1 root run"

    # check state
    outer_state = graph.get_state(config)

    if not FF_SEND_V2:
        # update state of dogs joke graph
        graph.update_state(outer_state.tasks[1].state, {"subject": "turtles - hohoho"})

        # continue past interrupt
        assert sorted(
            graph.stream(None, config=config),
            key=lambda d: d["generate_joke"]["jokes"][0],
        ) == [
            {"generate_joke": {"jokes": ["Joke about cats - hohoho"]}},
            {"generate_joke": {"jokes": ["Joke about turtles - hohoho"]}},
        ]
        return

    assert outer_state == StateSnapshot(
        values={"subjects": ["cats", "dogs"], "jokes": []},
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="__start__",
                path=("__pregel_pull", "__start__"),
                error=None,
                interrupts=(),
                state=None,
                result={"subjects": ["cats", "dogs"]},
            ),
            PregelTask(
                AnyStr(),
                "generate_joke",
                (PUSH, ("__pregel_pull", "__start__"), 1, AnyStr()),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("generate_joke:"),
                    }
                },
            ),
            PregelTask(
                AnyStr(),
                "generate_joke",
                (PUSH, ("__pregel_pull", "__start__"), 2, AnyStr()),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("generate_joke:"),
                    }
                },
            ),
        ),
        next=("generate_joke", "generate_joke"),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "input",
            "writes": {"__start__": {"subjects": ["cats", "dogs"]}},
            "step": -1,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config=None,
    )
    # check state of each of the inner tasks
    assert graph.get_state(outer_state.tasks[1].state) == StateSnapshot(
        values={"subject": "cats - hohoho", "jokes": []},
        next=("generate",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr("generate_joke:"),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("generate_joke:"): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "writes": {"edit": None},
            "parents": {"": AnyStr()},
            "thread_id": "1",
            "checkpoint_ns": AnyStr("generate_joke:"),
            "langgraph_checkpoint_ns": AnyStr("generate_joke:"),
            "langgraph_node": "generate_joke",
            "langgraph_path": [PUSH, ["__pregel_pull", "__start__"], 1, AnyStr()],
            "langgraph_step": 0,
            "langgraph_triggers": [PUSH],
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr("generate_joke:"),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("generate_joke:"): AnyStr(),
                    }
                ),
            }
        },
        tasks=(PregelTask(id=AnyStr(""), name="generate", path=(PULL, "generate")),),
    )
    assert graph.get_state(outer_state.tasks[2].state) == StateSnapshot(
        values={"subject": "dogs - hohoho", "jokes": []},
        next=("generate",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr("generate_joke:"),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("generate_joke:"): AnyStr(),
                    }
                ),
            }
        },
        metadata={
            "step": 1,
            "source": "loop",
            "writes": {"edit": None},
            "parents": {"": AnyStr()},
            "thread_id": "1",
            "checkpoint_ns": AnyStr("generate_joke:"),
            "langgraph_checkpoint_ns": AnyStr("generate_joke:"),
            "langgraph_node": "generate_joke",
            "langgraph_path": [PUSH, ["__pregel_pull", "__start__"], 2, AnyStr()],
            "langgraph_step": 0,
            "langgraph_triggers": [PUSH],
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": AnyStr("generate_joke:"),
                "checkpoint_id": AnyStr(),
                "checkpoint_map": AnyDict(
                    {
                        "": AnyStr(),
                        AnyStr("generate_joke:"): AnyStr(),
                    }
                ),
            }
        },
        tasks=(PregelTask(id=AnyStr(""), name="generate", path=(PULL, "generate")),),
    )
    # update state of dogs joke graph
    graph.update_state(
        outer_state.tasks[2 if FF_SEND_V2 else 1].state, {"subject": "turtles - hohoho"}
    )

    # continue past interrupt
    assert sorted(
        graph.stream(None, config=config), key=lambda d: d["generate_joke"]["jokes"][0]
    ) == [
        {"generate_joke": {"jokes": ["Joke about cats - hohoho"]}},
        {"generate_joke": {"jokes": ["Joke about turtles - hohoho"]}},
    ]

    actual_snapshot = graph.get_state(config)
    expected_snapshot = StateSnapshot(
        values={
            "subjects": ["cats", "dogs"],
            "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
        },
        tasks=(),
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "parents": {},
            "source": "loop",
            "writes": {
                "generate_joke": [
                    {"jokes": ["Joke about cats - hohoho"]},
                    {"jokes": ["Joke about turtles - hohoho"]},
                ]
            },
            "step": 0,
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
    )
    assert actual_snapshot == expected_snapshot

    # test full history
    actual_history = list(graph.get_state_history(config))

    # get subgraph node state for expected history
    expected_history = [
        StateSnapshot(
            values={
                "subjects": ["cats", "dogs"],
                "jokes": ["Joke about cats - hohoho", "Joke about turtles - hohoho"],
            },
            tasks=(),
            next=(),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "loop",
                "writes": {
                    "generate_joke": [
                        {"jokes": ["Joke about cats - hohoho"]},
                        {"jokes": ["Joke about turtles - hohoho"]},
                    ]
                },
                "step": 0,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
        ),
        StateSnapshot(
            values={"jokes": []},
            tasks=(
                PregelTask(
                    id=AnyStr(),
                    name="__start__",
                    path=("__pregel_pull", "__start__"),
                    error=None,
                    interrupts=(),
                    state=None,
                    result={"subjects": ["cats", "dogs"]},
                ),
                PregelTask(
                    AnyStr(),
                    "generate_joke",
                    (PUSH, ("__pregel_pull", "__start__"), 1, AnyStr()),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("generate_joke:"),
                        }
                    },
                    result={"jokes": ["Joke about cats - hohoho"]},
                ),
                PregelTask(
                    AnyStr(),
                    "generate_joke",
                    (PUSH, ("__pregel_pull", "__start__"), 2, AnyStr()),
                    state={
                        "configurable": {
                            "thread_id": "1",
                            "checkpoint_ns": AnyStr("generate_joke:"),
                        }
                    },
                    result={"jokes": ["Joke about turtles - hohoho"]},
                ),
            ),
            next=("__start__", "generate_joke", "generate_joke"),
            config={
                "configurable": {
                    "thread_id": "1",
                    "checkpoint_ns": "",
                    "checkpoint_id": AnyStr(),
                }
            },
            metadata={
                "parents": {},
                "source": "input",
                "writes": {"__start__": {"subjects": ["cats", "dogs"]}},
                "step": -1,
                "thread_id": "1",
            },
            created_at=AnyStr(),
            parent_config=None,
        ),
    ]
    assert actual_history == expected_history


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_weather_subgraph(
    request: pytest.FixtureRequest, checkpointer_name: str, snapshot: SnapshotAssertion
) -> None:
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, ToolCall
    from langchain_core.tools import tool

    from langgraph.graph import MessagesState

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)

    # setup subgraph

    @tool
    def get_weather(city: str):
        """Get the weather for a specific city"""
        return f"I'ts sunny in {city}!"

    weather_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="get_weather",
                        args={"city": "San Francisco"},
                    )
                ],
            )
        ]
    )

    class SubGraphState(MessagesState):
        city: str

    def model_node(state: SubGraphState, writer: StreamWriter):
        writer(" very")
        result = weather_model.invoke(state["messages"])
        return {"city": cast(AIMessage, result).tool_calls[0]["args"]["city"]}

    def weather_node(state: SubGraphState, writer: StreamWriter):
        writer(" good")
        result = get_weather.invoke({"city": state["city"]})
        return {"messages": [{"role": "assistant", "content": result}]}

    subgraph = StateGraph(SubGraphState)
    subgraph.add_node(model_node)
    subgraph.add_node(weather_node)
    subgraph.add_edge(START, "model_node")
    subgraph.add_edge("model_node", "weather_node")
    subgraph.add_edge("weather_node", END)
    subgraph = subgraph.compile(interrupt_before=["weather_node"])

    # setup main graph

    class RouterState(MessagesState):
        route: Literal["weather", "other"]

    router_model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tool_call123",
                        name="router",
                        args={"dest": "weather"},
                    )
                ],
            )
        ]
    )

    def router_node(state: RouterState, writer: StreamWriter):
        writer("I'm")
        system_message = "Classify the incoming query as either about weather or not."
        messages = [{"role": "system", "content": system_message}] + state["messages"]
        route = router_model.invoke(messages)
        return {"route": cast(AIMessage, route).tool_calls[0]["args"]["dest"]}

    def normal_llm_node(state: RouterState):
        return {"messages": [AIMessage("Hello!")]}

    def route_after_prediction(state: RouterState):
        if state["route"] == "weather":
            return "weather_graph"
        else:
            return "normal_llm_node"

    def weather_graph(state: RouterState):
        return subgraph.invoke(state)

    graph = StateGraph(RouterState)
    graph.add_node(router_node)
    graph.add_node(normal_llm_node)
    graph.add_node("weather_graph", weather_graph)
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges("router_node", route_after_prediction)
    graph.add_edge("normal_llm_node", END)
    graph.add_edge("weather_graph", END)
    graph = graph.compile(checkpointer=checkpointer)

    assert graph.get_graph(xray=1).draw_mermaid() == snapshot

    config = {"configurable": {"thread_id": "1"}}
    thread2 = {"configurable": {"thread_id": "2"}}
    inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}

    # run with custom output
    assert [c for c in graph.stream(inputs, thread2, stream_mode="custom")] == [
        "I'm",
        " very",
    ]
    assert [c for c in graph.stream(None, thread2, stream_mode="custom")] == [
        " good",
    ]

    # run until interrupt
    assert [
        c
        for c in graph.stream(
            inputs, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        ((), {"router_node": {"route": "weather"}}),
        ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
        ((), {"__interrupt__": ()}),
    ]

    # check current state
    state = graph.get_state(config)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"router_node": {"route": "weather"}},
            "step": 1,
            "parents": {},
            "thread_id": "1",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state={
                    "configurable": {
                        "thread_id": "1",
                        "checkpoint_ns": AnyStr("weather_graph:"),
                    }
                },
            ),
        ),
    )

    # update
    graph.update_state(state.tasks[0].state, {"city": "la"})

    # run after update
    assert [
        c
        for c in graph.stream(
            None, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        (
            (AnyStr("weather_graph:"),),
            {
                "weather_node": {
                    "messages": [{"role": "assistant", "content": "I'ts sunny in la!"}]
                }
            },
        ),
        (
            (),
            {
                "weather_graph": {
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf"),
                        _AnyIdAIMessage(content="I'ts sunny in la!"),
                    ]
                }
            },
        ),
    ]

    # try updating acting as weather node
    config = {"configurable": {"thread_id": "14"}}
    inputs = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    assert [
        c
        for c in graph.stream(
            inputs, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        ((), {"router_node": {"route": "weather"}}),
        ((AnyStr("weather_graph:"),), {"model_node": {"city": "San Francisco"}}),
        ((), {"__interrupt__": ()}),
    ]
    state = graph.get_state(config, subgraphs=True)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"router_node": {"route": "weather"}},
            "step": 1,
            "parents": {},
            "thread_id": "14",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state=StateSnapshot(
                    values={
                        "messages": [
                            _AnyIdHumanMessage(content="what's the weather in sf")
                        ],
                        "city": "San Francisco",
                    },
                    next=("weather_node",),
                    config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    metadata={
                        "source": "loop",
                        "writes": {"model_node": {"city": "San Francisco"}},
                        "step": 1,
                        "parents": {"": AnyStr()},
                        "thread_id": "14",
                        "checkpoint_ns": AnyStr("weather_graph:"),
                        "langgraph_node": "weather_graph",
                        "langgraph_path": [PULL, "weather_graph"],
                        "langgraph_step": 2,
                        "langgraph_triggers": [
                            "branch:router_node:route_after_prediction:weather_graph"
                        ],
                        "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    tasks=(
                        PregelTask(
                            id=AnyStr(),
                            name="weather_node",
                            path=(PULL, "weather_node"),
                        ),
                    ),
                ),
            ),
        ),
    )
    graph.update_state(
        state.tasks[0].state.config,
        {"messages": [{"role": "assistant", "content": "rainy"}]},
        as_node="weather_node",
    )
    state = graph.get_state(config, subgraphs=True)
    assert state == StateSnapshot(
        values={
            "messages": [_AnyIdHumanMessage(content="what's the weather in sf")],
            "route": "weather",
        },
        next=("weather_graph",),
        config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {"router_node": {"route": "weather"}},
            "step": 1,
            "parents": {},
            "thread_id": "14",
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "14",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(
            PregelTask(
                id=AnyStr(),
                name="weather_graph",
                path=(PULL, "weather_graph"),
                state=StateSnapshot(
                    values={
                        "messages": [
                            _AnyIdHumanMessage(content="what's the weather in sf"),
                            _AnyIdAIMessage(content="rainy"),
                        ],
                        "city": "San Francisco",
                    },
                    next=(),
                    config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    metadata={
                        "step": 2,
                        "source": "update",
                        "writes": {
                            "weather_node": {
                                "messages": [{"role": "assistant", "content": "rainy"}]
                            }
                        },
                        "parents": {"": AnyStr()},
                        "thread_id": "14",
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": AnyStr("weather_graph:"),
                        "langgraph_node": "weather_graph",
                        "langgraph_path": [PULL, "weather_graph"],
                        "langgraph_step": 2,
                        "langgraph_triggers": [
                            "branch:router_node:route_after_prediction:weather_graph"
                        ],
                        "langgraph_checkpoint_ns": AnyStr("weather_graph:"),
                    },
                    created_at=AnyStr(),
                    parent_config={
                        "configurable": {
                            "thread_id": "14",
                            "checkpoint_ns": AnyStr("weather_graph:"),
                            "checkpoint_id": AnyStr(),
                            "checkpoint_map": AnyDict(
                                {
                                    "": AnyStr(),
                                    AnyStr("weather_graph:"): AnyStr(),
                                }
                            ),
                        }
                    },
                    tasks=(),
                ),
            ),
        ),
    )
    assert [
        c
        for c in graph.stream(
            None, config=config, stream_mode="updates", subgraphs=True
        )
    ] == [
        (
            (),
            {
                "weather_graph": {
                    "messages": [
                        _AnyIdHumanMessage(content="what's the weather in sf"),
                        _AnyIdAIMessage(content="rainy"),
                    ]
                }
            },
        ),
    ]


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_remove_message_via_state_update(
    request: pytest.FixtureRequest, checkpointer_name: str
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = MessageGraph()
    workflow.add_node(
        "chatbot",
        lambda state: [
            AIMessage(
                content="Hello! How can I help you",
            )
        ],
    )

    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)

    checkpointer = request.getfixturevalue("checkpointer_" + checkpointer_name)
    app = workflow.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke([HumanMessage(content="Hi")], config=config)
    app.update_state(config, values=[RemoveMessage(id=output[-1].id)])

    updated_state = app.get_state(config)

    assert len(updated_state.values) == 1
    assert updated_state.values[-1].content == "Hi"


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_channel_values(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    config = {"configurable": {"thread_id": "1"}}
    chain = Channel.subscribe_to("input") | Channel.write_to("output")
    app = Pregel(
        nodes={
            "one": chain,
        },
        channels={
            "ephemeral": EphemeralValue(Any),
            "input": LastValue(int),
            "output": LastValue(int),
        },
        input_channels=["input", "ephemeral"],
        output_channels="output",
        checkpointer=checkpointer,
    )
    app.invoke({"input": 1, "ephemeral": "meow"}, config)
    assert checkpointer.get(config)["channel_values"] == {"input": 1, "output": 1}


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
@pytest.mark.parametrize("store_name", ALL_STORES_SYNC)
def test_store_injected(
    request: pytest.FixtureRequest, checkpointer_name: str, store_name: str
) -> None:
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    the_store = request.getfixturevalue(f"store_{store_name}")

    class State(TypedDict):
        count: Annotated[int, operator.add]

    doc_id = str(uuid.uuid4())
    doc = {"some-key": "this-is-a-val"}
    uid = uuid.uuid4().hex
    namespace = (f"foo-{uid}", "bar")
    thread_1 = str(uuid.uuid4())
    thread_2 = str(uuid.uuid4())

    class Node:
        def __init__(self, i: Optional[int] = None):
            self.i = i
        def __call__(self, inputs: State, config: RunnableConfig, store: BaseStore):
            assert isinstance(store, BaseStore)
            store.put(
                namespace
                if self.i is not None
                and config["configurable"]["thread_id"] in (thread_1, thread_2)
                else (f"foo_{self.i}", "bar"),
                doc_id,
                {
                    **doc,
                    "from_thread": config["configurable"]["thread_id"],
                    "some_val": inputs["count"],
                },
            )
            return {"count": 1}

    builder = StateGraph(State)
    builder.add_node("node", Node())
    builder.add_edge("__start__", "node")
    N = 500
    M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    graph = builder.compile(store=the_store, checkpointer=checkpointer)

    results = graph.batch(
        [{"count": 0}] * M,
        ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
        + [{"configurable": {"thread_id": thread_1}}],
    )
    result = results[-1]
    assert result == {"count": N + 1}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
    assert len(the_store.search(namespace)) == 1
    # Check results after another turn of the same thread
    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_1}})
    assert result == {"count": (N + 1) * 2}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
    assert len(the_store.search(namespace)) == 1

    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_2}})
    assert result == {"count": N + 1}
    returned_doc = the_store.get(namespace, doc_id).value
    assert returned_doc == {
        **doc,
        "from_thread": thread_2,
        "some_val": 0,
    }  # Overwrites the whole doc
    assert len(the_store.search(namespace)) == 1  # still overwriting the same one


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_parent_command(request: pytest.FixtureRequest, checkpointer_name: str) -> None:
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import tool

    @tool(return_direct=True)
    def get_user_name() -> Command:
        """Retrieve user name"""
        return Command(update={"user_name": "Meow"}, graph=Command.PARENT)

    subgraph_builder = StateGraph(MessagesState)
    subgraph_builder.add_node("tool", get_user_name)
    subgraph_builder.add_edge(START, "tool")
    subgraph = subgraph_builder.compile()

    class CustomParentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        # this key is not available to the child graph
        user_name: str

    builder = StateGraph(CustomParentState)
    builder.add_node("alice", subgraph)
    builder.add_edge(START, "alice")
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"messages": [("user", "get user name")]}, config) == {
        "messages": [
            _AnyIdHumanMessage(
                content="get user name", additional_kwargs={}, response_metadata={}
            ),
        ],
        "user_name": "Meow",
    }
    assert graph.get_state(config) == StateSnapshot(
        values={
            "messages": [
                _AnyIdHumanMessage(
                    content="get user name", additional_kwargs={}, response_metadata={}
                ),
            ],
            "user_name": "Meow",
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        metadata={
            "source": "loop",
            "writes": {
                "alice": {
                    "user_name": "Meow",
                }
            },
            "thread_id": "1",
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        tasks=(),
    )


@pytest.mark.parametrize("checkpointer_name", ALL_CHECKPOINTERS_SYNC)
def test_interrupt_subgraph(request: pytest.FixtureRequest, checkpointer_name: str):
    checkpointer = request.getfixturevalue(f"checkpointer_{checkpointer_name}")

    class State(TypedDict):
        baz: str

    def foo(state):
        return {"baz": "foo"}

    def bar(state):
        value = interrupt("Please provide baz value:")
        return {"baz": value}

    child_builder = StateGraph(State)
    child_builder.add_node(bar)
    child_builder.add_edge(START, "bar")

    builder = StateGraph(State)
    builder.add_node(foo)
    builder.add_node("bar", child_builder.compile())
    builder.add_edge(START, "foo")
    builder.add_edge("foo", "bar")
    graph = builder.compile(checkpointer=checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"baz": ""}, thread1)
    # Resume with answer
    assert graph.invoke(Command(resume="bar"), thread1)
