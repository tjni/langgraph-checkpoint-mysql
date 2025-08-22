import logging
import operator
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Union,
)

import pytest
from langchain_core.messages import AnyMessage
from langchain_core.runnables import (
    RunnableConfig,
)
from langsmith import traceable
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytest_mock import MockerFixture
from typing_extensions import TypedDict


from langgraph._internal._constants import CONFIG_KEY_NODE_FINISHED, ERROR, PULL
from langgraph.cache.base import BaseCache
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue
from langgraph.channels.topic import Topic
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.errors import InvalidUpdateError, ParentCommand
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.store.base import BaseStore
from langgraph.types import (
    CachePolicy,
    Command,
    Durability,
    Interrupt,
    PregelTask,
    RetryPolicy,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamWriter,
    interrupt,
)
from tests.any_str import AnyStr, AnyVersion, FloatBetween, UnsortedSequence
from tests.messages import (
    _AnyIdAIMessage,
    _AnyIdHumanMessage,
    _AnyIdHumanMessage,
    _AnyIdToolMessage,
)

pytestmark = pytest.mark.anyio

logger = logging.getLogger(__name__)


def test_run_from_checkpoint_id_retains_previous_writes(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
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
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread_id = uuid.uuid4()
    thread1 = {"configurable": {"thread_id": str(thread_id)}}

    result = graph.invoke({"myval": 1}, thread1, durability="async")
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


def test_invoke_checkpoint_two(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        NodeBuilder()
        .subscribe_to("input")
        .read_from("total")
        .do(add_one)
        .write_to("output", "total")
        .do(raise_if_above_10)
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
        checkpointer=sync_checkpointer,
        retry_policy=RetryPolicy(),
    )

    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, {"configurable": {"thread_id": "1"}}) == 2
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 2
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, {"configurable": {"thread_id": "1"}}) == 5
    assert errored_once, "errored and retried"
    checkpoint_tup = sync_checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, {"configurable": {"thread_id": "1"}})
    # checkpoint is not updated, error is recorded
    checkpoint_tup = sync_checkpointer.get_tuple({"configurable": {"thread_id": "1"}})
    assert checkpoint_tup is not None
    assert checkpoint_tup.checkpoint["channel_values"].get("total") == 7
    assert checkpoint_tup.pending_writes == [
        (AnyStr(), ERROR, "ValueError('Input is too large')")
    ]
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, {"configurable": {"thread_id": "2"}}) == 5
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "1"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 7
    checkpoint = sync_checkpointer.get({"configurable": {"thread_id": "2"}})
    assert checkpoint is not None
    assert checkpoint["channel_values"].get("total") == 5


def test_pending_writes_resume(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        value: Annotated[int, operator.add]

    class AwhileMaker:
        def __init__(self, sleep: float, rtn: Union[dict, Exception]) -> None:
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
    two = AwhileMaker(0.2, ConnectionError("I'm not good"))
    builder = StateGraph(State)
    builder.add_node("one", one)
    builder.add_node(
        "two",
        two,
        retry_policy=RetryPolicy(max_attempts=2, initial_interval=0, jitter=False)
    )
    builder.add_edge(START, "one")
    builder.add_edge(START, "two")
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread1: RunnableConfig = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke({"value": 1}, thread1, durability=durability)

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
    }
    # get_state with checkpoint_id should not apply any pending writes
    state = graph.get_state(state.config)
    assert state is not None
    assert state.values == {"value": 1}
    assert state.next == ("one", "two")
    # should contain pending write of "one"
    checkpoint = sync_checkpointer.get_tuple(thread1)
    assert checkpoint is not None
    # should contain error from "two"
    expected_writes = [
        (AnyStr(), "value", 2),
        (AnyStr(), ERROR, 'ConnectionError("I\'m not good")'),
    ]
    assert len(checkpoint.pending_writes) == 2
    assert all(w in expected_writes for w in checkpoint.pending_writes)
    # both non-error pending writes come from same task
    non_error_writes = [w for w in checkpoint.pending_writes if w[1] != ERROR]
    # error write is from the other task
    error_write = next(w for w in checkpoint.pending_writes if w[1] == ERROR)
    assert error_write[0] != non_error_writes[0][0]

    # resume execution
    with pytest.raises(ConnectionError, match="I'm not good"):
        graph.invoke(None, thread1, durability=durability)

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
    assert graph.invoke(None, thread1, durability=durability) == {
        "value": 6
    }

    # check all final checkpoints
    checkpoints = [c for c in sync_checkpointer.list(thread1)]
    # we should have 3
    assert len(checkpoints) == (3 if durability != "exit" else 2)
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
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {
                "one": {
                    "branch:to:one": AnyVersion(),
                },
                "two": {
                    "branch:to:two": AnyVersion(),
                },
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
                "__interrupt__": {
                    "value": AnyVersion(),
                    "__start__": AnyVersion(),
                    "branch:to:one": AnyVersion(),
                    "branch:to:two": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "branch:to:one": AnyVersion(),
                "branch:to:two": AnyVersion(),
            },
            "channel_values": {"value": 6},
            "updated_channels": ["value"],
        },
        metadata={
            "parents": {},
            "step": 1,
            "source": "loop",
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
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": AnyVersion(),
                },
            },
            "channel_versions": {
                "value": AnyVersion(),
                "__start__": AnyVersion(),
                "branch:to:one": AnyVersion(),
                "branch:to:two": AnyVersion(),
            },
            "channel_values": {
                "value": 1,
                "branch:to:one": None,
                "branch:to:two": None,
            },
            "updated_channels": ["branch:to:one", "branch:to:two", "value"],
        },
        metadata={
            "parents": {},
            "step": 0,
            "source": "loop",
        },
        parent_config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoints[2].config["configurable"]["checkpoint_id"]
            }
        }
        if durability != "exit"
        else None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 2),
            (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
            (AnyStr(), "value", 3),
        )
        if durability != "exit"
        else UnsortedSequence(
            (AnyStr(), "value", 2),
            (AnyStr(), "__error__", 'ConnectionError("I\'m not good")'),
            # the write against the previous checkpoint is not saved, as it is
            # produced in a run where only the next checkpoint (the last) is saved
        ),
    )
    if durability == "exit":
        return
    assert checkpoints[2] == CheckpointTuple(
        config={
            "configurable": {
                "thread_id": "1",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        checkpoint={
            "v": 4,
            "id": AnyStr(),
            "ts": AnyStr(),
            "versions_seen": {"__input__": {}},
            "channel_versions": {
                "__start__": AnyVersion(),
            },
            "channel_values": {"__start__": {"value": 1}},
            "updated_channels": ["__start__"],
        },
        metadata={
            "parents": {},
            "step": -1,
            "source": "input",
        },
        parent_config=None,
        pending_writes=UnsortedSequence(
            (AnyStr(), "value", 1),
            (AnyStr(), "branch:to:one", None),
            (AnyStr(), "branch:to:two", None),
        ),
    )


def test_imp_task(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    mapper_calls = 0

    class Context(TypedDict):
        model: str

    @task()
    def mapper(input: int) -> str:
        nonlocal mapper_calls
        mapper_calls += 1
        time.sleep(input / 100)
        return str(input) * 2

    @entrypoint(checkpointer=sync_checkpointer, context_schema=Context)
    def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = [f.result() for f in futures]
        answer = interrupt("question")
        return [m + answer for m in mapped]

    assert graph.get_input_jsonschema() == {
        "type": "array",
        "items": {"type": "integer"},
        "title": "LangGraphInput",
    }
    assert graph.get_output_jsonschema() == {
        "type": "array",
        "items": {"type": "string"},
        "title": "LangGraphOutput",
    }
    assert graph.get_context_jsonschema() == {
        "properties": {"model": {"title": "Model", "type": "string"}},
        "required": ["model"],
        "title": "Context",
        "type": "object",
    }

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [*graph.stream([0, 1], thread1, durability=durability)] == [
        {"mapper": "00"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    id=AnyStr(),
                ),
            )
        },
    ]
    assert mapper_calls == 2

    assert graph.invoke(
        Command(resume="answer"), thread1, durability=durability
    ) == [
        "00answer",
        "11answer",
    ]
    assert mapper_calls == 2


def test_imp_nested(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    def mynode(input: list[str]) -> list[str]:
        return [it + "a" for it in input]

    builder = StateGraph(list[str])
    builder.add_node(mynode)
    builder.add_edge(START, "mynode")
    add_a = builder.compile()

    @task
    def submapper(input: int) -> str:
        time.sleep(input / 100)
        return str(input)

    @task()
    def mapper(input: int) -> str:
        sub = submapper(input)
        time.sleep(input / 100)
        return sub.result() * 2

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(input: list[int]) -> list[str]:
        futures = [mapper(i) for i in input]
        mapped = [f.result() for f in futures]
        answer = interrupt("question")
        final = [m + answer for m in mapped]
        return add_a.invoke(final)

    assert graph.get_input_jsonschema() == {
        "type": "array",
        "items": {"type": "integer"},
        "title": "LangGraphInput",
    }
    assert graph.get_output_jsonschema() == {
        "type": "array",
        "items": {"type": "string"},
        "title": "LangGraphOutput",
    }

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [*graph.stream([0, 1], thread1, durability=durability)] == [
        {"submapper": "0"},
        {"mapper": "00"},
        {"submapper": "1"},
        {"mapper": "11"},
        {
            "__interrupt__": (
                Interrupt(
                    value="question",
                    id=AnyStr(),
                ),
            )
        },
    ]

    assert graph.invoke(
        Command(resume="answer"), thread1, durability=durability
    ) == [
        "00answera",
        "11answera",
    ]


def test_imp_stream_order(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    @task()
    def foo(state: dict) -> tuple:
        return state["a"] + "foo", "bar"

    @task
    def bar(a: str, b: str, c: Optional[str] = None) -> dict:
        return {"a": a + b, "c": (c or "") + "bark"}

    @task
    def baz(state: dict) -> dict:
        return {"a": state["a"] + "baz", "c": "something else"}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        fut_foo = foo(state)
        fut_bar = bar(*fut_foo.result())
        fut_baz = baz(fut_bar.result())
        return fut_baz.result()

    thread1 = {"configurable": {"thread_id": "1"}}
    assert [
        c
        for c in graph.stream({"a": "0"}, thread1, durability=durability)
    ] == [
        {
            "foo": (
                "0foo",
                "bar",
            )
        },
        {"bar": {"a": "0foobar", "c": "bark"}},
        {"baz": {"a": "0foobarbaz", "c": "something else"}},
        {"graph": {"a": "0foobarbaz", "c": "something else"}},
    ]

    assert graph.get_state(thread1).values == {"a": "0foobarbaz", "c": "something else"}


def test_invoke_checkpoint_three(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
    adder = mocker.Mock(side_effect=lambda x: x["total"] + x["input"])

    def raise_if_above_10(input: int) -> int:
        if input > 10:
            raise ValueError("Input is too large")
        return input

    one = (
        NodeBuilder()
        .subscribe_to("input")
        .read_from("total")
        .do(adder)
        .write_to("output", "total")
        .do(raise_if_above_10)
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
        checkpointer=sync_checkpointer,
    )

    thread_1 = {"configurable": {"thread_id": "1"}}
    # total starts out as 0, so output is 0+2=2
    assert app.invoke(2, thread_1, durability="async") == 2
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 2
    assert state.next == ()
    assert (
        state.config["configurable"]["checkpoint_id"]
        == sync_checkpointer.get(thread_1)["id"]
    )
    # total is now 2, so output is 2+3=5
    assert app.invoke(3, thread_1, durability="async") == 5
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert (
        state.config["configurable"]["checkpoint_id"]
        == sync_checkpointer.get(thread_1)["id"]
    )
    # total is now 2+5=7, so output would be 7+4=11, but raises ValueError
    with pytest.raises(ValueError):
        app.invoke(4, thread_1, durability="async")
    # checkpoint is updated with new input
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 7
    assert state.next == ("one",)
    """we checkpoint inputs and it failed on "one", so the next node is one"""
    # we can recover from error by sending new inputs
    assert app.invoke(2, thread_1, durability="async") == 9
    state = app.get_state(thread_1)
    assert state is not None
    assert state.values.get("total") == 16, "total is now 7+9=16"
    assert state.next == ()

    thread_2 = {"configurable": {"thread_id": "2"}}
    # on a new thread, total starts out as 0, so output is 0+5=5
    assert app.invoke(5, thread_2) == 5
    state = app.get_state(thread_1)
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
        sync_checkpointer.get(thread_1_history[0].config)["id"]
        == thread_1_history[0].config["configurable"]["checkpoint_id"]
    )
    assert (
        sync_checkpointer.get(thread_1_history[1].config)["id"]
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


def test_invoke_join_then_call_other_pregel(
    mocker: MockerFixture, sync_checkpointer: BaseCheckpointSaver
) -> None:
    add_one = mocker.Mock(side_effect=lambda x: x + 1)
    add_10_each = mocker.Mock(side_effect=lambda x: [y + 10 for y in x])

    inner_app = Pregel(
        nodes={
            "one": NodeBuilder().subscribe_only("input").do(add_one).write_to("output")
        },
        channels={
            "output": LastValue(int),
            "input": LastValue(int),
        },
        input_channels="input",
        output_channels="output",
    )

    one = NodeBuilder().subscribe_only("input").do(add_10_each).write_to("inbox_one")
    two = (
        NodeBuilder()
        .subscribe_only("inbox_one")
        .do(inner_app.map())
        .write_to("outbox_one")
    )
    chain_three = NodeBuilder().subscribe_only("outbox_one").do(sum).write_to("output")

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
    app.checkpointer = sync_checkpointer
    # subgraph is called twice in the same node, but that works
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27

    # set inner graph checkpointer NeverCheckpoint
    inner_app.checkpointer = False
    # subgraph still called twice, but checkpointing for inner graph is disabled
    assert app.invoke([2, 3], {"configurable": {"thread_id": "1"}}) == 27


def test_in_one_fan_out_state_graph_waiting_edge(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
        checkpointer=sync_checkpointer,
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
        checkpointer=sync_checkpointer,
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
    expected_parent_config = (
        list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
    )
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        },
        tasks=(PregelTask(AnyStr(), "qa", (PULL, "qa")),),
        next=("qa",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
        },
        parent_config=expected_parent_config,
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


@pytest.mark.parametrize("use_waiting_edge", (True, False))
def test_in_one_fan_out_state_graph_defer_node(
    sync_checkpointer: BaseCheckpointSaver,
    use_waiting_edge: bool,
) -> None:
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
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
    workflow.add_node(qa, defer=True)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retriever_one")
    workflow.add_edge("retriever_one", "analyzer_one")
    workflow.add_edge("rewrite_query", "retriever_two")
    if use_waiting_edge:
        workflow.add_edge(["retriever_one", "retriever_two"], "qa")
    else:
        workflow.add_edge("retriever_one", "qa")
        workflow.add_edge("retriever_two", "qa")
    workflow.set_finish_point("qa")

    app = workflow.compile()

    assert app.invoke({"query": "what is weather in sf"}) == {
        "query": "analyzed: query: what is weather in sf",
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [*app.stream({"query": "what is weather in sf"})] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert [*app.stream({"query": "what is weather in sf"}, stream_mode="debug")] == [
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "rewrite_query",
                "input": {"query": "what is weather in sf", "docs": []},
                "triggers": ("branch:to:rewrite_query",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 1,
            "payload": {
                "id": AnyStr(),
                "name": "rewrite_query",
                "error": None,
                "result": [("query", "query: what is weather in sf")],
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_one",
                "input": {"query": "query: what is weather in sf", "docs": []},
                "triggers": ("branch:to:retriever_one",),
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_two",
                "input": {"query": "query: what is weather in sf", "docs": []},
                "triggers": ("branch:to:retriever_two",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_one",
                "error": None,
                "result": [("docs", ["doc1", "doc2"])],
                "interrupts": [],
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 2,
            "payload": {
                "id": AnyStr(),
                "name": "retriever_two",
                "error": None,
                "result": [("docs", ["doc3", "doc4"])],
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "analyzer_one",
                "input": {
                    "query": "query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                },
                "triggers": ("branch:to:analyzer_one",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 3,
            "payload": {
                "id": AnyStr(),
                "name": "analyzer_one",
                "error": None,
                "result": [("query", "analyzed: query: what is weather in sf")],
                "interrupts": [],
            },
        },
        {
            "type": "task",
            "timestamp": AnyStr(),
            "step": 4,
            "payload": {
                "id": AnyStr(),
                "name": "qa",
                "input": {
                    "query": "analyzed: query: what is weather in sf",
                    "docs": ["doc1", "doc2", "doc3", "doc4"],
                },
                "triggers": ("branch:to:qa", "join:retriever_one+retriever_two:qa")
                if use_waiting_edge
                else ("branch:to:qa",),
            },
        },
        {
            "type": "task_result",
            "timestamp": AnyStr(),
            "step": 4,
            "payload": {
                "id": AnyStr(),
                "name": "qa",
                "error": None,
                "result": [("answer", "doc1,doc2,doc3,doc4")],
                "interrupts": [],
            },
        },
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["analyzer_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"__interrupt__": ()},
    ]

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_before=["qa"],
    )
    config = {"configurable": {"thread_id": "2"}}

    assert [
        c for c in app_w_interrupt.stream({"query": "what is weather in sf"}, config)
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"__interrupt__": ()},
    ]

    app_w_interrupt.update_state(config, {"docs": ["doc5"]})
    expected_parent_config = (
        list(app_w_interrupt.checkpointer.list(config, limit=2))[-1].config
    )
    assert app_w_interrupt.get_state(config) == StateSnapshot(
        values={
            "query": "analyzed: query: what is weather in sf",
            "docs": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        },
        tasks=(PregelTask(AnyStr(), "qa", (PULL, "qa")),),
        next=("qa",),
        config={
            "configurable": {
                "thread_id": "2",
                "checkpoint_ns": "",
                "checkpoint_id": AnyStr(),
            }
        },
        created_at=AnyStr(),
        metadata={
            "parents": {},
            "source": "update",
            "step": 4,
        },
        parent_config=expected_parent_config,
        interrupts=(),
    )

    assert [c for c in app_w_interrupt.stream(None, config, debug=1)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4,doc5"}},
    ]


def test_in_one_fan_out_state_graph_waiting_edge_via_branch(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        return {"query": f"analyzed: {data['query']}"}

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
        checkpointer=sync_checkpointer,
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


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic2(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        inner: Annotated[InnerObject, lambda x, y: y]
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    class UpdateDocs34(BaseModel):
        docs: list[str] = Field(default_factory=lambda: ["doc3", "doc4"])

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
        return UpdateDocs34()

    def qa(data: State) -> State:
        return {"answer": ",".join(data.docs)}

    def decider(data: State) -> str:
        assert isinstance(data, State)
        return "retriever_two"

    workflow = StateGraph(State, input_schema=Input, output_schema=Output)

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

    with pytest.raises(ValidationError):
        app.invoke({"query": {}})

    assert app.invoke({"query": "what is weather in sf", "inner": {"yo": 1}}) == {
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

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
        checkpointer=sync_checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

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

    assert [c for c in app_w_interrupt.stream(None, config)] == [
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    assert app_w_interrupt.update_state(
        config, {"docs": ["doc5"]}, as_node="rewrite_query"
    ) == {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": AnyStr(),
            "checkpoint_ns": "",
        }
    }


def test_in_one_fan_out_state_graph_waiting_edge_custom_state_class_pydantic_input(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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

    class QueryModel(BaseModel):
        query: str

    class State(QueryModel):
        inner: InnerObject
        answer: Optional[str] = None
        docs: Annotated[list[str], sorted_add]

    class StateUpdate(BaseModel):
        query: Optional[str] = None
        answer: Optional[str] = None
        docs: Optional[list[str]] = None

    class Input(QueryModel):
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

    workflow = StateGraph(State, input_schema=Input, output_schema=Output)

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

    assert app.invoke(
        Input(query="what is weather in sf", inner=InnerObject(yo=1))
    ) == {
        "docs": ["doc1", "doc2", "doc3", "doc4"],
        "answer": "doc1,doc2,doc3,doc4",
    }

    assert [
        *app.stream(Input(query="what is weather in sf", inner=InnerObject(yo=1)))
    ] == [
        {"rewrite_query": {"query": "query: what is weather in sf"}},
        {"analyzer_one": {"query": "analyzed: query: what is weather in sf"}},
        {"retriever_two": {"docs": ["doc3", "doc4"]}},
        {"retriever_one": {"docs": ["doc1", "doc2"]}},
        {"qa": {"answer": "doc1,doc2,doc3,doc4"}},
    ]

    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer,
        interrupt_after=["retriever_one"],
    )
    config = {"configurable": {"thread_id": "1"}}

    assert [
        c
        for c in app_w_interrupt.stream(
            Input(query="what is weather in sf", inner=InnerObject(yo=1)), config
        )
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

    assert app_w_interrupt.update_state(
        config, {"docs": ["doc5"]}, as_node="rewrite_query"
    ) == {
        "configurable": {
            "thread_id": "1",
            "checkpoint_id": AnyStr(),
            "checkpoint_ns": "",
        }
    }


def test_in_one_fan_out_state_graph_waiting_edge_plus_regular(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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
        return {"query": f"query: {data['query']}"}

    def analyzer_one(data: State) -> State:
        time.sleep(0.1)
        return {"query": f"analyzed: {data['query']}"}

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
        checkpointer=sync_checkpointer,
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


def test_subgraph_checkpoint_true(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class InnerState(TypedDict):
        my_key: Annotated[str, operator.add]
        my_other_key: str

    def inner_1(state: InnerState):
        return {"my_key": " got here", "my_other_key": state["my_key"]}

    def inner_2(state: InnerState):
        return {"my_key": " and there"}

    inner = StateGraph(InnerState)
    inner.add_node("inner_1", inner_1)
    inner.add_node("inner_2", inner_2)
    inner.add_edge("inner_1", "inner_2")
    inner.set_entry_point("inner_1")
    inner.set_finish_point("inner_2")

    class State(TypedDict):
        my_key: str

    graph = StateGraph(State)
    graph.add_node("inner", inner.compile(checkpointer=True))
    graph.add_edge(START, "inner")
    graph.add_conditional_edges(
        "inner", lambda s: "inner" if s["my_key"].count("there") < 2 else END
    )
    app = graph.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        for c in app.stream(
            {"my_key": ""}, config, subgraphs=True, durability=durability
        )
    ] == [
        (("inner",), {"inner_1": {"my_key": " got here", "my_other_key": ""}}),
        (("inner",), {"inner_2": {"my_key": " and there"}}),
        ((), {"inner": {"my_key": " got here and there"}}),
        (
            ("inner",),
            {
                "inner_1": {
                    "my_key": " got here",
                    "my_other_key": " got here and there got here and there",
                }
            },
        ),
        (("inner",), {"inner_2": {"my_key": " and there"}}),
        (
            (),
            {
                "inner": {
                    "my_key": " got here and there got here and there got here and there"
                }
            },
        ),
    ]

    checkpoints = list(app.get_state_history(config))
    if durability != "exit":
        assert len(checkpoints) == 4
    else:
        assert len(checkpoints) == 1


def test_subgraph_checkpoint_true_interrupt(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    # Define subgraph
    class SubgraphState(TypedDict):
        # note that none of these keys are shared with the parent graph state
        bar: str
        baz: str

    def subgraph_node_1(state: SubgraphState):
        baz_value = interrupt("Provide baz value")
        return {"baz": baz_value}

    def subgraph_node_2(state: SubgraphState):
        return {"bar": state["bar"] + state["baz"]}

    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile(checkpointer=True)

    class ParentState(TypedDict):
        foo: str

    def node_1(state: ParentState):
        return {"foo": "hi! " + state["foo"]}

    def node_2(state: ParentState):
        response = subgraph.invoke({"bar": state["foo"]})
        return {"foo": response["bar"]}

    builder = StateGraph(ParentState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke(
        {"foo": "foo"}, config, durability=durability
    ) == {
        "foo": "hi! foo",
        "__interrupt__": [
            Interrupt(
                value="Provide baz value",
                id=AnyStr(),
            )
        ],
    }
    assert graph.get_state(config, subgraphs=True).tasks[0].state.values == {
        "bar": "hi! foo"
    }
    assert graph.invoke(
        Command(resume="baz"), config, durability=durability
    ) == {"foo": "hi! foobaz"}


def test_stream_subgraphs_during_execution(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

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


def test_stream_buffering_single_node(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
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
    graph = builder.compile(checkpointer=sync_checkpointer)

    start = time.perf_counter()
    chunks: list[tuple[float, Any]] = []
    config = {"configurable": {"thread_id": "2"}}
    for c in graph.stream({"my_key": ""}, config, stream_mode="custom"):
        chunks.append((round(time.perf_counter() - start, 1), c))

    assert chunks == [
        (FloatBetween(0.0, 0.1), "Before sleep"),
        (FloatBetween(0.2, 0.3), "After sleep"),
    ]


def test_nested_graph_interrupts_parallel(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke({"my_key": ""}, config, durability=durability) == {
        "my_key": " and parallel",
    }

    assert app.invoke(None, config, durability=durability) == {
        "my_key": "got here and there and parallel and back again",
    }

    # below combo of assertions is asserting two things
    # - outer_1 finishes before inner interrupts (because we see its output in stream, which only happens after node finishes)
    # - the writes of outer are persisted in 1st call and used in 2nd call, ie outer isn't called again (because we dont see outer_1 output again in 2nd stream)
    # test stream updates w/ nested interrupt
    config = {"configurable": {"thread_id": "2"}}
    assert [
        *app.stream(
            {"my_key": ""}, config, subgraphs=True, durability=durability
        )
    ] == [
        # we got to parallel node first
        ((), {"outer_1": {"my_key": " and parallel"}}),
        ((AnyStr("inner:"),), {"inner_1": {"my_key": "got here", "my_other_key": ""}}),
        ((), {"__interrupt__": ()}),
    ]
    assert [*app.stream(None, config, durability=durability)] == [
        {"outer_1": {"my_key": " and parallel"}, "__metadata__": {"cached": True}},
        {"inner": {"my_key": "got here and there"}},
        {"outer_2": {"my_key": " and back again"}},
    ]

    # test stream values w/ nested interrupt
    config = {"configurable": {"thread_id": "3"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts BEFORE the parallel node
    app = graph.compile(checkpointer=sync_checkpointer, interrupt_before=["outer_1"])
    config = {"configurable": {"thread_id": "4"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [{"my_key": ""}]
    # while we're waiting for the node w/ interrupt inside to finish
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]

    # test interrupts AFTER the parallel node
    app = graph.compile(checkpointer=sync_checkpointer, interrupt_after=["outer_1"])
    config = {"configurable": {"thread_id": "5"}}
    assert [
        *app.stream(
            {"my_key": ""},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": ""},
        {"my_key": " and parallel"},
    ]
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": ""},
        {"my_key": "got here and there and parallel"},
    ]
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": "got here and there and parallel"},
        {"my_key": "got here and there and parallel and back again"},
    ]


def test_doubly_nested_graph_interrupts(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
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

    app = graph.compile(checkpointer=sync_checkpointer)

    # test invoke w/ nested interrupt
    config = {"configurable": {"thread_id": "1"}}
    assert app.invoke(
        {"my_key": "my value"}, config, durability=durability
    ) == {
        "my_key": "hi my value",
    }

    assert app.invoke(None, config, durability=durability) == {
        "my_key": "hi my value here and there and back again",
    }

    # test stream updates w/ nested interrupt
    nodes: list[str] = []
    config = {
        "configurable": {"thread_id": "2", CONFIG_KEY_NODE_FINISHED: nodes.append}
    }
    assert [
        *app.stream({"my_key": "my value"}, config, durability=durability)
    ] == [
        {"parent_1": {"my_key": "hi my value"}},
        {"__interrupt__": ()},
    ]
    assert nodes == ["parent_1", "grandchild_1"]
    assert [*app.stream(None, config, durability=durability)] == [
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
    assert [
        *app.stream(
            {"my_key": "my value"},
            config,
            stream_mode="values",
            durability=durability,
        )
    ] == [
        {"my_key": "my value"},
        {"my_key": "hi my value"},
    ]
    assert [
        *app.stream(
            None, config, stream_mode="values", durability=durability
        )
    ] == [
        {"my_key": "hi my value"},
        {"my_key": "hi my value here and there"},
        {"my_key": "hi my value here and there and back again"},
    ]


def test_checkpoint_metadata(sync_checkpointer: BaseCheckpointSaver) -> None:
    """This test verifies that a run's configurable fields are merged with the
    previous checkpoint config for each step in the run.
    """
    # set up test
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, AnyMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool

    # graph state
    class BaseState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # initialize graph nodes
    @tool()
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"result for {query}"

    tools = [search_api]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a nice assistant."),
            ("placeholder", "{messages}"),
        ]
    )

    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_call123",
                        "name": "search_api",
                        "args": {"query": "query"},
                    },
                ],
            ),
            AIMessage(content="answer"),
        ]
    )

    @traceable(run_type="llm")
    def agent(state: BaseState) -> BaseState:
        formatted = prompt.invoke(state)
        response = model.invoke(formatted)
        return {"messages": response, "usage_metadata": {"total_tokens": 123}}

    def should_continue(data: BaseState) -> str:
        # Logic to decide whether to continue in the loop or exit
        if not data["messages"][-1].tool_calls:
            return "exit"
        else:
            return "continue"

    # define graphs w/ and w/o interrupt
    workflow = StateGraph(BaseState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "exit": END}
    )
    workflow.add_edge("tools", "agent")

    # graph w/o interrupt
    app = workflow.compile(checkpointer=sync_checkpointer)

    # graph w/ interrupt
    app_w_interrupt = workflow.compile(
        checkpointer=sync_checkpointer, interrupt_before=["tools"]
    )

    # assertions

    # invoke graph w/o interrupt
    assert app.invoke(
        {"messages": ["what is weather in sf"]},
        {
            "configurable": {
                "thread_id": "1",
                "test_config_1": "foo",
                "test_config_2": "bar",
            },
        },
    ) == {
        "messages": [
            _AnyIdHumanMessage(content="what is weather in sf"),
            _AnyIdAIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_api",
                        "args": {"query": "query"},
                        "id": "tool_call123",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(
                content="result for query",
                name="search_api",
                tool_call_id="tool_call123",
            ),
            _AnyIdAIMessage(content="answer"),
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_1 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_1["test_config_1"] == "foo"
    assert chkpnt_metadata_1["test_config_2"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_1 = sync_checkpointer.list(config)
    for chkpnt_tuple in chkpnt_tuples_1:
        assert chkpnt_tuple.metadata["test_config_1"] == "foo"
        assert chkpnt_tuple.metadata["test_config_2"] == "bar"

    # invoke graph, but interrupt before tool call
    app_w_interrupt.invoke(
        {"messages": ["what is weather in sf"]},
        {
            "configurable": {
                "thread_id": "2",
                "test_config_3": "foo",
                "test_config_4": "bar",
            },
        },
    )

    config = {"configurable": {"thread_id": "2"}}

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_2 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_2["test_config_3"] == "foo"
    assert chkpnt_metadata_2["test_config_4"] == "bar"

    # resume graph execution
    app_w_interrupt.invoke(
        input=None,
        config={
            "configurable": {
                "thread_id": "2",
                "test_config_3": "foo",
                "test_config_4": "bar",
            }
        },
    )

    # assert that checkpoint metadata contains the run's configurable fields
    chkpnt_metadata_3 = sync_checkpointer.get_tuple(config).metadata
    assert chkpnt_metadata_3["test_config_3"] == "foo"
    assert chkpnt_metadata_3["test_config_4"] == "bar"

    # Verify that all checkpoint metadata have the expected keys. This check
    # is needed because a run may have an arbitrary number of steps depending
    # on how the graph is constructed.
    chkpnt_tuples_2 = sync_checkpointer.list(config)
    for chkpnt_tuple in chkpnt_tuples_2:
        assert chkpnt_tuple.metadata["test_config_3"] == "foo"
        assert chkpnt_tuple.metadata["test_config_4"] == "bar"


def test_remove_message_via_state_update(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

    workflow = StateGraph(state_schema=Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]
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

    app = workflow.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke([HumanMessage(content="Hi")], config=config)
    app.update_state(config, values=[RemoveMessage(id=output[-1].id)])

    updated_state = app.get_state(config)

    assert len(updated_state.values) == 1
    assert updated_state.values[-1].content == "Hi"

    app.checkpointer.delete_thread(config["configurable"]["thread_id"])

    # Verify that the message was removed from the checkpointer
    assert app.checkpointer.get_tuple(config) is None
    assert [*app.get_state_history(config)] == []


def test_channel_values(sync_checkpointer: BaseCheckpointSaver) -> None:
    config = {"configurable": {"thread_id": "1"}}
    chain = NodeBuilder().subscribe_only("input").write_to("output")
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
        checkpointer=sync_checkpointer,
    )
    app.invoke({"input": 1, "ephemeral": "meow"}, config)
    assert sync_checkpointer.get(config)["channel_values"] == {"input": 1, "output": 1}


def test_store_injected(
    sync_checkpointer: BaseCheckpointSaver, sync_store: BaseStore
) -> None:
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
                (
                    namespace
                    if self.i is not None
                    and config["configurable"]["thread_id"] in (thread_1, thread_2)
                    else (f"foo_{self.i}", "bar")
                ),
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
    N = 50
    M = 1

    for i in range(N):
        builder.add_node(f"node_{i}", Node(i))
        builder.add_edge("__start__", f"node_{i}")

    graph = builder.compile(store=sync_store, checkpointer=sync_checkpointer)

    results = graph.batch(
        [{"count": 0}] * M,
        ([{"configurable": {"thread_id": str(uuid.uuid4())}}] * (M - 1))
        + [{"configurable": {"thread_id": thread_1}}],
    )
    result = results[-1]
    assert result == {"count": N + 1}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": 0}
    assert len(sync_store.search(namespace)) == 1
    # Check results after another turn of the same thread
    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_1}})
    assert result == {"count": (N + 1) * 2}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {**doc, "from_thread": thread_1, "some_val": N + 1}
    assert len(sync_store.search(namespace)) == 1

    result = graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_2}})
    assert result == {"count": N + 1}
    returned_doc = sync_store.get(namespace, doc_id).value
    assert returned_doc == {
        **doc,
        "from_thread": thread_2,
        "some_val": 0,
    }  # Overwrites the whole doc
    assert len(sync_store.search(namespace)) == 1  # still overwriting the same one


def test_debug_retry(sync_checkpointer: BaseCheckpointSaver):
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    builder = StateGraph(State)
    builder.add_node("one", node("one"))
    builder.add_node("two", node("two"))
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)

    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    graph.invoke({"messages": []}, config=config, durability="async")

    # re-run step: 1
    target_config = next(
        c.parent_config
        for c in sync_checkpointer.list(config)
        if c.metadata["step"] == 1
    )
    update_config = graph.update_state(target_config, values=None)

    events = [
        *graph.stream(
            None, config=update_config, stream_mode="debug", durability="async"
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )

    checkpoint_history = {
        c.config["configurable"]["checkpoint_id"]: c
        for c in graph.get_state_history(config)
    }

    def lax_normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None
        return config["configurable"]

    for stream in checkpoint_events:
        stream_conf = lax_normalize_config(stream["config"])
        stream_parent_conf = lax_normalize_config(stream["parent_config"])
        assert stream_conf != stream_parent_conf

        # ensure the streamed checkpoint == checkpoint from checkpointer.list()
        history = checkpoint_history[stream["config"]["configurable"]["checkpoint_id"]]
        history_conf = lax_normalize_config(history.config)
        assert stream_conf == history_conf

        history_parent_conf = lax_normalize_config(history.parent_config)
        assert stream_parent_conf == history_parent_conf


def test_debug_subgraphs(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    parent = StateGraph(State)
    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))
    child.add_edge(START, "c_one")
    child.add_edge("c_one", "c_two")
    child.add_edge("c_two", END)

    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())
    parent.add_edge(START, "p_one")
    parent.add_edge("p_one", "p_two")
    parent.add_edge("p_two", END)

    graph = parent.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
            durability=durability,
        )
    ]

    checkpoint_events = list(
        reversed([e["payload"] for e in events if e["type"] == "checkpoint"])
    )
    if durability == "exit":
        checkpoint_events = checkpoint_events[:1]
    checkpoint_history = list(graph.get_state_history(config))

    assert len(checkpoint_events) == len(checkpoint_history)

    def lax_normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None
        return config["configurable"]

    for stream, history in zip(checkpoint_events, checkpoint_history):
        assert stream["values"] == history.values
        assert stream["next"] == list(history.next)
        assert lax_normalize_config(stream["config"]) == lax_normalize_config(
            history.config
        )
        assert lax_normalize_config(stream["parent_config"]) == lax_normalize_config(
            history.parent_config
        )

        assert len(stream["tasks"]) == len(history.tasks)
        for stream_task, history_task in zip(stream["tasks"], history.tasks):
            assert stream_task["id"] == history_task.id
            assert stream_task["name"] == history_task.name
            assert stream_task["interrupts"] == history_task.interrupts
            assert stream_task.get("error") == history_task.error
            assert stream_task.get("state") == history_task.state


def test_debug_nested_subgraphs(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
    from collections import defaultdict

    class State(TypedDict):
        messages: Annotated[list[str], operator.add]

    def node(name):
        def _node(state: State):
            return {"messages": [f"entered {name} node"]}

        return _node

    grand_parent = StateGraph(State)
    parent = StateGraph(State)
    child = StateGraph(State)

    child.add_node("c_one", node("c_one"))
    child.add_node("c_two", node("c_two"))
    child.add_edge(START, "c_one")
    child.add_edge("c_one", "c_two")
    child.add_edge("c_two", END)

    parent.add_node("p_one", node("p_one"))
    parent.add_node("p_two", child.compile())
    parent.add_edge(START, "p_one")
    parent.add_edge("p_one", "p_two")
    parent.add_edge("p_two", END)

    grand_parent.add_node("gp_one", node("gp_one"))
    grand_parent.add_node("gp_two", parent.compile())
    grand_parent.add_edge(START, "gp_one")
    grand_parent.add_edge("gp_one", "gp_two")
    grand_parent.add_edge("gp_two", END)

    graph = grand_parent.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}
    events = [
        *graph.stream(
            {"messages": []},
            config=config,
            stream_mode="debug",
            subgraphs=True,
            durability=durability,
        )
    ]

    stream_ns: dict[tuple, dict] = defaultdict(list)
    for ns, e in events:
        if e["type"] == "checkpoint":
            stream_ns[ns].append(e["payload"])

    assert list(stream_ns.keys()) == [
        (),
        (AnyStr("gp_two:"),),
        (AnyStr("gp_two:"), AnyStr("p_two:")),
    ]

    history_ns = {
        ns: list(
            graph.get_state_history(
                {"configurable": {"thread_id": "1", "checkpoint_ns": "|".join(ns)}}
            )
        )[::-1]
        for ns in stream_ns.keys()
    }

    def normalize_config(config: Optional[dict]) -> Optional[dict]:
        if config is None:
            return None

        clean_config = {}
        clean_config["thread_id"] = config["configurable"]["thread_id"]
        clean_config["checkpoint_id"] = config["configurable"]["checkpoint_id"]
        clean_config["checkpoint_ns"] = config["configurable"]["checkpoint_ns"]
        if "checkpoint_map" in config["configurable"]:
            clean_config["checkpoint_map"] = config["configurable"]["checkpoint_map"]

        return clean_config

    for checkpoint_events, checkpoint_history, ns in zip(
        stream_ns.values(), history_ns.values(), stream_ns.keys()
    ):
        if durability == "exit":
            checkpoint_events = checkpoint_events[-1:]
            if ns:  # Save no checkpoints for subgraphs when durability="exit"
                assert not checkpoint_history
                continue
        assert len(checkpoint_events) == len(checkpoint_history)
        for stream, history in zip(checkpoint_events, checkpoint_history):
            assert stream["values"] == history.values
            assert stream["next"] == list(history.next)
            assert normalize_config(stream["config"]) == normalize_config(
                history.config
            )
            assert normalize_config(stream["parent_config"]) == normalize_config(
                history.parent_config
            )

            assert len(stream["tasks"]) == len(history.tasks)
            for stream_task, history_task in zip(stream["tasks"], history.tasks):
                assert stream_task["id"] == history_task.id
                assert stream_task["name"] == history_task.name
                assert stream_task["interrupts"] == history_task.interrupts
                assert stream_task.get("error") == history_task.error
                assert stream_task.get("state") == history_task.state


@pytest.mark.parametrize("subgraph_persist", [True, False])
def test_parent_command(
    sync_checkpointer: BaseCheckpointSaver, subgraph_persist: bool
) -> None:
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import tool

    @tool(return_direct=True)
    def get_user_name() -> Command:
        """Retrieve user name"""
        return Command(update={"user_name": "Meow"}, graph=Command.PARENT)

    subgraph_builder = StateGraph(MessagesState)
    subgraph_builder.add_node("tool", get_user_name)
    subgraph_builder.add_edge(START, "tool")
    subgraph = subgraph_builder.compile(checkpointer=subgraph_persist)

    class CustomParentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        # this key is not available to the child graph
        user_name: str

    builder = StateGraph(CustomParentState)
    builder.add_node("alice", subgraph)
    builder.add_edge(START, "alice")
    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke(
        {"messages": [("user", "get user name")]}, config, durability="exit"
    ) == {
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
            "step": 1,
            "parents": {},
        },
        created_at=AnyStr(),
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


def test_interrupt_subgraph(sync_checkpointer: BaseCheckpointSaver):
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
    graph = builder.compile(checkpointer=sync_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"baz": ""}, thread1)
    # Resume with answer
    assert graph.invoke(Command(resume="bar"), thread1)


@pytest.mark.parametrize("resume_style", ["null", "map"])
def test_interrupt_multiple(
    sync_checkpointer: BaseCheckpointSaver, resume_style: Literal["null", "map"]
):
    class State(TypedDict):
        my_key: Annotated[str, operator.add]

    def node(s: State) -> State:
        answer = interrupt({"value": 1})
        answer2 = interrupt({"value": 2})
        return {"my_key": answer + " " + answer2}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=sync_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    result = [e for e in graph.stream({"my_key": "DE", "market": "DE"}, thread1)]
    assert result == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 1},
                    id=AnyStr(),
                ),
            )
        }
    ]

    result = [
        event
        for event in graph.stream(
            Command(
                resume="answer 1"
                if resume_style == "null"
                else {result[0]["__interrupt__"][0].id: "answer 1"},
                update={"my_key": " foofoo "},
            ),
            thread1,
        )
    ]
    assert result == [
        {
            "__interrupt__": (
                Interrupt(
                    value={"value": 2},
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(
                resume="answer 2"
                if resume_style == "null"
                else {result[0]["__interrupt__"][0].id: "answer 2"}
            ),
            thread1,
            stream_mode="values",
        )
    ] == [
        {"my_key": "DE foofoo "},
        {"my_key": "DE foofoo answer 1 answer 2"},
    ]


def test_interrupt_loop(sync_checkpointer: BaseCheckpointSaver):
    class State(TypedDict):
        age: int
        other: str

    def ask_age(s: State):
        """Ask an expert for help."""
        question = "How old are you?"
        value = None
        for _ in range(10):
            value: str = interrupt(question)
            if not value.isdigit() or int(value) < 18:
                question = "invalid response"
                value = None
            else:
                break

        return {"age": int(value)}

    builder = StateGraph(State)
    builder.add_node("node", ask_age)
    builder.add_edge(START, "node")

    graph = builder.compile(checkpointer=sync_checkpointer)
    thread1 = {"configurable": {"thread_id": "1"}}

    assert [e for e in graph.stream({"other": ""}, thread1)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="How old are you?",
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(resume="13"),
            thread1,
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="invalid response",
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [
        event
        for event in graph.stream(
            Command(resume="15"),
            thread1,
        )
    ] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="invalid response",
                    id=AnyStr(),
                ),
            )
        }
    ]

    assert [event for event in graph.stream(Command(resume="19"), thread1)] == [
        {"node": {"age": 19}},
    ]


def test_interrupt_functional(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        return {"a": state["a"] + "bar", "b": state["b"]}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        value = interrupt("Provide value for bar:")
        bar_input = {**fut_foo.result(), "b": value}
        fut_bar = bar(bar_input)
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            )
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar", "b": "bar"}


def test_interrupt_task_functional(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    @task
    def foo(state: dict) -> dict:
        return {"a": state["a"] + "foo"}

    @task
    def bar(state: dict) -> dict:
        value = interrupt("Provide value for bar:")
        return {"a": state["a"] + value}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        fut_foo = foo(inputs)
        fut_bar = bar(fut_foo.result())
        return fut_bar.result()

    config = {"configurable": {"thread_id": "1"}}
    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            ),
        ]
    }
    # Resume with an answer
    res = graph.invoke(Command(resume="bar"), config)
    assert res == {"a": "foobar"}

    # Test that we can interrupt the same task multiple times
    config = {"configurable": {"thread_id": "2"}}

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(inputs: dict) -> dict:
        foo_result = foo(inputs).result()
        bar_result = bar(foo_result).result()
        baz_result = bar(bar_result).result()
        return baz_result

    # First run, interrupted at bar
    assert graph.invoke({"a": ""}, config) == {
        "__interrupt__": [
            Interrupt(
                value="Provide value for bar:",
                id=AnyStr(),
            ),
        ]
    }
    # Provide resumes
    graph.invoke(Command(resume="bar"), config)
    assert graph.invoke(Command(resume="baz"), config) == {"a": "foobarbaz"}


def test_command_with_static_breakpoints(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    """Test that we can use Command to resume and update with static breakpoints."""

    class State(TypedDict):
        """The graph state."""

        foo: str

    def node1(state: State):
        return {
            "foo": state["foo"] + "|node-1",
        }

    def node2(state: State):
        return {
            "foo": state["foo"] + "|node-2",
        }

    builder = StateGraph(State)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["node1"])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    graph.invoke({"foo": "abc"}, config)
    result = graph.invoke(Command(resume="node1"), config)
    assert result == {"foo": "abc|node-1|node-2"}


def test_multistep_plan(sync_checkpointer: BaseCheckpointSaver):
    from langchain_core.messages import AnyMessage

    class State(TypedDict, total=False):
        plan: list[Union[str, list[str]]]
        messages: Annotated[list[AnyMessage], add_messages]

    def planner(state: State):
        if state.get("plan") is None:
            # create plan somehow
            plan = ["step1", ["step2", "step3"], "step4"]
            # pick the first step to execute next
            first_step, *plan = plan
            # put the rest of plan in state
            return Command(goto=first_step, update={"plan": plan})
        elif state["plan"]:
            # go to the next step of the plan
            next_step, *next_plan = state["plan"]
            return Command(goto=next_step, update={"plan": next_plan})
        else:
            # the end of the plan
            pass

    def step1(state: State):
        return Command(goto="planner", update={"messages": [("human", "step1")]})

    def step2(state: State):
        return Command(goto="planner", update={"messages": [("human", "step2")]})

    def step3(state: State):
        return Command(goto="planner", update={"messages": [("human", "step3")]})

    def step4(state: State):
        return Command(goto="planner", update={"messages": [("human", "step4")]})

    builder = StateGraph(State)
    builder.add_node(planner)
    builder.add_node(step1)
    builder.add_node(step2)
    builder.add_node(step3)
    builder.add_node(step4)
    builder.add_edge(START, "planner")
    graph = builder.compile(checkpointer=sync_checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    assert graph.invoke({"messages": [("human", "start")]}, config) == {
        "messages": [
            _AnyIdHumanMessage(content="start"),
            _AnyIdHumanMessage(content="step1"),
            _AnyIdHumanMessage(content="step2"),
            _AnyIdHumanMessage(content="step3"),
            _AnyIdHumanMessage(content="step4"),
        ],
        "plan": [],
    }


def test_command_goto_with_static_breakpoints(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    """Use Command goto with static breakpoints."""

    class State(TypedDict):
        """The graph state."""

        foo: Annotated[str, operator.add]

    def node1(state: State):
        return {
            "foo": "|node-1",
        }

    def node2(state: State):
        return {
            "foo": "|node-2",
        }

    builder = StateGraph(State)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=sync_checkpointer, interrupt_before=["node1"])

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Start the graph and interrupt at the first node
    graph.invoke({"foo": "abc"}, config)
    result = graph.invoke(Command(goto=["node2"]), config)
    assert result == {"foo": "abc|node-1|node-2|node-2"}


def test_multiple_interrupt_state_persistence(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    """Test that state is preserved correctly across multiple interrupts."""

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]

    def interruptible_node(state: State):
        first = interrupt("First interrupt")
        second = interrupt("Second interrupt")
        return {"steps": [first, second]}

    builder = StateGraph(State)
    builder.add_node("node", interruptible_node)
    builder.add_edge(START, "node")

    app = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First execution - should hit first interrupt
    app.invoke({"steps": []}, config)

    # State should still be empty since node hasn't returned
    state = app.get_state(config)
    assert state.values == {"steps": []}

    # Resume after first interrupt - should hit second interrupt
    app.invoke(Command(resume="step1"), config)

    # State should still be empty since node hasn't returned
    state = app.get_state(config)
    assert state.values == {"steps": []}

    # Resume after second interrupt - node should complete
    result = app.invoke(Command(resume="step2"), config)

    # Now state should contain both steps since node returned
    assert result["steps"] == ["step1", "step2"]
    state = app.get_state(config)
    assert state.values["steps"] == ["step1", "step2"]


def test_checkpoint_recovery(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
):
    """Test recovery from checkpoints after failures."""

    class State(TypedDict):
        steps: Annotated[list[str], operator.add]
        attempt: int  # Track number of attempts

    def failing_node(state: State):
        # Fail on first attempt, succeed on retry
        if state["attempt"] == 1:
            raise RuntimeError("Simulated failure")
        return {"steps": ["node1"]}

    def second_node(state: State):
        return {"steps": ["node2"]}

    builder = StateGraph(State)
    builder.add_node("node1", failing_node)
    builder.add_node("node2", second_node)
    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")

    graph = builder.compile(checkpointer=sync_checkpointer)
    config = {"configurable": {"thread_id": "1"}}

    # First attempt should fail
    with pytest.raises(RuntimeError):
        graph.invoke(
            {"steps": ["start"], "attempt": 1},
            config,
            durability=durability,
        )

    # Verify checkpoint state
    state = graph.get_state(config)
    assert state is not None
    assert state.values == {"steps": ["start"], "attempt": 1}  # input state saved
    assert state.next == ("node1",)  # Should retry failed node
    assert "RuntimeError('Simulated failure')" in state.tasks[0].error

    # Retry with updated attempt count
    result = graph.invoke(
        {"steps": [], "attempt": 2}, config, durability=durability
    )
    assert result == {"steps": ["start", "node1", "node2"], "attempt": 2}

    # Verify checkpoint history shows both attempts
    history = list(graph.get_state_history(config))
    if durability != "exit":
        assert len(history) == 6  # Initial + failed attempt + successful attempt
    else:
        assert len(history) == 2  # error + success

    # Verify the error was recorded in checkpoint
    failed_checkpoint = next(c for c in history if c.tasks and c.tasks[0].error)
    assert "RuntimeError('Simulated failure')" in failed_checkpoint.tasks[0].error

    # Verify delete leaves it empty
    graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    assert graph.checkpointer.get_tuple(config) is None
    assert [*graph.get_state_history(config)] == []


def test_falsy_return_from_task(sync_checkpointer: BaseCheckpointSaver):
    """Test with a falsy return from a task."""

    @task
    def falsy_task() -> bool:
        return False

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        """React tool."""
        falsy_task().result()
        interrupt("test")

    configurable = {"configurable": {"thread_id": uuid.uuid4()}}
    assert [
        chunk
        for chunk in graph.stream(
            {"a": 5}, configurable, stream_mode="debug", durability="exit"
        )
    ] == [
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                },
                "next": [
                    "graph",
                ],
                "parent_config": None,
                "tasks": [
                    {
                        "id": AnyStr(),
                        "interrupts": (),
                        "name": "graph",
                        "state": None,
                    },
                ],
                "values": None,
            },
            "step": -1,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": {
                    "a": 5,
                },
                "name": "graph",
                "triggers": ("__start__",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": (
                    (),
                    {},
                ),
                "name": "falsy_task",
                "triggers": ("__pregel_push",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [],
                "name": "falsy_task",
                "result": [
                    (
                        "__return__",
                        False,
                    ),
                ],
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [
                    {
                        "id": AnyStr(),
                        "value": "test",
                    },
                ],
                "name": "graph",
                "result": [],
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
    ]
    assert [
        c
        for c in graph.stream(
            Command(resume="123"),
            configurable,
            stream_mode="debug",
            durability="exit",
        )
    ] == [
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "input",
                    "step": -1,
                },
                "next": [
                    "graph",
                ],
                "parent_config": None,
                "tasks": [
                    {
                        "id": AnyStr(),
                        "interrupts": (
                            {
                                "id": AnyStr(),
                                "value": "test",
                            },
                        ),
                        "name": "graph",
                        "state": None,
                    },
                ],
                "values": None,
            },
            "step": -1,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": {
                    "a": 5,
                },
                "name": "graph",
                "triggers": ("__start__",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "id": AnyStr(),
                "input": (
                    (),
                    {},
                ),
                "name": "falsy_task",
                "triggers": ("__pregel_push",),
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task",
        },
        {
            "payload": {
                "error": None,
                "id": AnyStr(),
                "interrupts": [],
                "name": "graph",
                "result": [
                    (
                        "__end__",
                        None,
                    ),
                ],
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "task_result",
        },
        {
            "payload": {
                "config": {
                    "configurable": {
                        "checkpoint_id": AnyStr(),
                        "checkpoint_ns": "",
                        "thread_id": AnyStr(),
                    },
                },
                "metadata": {
                    "parents": {},
                    "source": "loop",
                    "step": 0,
                },
                "next": [],
                "parent_config": None,
                "tasks": [],
                "values": None,
            },
            "step": 0,
            "timestamp": AnyStr(),
            "type": "checkpoint",
        },
    ]


def test_multiple_interrupts_functional(
    sync_checkpointer: BaseCheckpointSaver
):
    """Test multiple interrupts with functional API."""
    counter = 0

    @task
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=sync_checkpointer)
    def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 2, 3]:
            values.extend([double(idx).result(), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    result = graph.invoke(Command(resume="c"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 4, "b", 6, "c"],
    }
    assert counter == 3


def test_multiple_interrupts_functional_cache(
    sync_checkpointer: BaseCheckpointSaver, cache: BaseCache
):
    """Test multiple interrupts with functional API."""
    counter = 0

    @task(cache_policy=CachePolicy())
    def double(x: int) -> int:
        """Increment the counter."""
        nonlocal counter
        counter += 1
        return 2 * x

    @entrypoint(checkpointer=sync_checkpointer, cache=cache)
    def graph(state: dict) -> dict:
        """React tool."""

        values = []

        for idx in [1, 1, 2, 2, 3, 3]:
            values.extend([double(idx).result(), interrupt({"a": "boo"})])

        return {"values": values}

    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    # should all be cached now
    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 3

    # clear cache
    double.clear_cache(cache)

    # should recompute now
    configurable = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({}, configurable)
    graph.invoke(Command(resume="a"), configurable)
    graph.invoke(Command(resume="b"), configurable)
    graph.invoke(Command(resume="c"), configurable)
    graph.invoke(Command(resume="d"), configurable)
    graph.invoke(Command(resume="e"), configurable)
    result = graph.invoke(Command(resume="f"), configurable)
    # `double` value should be cached appropriately when used w/ `interrupt`
    assert result == {
        "values": [2, "a", 2, "b", 4, "c", 4, "d", 6, "e", 6, "f"],
    }
    assert counter == 6


def test_double_interrupt_subgraph(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    class AgentState(TypedDict):
        input: str

    def node_1(state: AgentState):
        result = interrupt("interrupt node 1")
        return {"input": result}

    def node_2(state: AgentState):
        result = interrupt("interrupt node 2")
        return {"input": result}

    subgraph_builder = (
        StateGraph(AgentState)
        .add_node("node_1", node_1)
        .add_node("node_2", node_2)
        .add_edge(START, "node_1")
        .add_edge("node_1", "node_2")
        .add_edge("node_2", END)
    )

    # invoke the sub graph
    subgraph = subgraph_builder.compile(checkpointer=sync_checkpointer)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    assert [c for c in subgraph.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    id=AnyStr(),
                ),
            )
        },
    ]
    # resume from the first interrupt
    assert [c for c in subgraph.stream(Command(resume="123"), thread)] == [
        {
            "node_1": {"input": "123"},
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 2",
                    id=AnyStr(),
                ),
            )
        },
    ]
    # resume from the second interrupt
    assert [c for c in subgraph.stream(Command(resume="123"), thread)] == [
        {
            "node_2": {"input": "123"},
        },
    ]

    subgraph = subgraph_builder.compile()

    def invoke_sub_agent(state: AgentState):
        return subgraph.invoke(state)

    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    parent_agent = (
        StateGraph(AgentState)
        .add_node("invoke_sub_agent", invoke_sub_agent)
        .add_edge(START, "invoke_sub_agent")
        .add_edge("invoke_sub_agent", END)
        .compile(checkpointer=sync_checkpointer)
    )

    assert [c for c in parent_agent.stream({"input": "test"}, thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 1",
                    id=AnyStr(),
                ),
            )
        },
    ]

    # resume from the first interrupt
    assert [c for c in parent_agent.stream(Command(resume=True), thread)] == [
        {
            "__interrupt__": (
                Interrupt(
                    value="interrupt node 2",
                    id=AnyStr(),
                ),
            )
        }
    ]

    # resume from 2nd interrupt
    assert [c for c in parent_agent.stream(Command(resume=True), thread)] == [
        {
            "invoke_sub_agent": {"input": True},
        },
    ]


def test_multiple_subgraphs(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output).add_node(add).add_edge(START, "add").compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    def call_same_subgraph(state):
        result = add_subgraph.invoke(state)
        another_result = add_subgraph.invoke({"a": result["result"], "b": 10})
        return another_result

    parent_call_same_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke({"a": 2, "b": 3}, config) == {"result": 15}

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    def call_multiple_subgraphs(state):
        add_result = add_subgraph.invoke(state)
        multiply_result = multiply_subgraph.invoke(state)
        return {
            "add_result": add_result["result"],
            "multiply_result": multiply_result["result"],
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


def test_multiple_subgraphs_functional(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    # Define addition subgraph
    @entrypoint()
    def add(inputs: tuple[int, int]):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    def multiply_task(a, b):
        return a * b

    @entrypoint()
    def multiply(inputs: tuple[int, int]):
        return multiply_task(*inputs).result()

    # Test calling the same subgraph multiple times
    @task
    def call_same_subgraph(a, b):
        result = add.invoke([a, b])
        another_result = add.invoke([result, 10])
        return another_result

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_same_subgraph(inputs):
        return call_same_subgraph(*inputs).result()

    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    def call_multiple_subgraphs(a, b):
        add_result = add.invoke([a, b])
        multiply_result = multiply.invoke([a, b])
        return [add_result, multiply_result]

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


def test_multiple_subgraphs_mixed_entrypoint(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    """Test calling multiple StateGraph subgraphs from an entrypoint."""
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define the subgraphs
    def add(state):
        return {"result": state["a"] + state["b"]}

    add_subgraph = (
        StateGraph(State, output_schema=Output).add_node(add).add_edge(START, "add").compile()
    )

    def multiply(state):
        return {"result": state["a"] * state["b"]}

    multiply_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(multiply)
        .add_edge(START, "multiply")
        .compile()
    )

    # Test calling the same subgraph multiple times
    @task
    def call_same_subgraph(a, b):
        result = add_subgraph.invoke({"a": a, "b": b})["result"]
        another_result = add_subgraph.invoke({"a": result, "b": 10})["result"]
        return another_result

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_same_subgraph(inputs):
        return call_same_subgraph(*inputs).result()

    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke([2, 3], config) == 15

    # Test calling multiple subgraphs
    @task
    def call_multiple_subgraphs(a, b):
        add_result = add_subgraph.invoke({"a": a, "b": b})["result"]
        multiply_result = multiply_subgraph.invoke({"a": a, "b": b})["result"]
        return [add_result, multiply_result]

    @entrypoint(checkpointer=sync_checkpointer)
    def parent_call_multiple_subgraphs(inputs):
        return call_multiple_subgraphs(*inputs).result()

    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke([2, 3], config) == [5, 6]


def test_multiple_subgraphs_mixed_state_graph(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    """Test calling multiple entrypoint "subgraphs" from a StateGraph."""
    class State(TypedDict):
        a: int
        b: int

    class Output(TypedDict):
        result: int

    # Define addition subgraph
    @entrypoint()
    def add(inputs: tuple[int, int]):
        a, b = inputs
        return a + b

    # Define multiplication subgraph using tasks
    @task
    def multiply_task(a, b):
        return a * b

    @entrypoint()
    def multiply(inputs: tuple[int, int]):
        return multiply_task(*inputs).result()

    # Test calling the same subgraph multiple times
    def call_same_subgraph(state):
        result = add.invoke([state["a"], state["b"]])
        another_result = add.invoke([result, 10])
        return {"result": another_result}

    parent_call_same_subgraph = (
        StateGraph(State, output_schema=Output)
        .add_node(call_same_subgraph)
        .add_edge(START, "call_same_subgraph")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "1"}}
    assert parent_call_same_subgraph.invoke({"a": 2, "b": 3}, config) == {"result": 15}

    # Test calling multiple subgraphs
    class Output(TypedDict):
        add_result: int
        multiply_result: int

    def call_multiple_subgraphs(state):
        add_result = add.invoke([state["a"], state["b"]])
        multiply_result = multiply.invoke([state["a"], state["b"]])
        return {
            "add_result": add_result,
            "multiply_result": multiply_result,
        }

    parent_call_multiple_subgraphs = (
        StateGraph(State, output_schema=Output)
        .add_node(call_multiple_subgraphs)
        .add_edge(START, "call_multiple_subgraphs")
        .compile(checkpointer=sync_checkpointer)
    )
    config = {"configurable": {"thread_id": "2"}}
    assert parent_call_multiple_subgraphs.invoke({"a": 2, "b": 3}, config) == {
        "add_result": 5,
        "multiply_result": 6,
    }


def test_multiple_subgraphs_checkpointer(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    class SubgraphState(TypedDict):
        sub_counter: Annotated[int, operator.add]

    def subgraph_node(state):
        return {"sub_counter": 2}

    sub_graph_1 = (
        StateGraph(SubgraphState)
        .add_node(subgraph_node)
        .add_edge(START, "subgraph_node")
        .compile(checkpointer=True)
    )

    class OtherSubgraphState(TypedDict):
        other_sub_counter: Annotated[int, operator.add]

    def other_subgraph_node(state):
        return {"other_sub_counter": 3}

    sub_graph_2 = (
        StateGraph(OtherSubgraphState)
        .add_node(other_subgraph_node)
        .add_edge(START, "other_subgraph_node")
        .compile()
    )

    class ParentState(TypedDict):
        parent_counter: int

    def parent_node(state):
        result = sub_graph_1.invoke({"sub_counter": state["parent_counter"]})
        other_result = sub_graph_2.invoke({"other_sub_counter": result["sub_counter"]})
        return {"parent_counter": other_result["other_sub_counter"]}

    parent_graph = (
        StateGraph(ParentState)
        .add_node(parent_node)
        .add_edge(START, "parent_node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert parent_graph.invoke({"parent_counter": 0}, config) == {"parent_counter": 5}
    assert parent_graph.invoke({"parent_counter": 0}, config) == {"parent_counter": 7}
    config = {"configurable": {"thread_id": "2"}}
    assert [
        c
        for c in parent_graph.stream(
            {"parent_counter": 0}, config, subgraphs=True, stream_mode="updates"
        )
    ] == [
        (("parent_node",), {"subgraph_node": {"sub_counter": 2}}),
        (
            (AnyStr("parent_node:"), "1"),
            {"other_subgraph_node": {"other_sub_counter": 3}},
        ),
        ((), {"parent_node": {"parent_counter": 5}}),
    ]
    assert [
        c
        for c in parent_graph.stream(
            {"parent_counter": 0}, config, subgraphs=True, stream_mode="updates"
        )
    ] == [
        (("parent_node",), {"subgraph_node": {"sub_counter": 2}}),
        (
            (AnyStr("parent_node:"), "1"),
            {"other_subgraph_node": {"other_sub_counter": 3}},
        ),
        ((), {"parent_node": {"parent_counter": 7}}),
    ]


def test_entrypoint_with_return_and_save(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test entrypoint with return and save."""
    previous_ = None

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(msg: str, *, previous: Any) -> entrypoint.final[int, list[str]]:
        nonlocal previous_
        previous_ = previous
        previous = previous or []
        return entrypoint.final(value=len(previous), save=previous + [msg])

    assert foo.get_output_schema().model_json_schema() == {
        "title": "LangGraphOutput",
        "type": "integer",
    }

    config = {"configurable": {"thread_id": "1"}}
    assert foo.invoke("hello", config) == 0
    assert previous_ is None
    assert foo.invoke("goodbye", config) == 1
    assert previous_ == ["hello"]
    assert foo.invoke("definitely", config) == 2
    assert previous_ == ["hello", "goodbye"]


def test_overriding_injectable_args_with_tasks(sync_store: BaseStore) -> None:
    """Test overriding injectable args in tasks."""

    @task
    def foo(store: BaseStore, writer: StreamWriter, value: Any) -> None:
        assert store is value
        assert writer is value

    @entrypoint(store=sync_store)
    def main(inputs, store: BaseStore) -> str:
        assert store is not None
        foo(store=None, writer=None, value=None).result()
        foo(store="hello", writer="hello", value="hello").result()
        return "OK"

    assert main.invoke({}) == "OK"


def test_stream_messages_dedupe_state(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    from langchain_core.messages import AIMessage

    to_emit = [AIMessage("bye", id="1"), AIMessage("bye again", id="2")]

    def call_model(state):
        return {"messages": to_emit.pop(0)}

    def route(state):
        return Command(goto="node_2", graph=Command.PARENT)

    subgraph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_node(route)
        .add_edge(START, "call_model")
        .add_edge("call_model", "route")
        .compile()
    )

    graph = (
        StateGraph(MessagesState)
        .add_node("node_1", subgraph)
        .add_node("node_2", lambda state: state)
        .add_edge(START, "node_1")
        .compile(checkpointer=sync_checkpointer)
    )

    thread1 = {"configurable": {"thread_id": "1"}}

    chunks = [
        chunk
        for ns, chunk in graph.stream(
            {"messages": "hi"}, thread1, stream_mode="messages", subgraphs=True
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye", id="1")
    assert chunks[0][1]["langgraph_node"] == "call_model"

    chunks = [
        chunk
        for ns, chunk in graph.stream(
            {"messages": "hi again"},
            thread1,
            stream_mode="messages",
            subgraphs=True,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0] == AIMessage("bye again", id="2")
    assert chunks[0][1]["langgraph_node"] == "call_model"


def test_interrupt_subgraph_reenter_checkpointer_true(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    class SubgraphState(TypedDict):
        foo: str
        bar: str

    class ParentState(TypedDict):
        foo: str
        counter: int

    called = []
    bar_values = []

    def subnode_1(state: SubgraphState):
        called.append("subnode_1")
        bar_values.append(state.get("bar"))
        return {"foo": "subgraph_1"}

    def subnode_2(state: SubgraphState):
        called.append("subnode_2")
        value = interrupt("Provide value")
        value += "baz"
        return {"foo": "subgraph_2", "bar": value}

    subgraph = (
        StateGraph(SubgraphState)
        .add_node(subnode_1)
        .add_node(subnode_2)
        .add_edge(START, "subnode_1")
        .add_edge("subnode_1", "subnode_2")
        .compile(checkpointer=True)
    )

    def call_subgraph(state: ParentState):
        called.append("call_subgraph")
        return subgraph.invoke(state)

    def node(state: ParentState):
        called.append("parent")
        if state["counter"] < 1:
            return Command(
                goto="call_subgraph", update={"counter": state["counter"] + 1}
            )

        return {"foo": state["foo"] + "|" + "parent"}

    parent = (
        StateGraph(ParentState)
        .add_node(call_subgraph)
        .add_node(node)
        .add_edge(START, "call_subgraph")
        .add_edge("call_subgraph", "node")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    assert parent.invoke({"foo": "", "counter": 0}, config) == {
        "foo": "",
        "counter": 0,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    assert parent.invoke(Command(resume="bar"), config) == {
        "foo": "subgraph_2",
        "counter": 1,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    assert parent.invoke(Command(resume="qux"), config) == {
        "foo": "subgraph_2|parent",
        "counter": 1,
    }
    assert called == [
        "call_subgraph",
        "subnode_1",
        "subnode_2",
        "call_subgraph",
        "subnode_2",
        "parent",
        "call_subgraph",
        "subnode_1",
        "subnode_2",
        "call_subgraph",
        "subnode_2",
        "parent",
    ]

    # invoke parent again (new turn)
    assert parent.invoke({"foo": "meow", "counter": 0}, config) == {
        "foo": "meow",
        "counter": 0,
        "__interrupt__": [
            Interrupt(
                value="Provide value",
                id=AnyStr(),
            )
        ],
    }
    # confirm that we preserve the state values from the previous invocation
    assert bar_values == [None, "barbaz", "quxbaz"]


def test_parallel_interrupts(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: Optional[str] = Field(None, description="What the human said")
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def get_human_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_input=human_input,  # update child state
            human_inputs=[human_input],  # update parent state
        )

    child_graph_builder = StateGraph(ChildState)
    child_graph_builder.add_node("get_human_input", get_human_input)
    child_graph_builder.add_edge(START, "get_human_input")
    child_graph_builder.add_edge("get_human_input", END)
    child_graph = child_graph_builder.compile()

    # --- PARENT GRAPH ---

    class ParentState(BaseModel):
        prompts: list[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def assign_workers(state: ParentState):
        return [
            Send(
                "child_graph",
                dict(
                    prompt=prompt,
                ),
            )
            for prompt in state.prompts
        ]

    def cleanup(state: ParentState):
        assert len(state.human_inputs) == len(state.prompts)

    parent_graph_builder = StateGraph(ParentState)
    parent_graph_builder.add_node("child_graph", child_graph)
    parent_graph_builder.add_node("cleanup", cleanup)

    parent_graph_builder.add_conditional_edges(START, assign_workers, ["child_graph"])
    parent_graph_builder.add_edge("child_graph", "cleanup")
    parent_graph_builder.add_edge("cleanup", END)

    parent_graph = parent_graph_builder.compile(checkpointer=sync_checkpointer)

    # --- CLIENT INVOCATION ---

    thread_config = dict(
        configurable=dict(
            thread_id=str(uuid.uuid4()),
        )
    )
    current_input = dict(
        prompts=["a", "b"],
    )

    invokes = 0
    events: dict[int, list[dict]] = {}
    while invokes < 10:
        # reset interrupt
        invokes += 1
        events[invokes] = []
        current_interrupts: list[Interrupt] = []

        # start / resume the graph
        for event in parent_graph.stream(
            input=current_input,
            config=thread_config,
            stream_mode="updates",
        ):
            events[invokes].append(event)
            # handle the interrupt
            if "__interrupt__" in event:
                current_interrupts.extend(event["__interrupt__"])
                # assume that it breaks here, because it is an interrupt

        # get human input and resume
        if len(current_interrupts) > 0:
            current_input = Command(resume=f"Resume #{invokes}")

        # not more human input required, must be completed
        else:
            break
    else:
        assert False, "Detected infinite loop"

    assert invokes == 3
    assert len(events) == 3

    assert events[1] == UnsortedSequence(
        {
            "__interrupt__": (
                Interrupt(
                    value="a",
                    id=AnyStr(),
                ),
            )
        },
        {
            "__interrupt__": (
                Interrupt(
                    value="b",
                    id=AnyStr(),
                ),
            )
        },
    )
    assert events[2] in (
        UnsortedSequence(
            {
                "__interrupt__": (
                    Interrupt(
                        value="a",
                        id=AnyStr(),
                    ),
                )
            },
            {"child_graph": {"human_inputs": ["Resume #1"]}},
        ),
        UnsortedSequence(
            {
                "__interrupt__": (
                    Interrupt(
                        value="b",
                        id=AnyStr(),
                    ),
                )
            },
            {"child_graph": {"human_inputs": ["Resume #1"]}},
        ),
    )
    assert events[3] == UnsortedSequence(
        {
            "child_graph": {"human_inputs": ["Resume #1"]},
            "__metadata__": {"cached": True},
        },
        {"child_graph": {"human_inputs": ["Resume #2"]}},
        {"cleanup": None},
    )


def test_parallel_interrupts_double(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    # --- CHILD GRAPH ---

    class ChildState(BaseModel):
        prompt: str = Field(..., description="What is going to be asked to the user?")
        human_input: Optional[str] = Field(None, description="What the human said")
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def get_human_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_inputs=[human_input],  # update parent state
        )

    def get_dolphin_input(state: ChildState):
        human_input = interrupt(state.prompt)

        return dict(
            human_inputs=[human_input],  # update parent state
        )

    child_graph_builder = StateGraph(ChildState)
    child_graph_builder.add_node("get_human_input", get_human_input)
    child_graph_builder.add_node("get_dolphin_input", get_dolphin_input)
    child_graph_builder.add_edge(START, "get_human_input")
    child_graph_builder.add_edge(START, "get_dolphin_input")
    child_graph = child_graph_builder.compile()

    # --- PARENT GRAPH ---

    class ParentState(BaseModel):
        prompts: list[str] = Field(
            ..., description="What is going to be asked to the user?"
        )
        human_inputs: Annotated[list[str], operator.add] = Field(
            default_factory=list, description="All of my messages"
        )

    def assign_workers(state: ParentState):
        return [
            Send(
                "child_graph",
                dict(
                    prompt=prompt,
                ),
            )
            for prompt in state.prompts
        ]

    def cleanup(state: ParentState):
        assert len(state.human_inputs) == len(state.prompts) * 2

    parent_graph_builder = StateGraph(ParentState)
    parent_graph_builder.add_node("child_graph", child_graph)
    parent_graph_builder.add_node("cleanup", cleanup)

    parent_graph_builder.add_conditional_edges(START, assign_workers, ["child_graph"])
    parent_graph_builder.add_edge("child_graph", "cleanup")
    parent_graph_builder.add_edge("cleanup", END)

    parent_graph = parent_graph_builder.compile(checkpointer=sync_checkpointer)

    # --- CLIENT INVOCATION ---

    thread_config = dict(
        configurable=dict(
            thread_id=str(uuid.uuid4()),
        )
    )
    current_input = dict(
        prompts=["a", "b"],
    )

    invokes = 0
    events: dict[int, list[dict]] = {}
    while invokes < 10:
        # reset interrupt
        invokes += 1
        events[invokes] = []
        current_interrupts: list[Interrupt] = []

        # start / resume the graph
        for event in parent_graph.stream(
            input=current_input,
            config=thread_config,
            stream_mode="updates",
        ):
            events[invokes].append(event)
            # handle the interrupt
            if "__interrupt__" in event:
                current_interrupts.extend(event["__interrupt__"])
                # assume that it breaks here, because it is an interrupt

        # get human input and resume
        if len(current_interrupts) > 0:
            current_input = Command(resume=f"Resume #{invokes}")

        # not more human input required, must be completed
        else:
            break
    else:
        assert False, "Detected infinite loop"

    assert invokes == 5
    assert len(events) == 5


def test_bulk_state_updates(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    class State(TypedDict):
        foo: str
        baz: str

    def node_a(state: State) -> State:
        return {"foo": "bar"}

    def node_b(state: State) -> State:
        return {"baz": "qux"}

    graph = (
        StateGraph(State)
        .add_node("node_a", node_a)
        .add_node("node_b", node_b)
        .add_edge(START, "node_a")
        .add_edge("node_a", "node_b")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}

    # First update with node_a
    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "bar"}, as_node="node_a"),
            ]
        ],
    )

    # Then bulk update with both nodes
    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "updated"}, as_node="node_a"),
                StateUpdate(values={"baz": "new"}, as_node="node_b"),
            ]
        ],
    )

    state = graph.get_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    # Check if there are only two checkpoints
    checkpoints = list(sync_checkpointer.list(config))
    assert len(checkpoints) == 2

    # perform multiple steps at the same time
    config = {"configurable": {"thread_id": "2"}}

    graph.bulk_update_state(
        config,
        [
            [
                StateUpdate(values={"foo": "bar"}, as_node="node_a"),
            ],
            [
                StateUpdate(values={"foo": "updated"}, as_node="node_a"),
                StateUpdate(values={"baz": "new"}, as_node="node_b"),
            ],
        ],
    )

    state = graph.get_state(config)
    assert state.values == {"foo": "updated", "baz": "new"}

    checkpoints = list(sync_checkpointer.list(config))
    assert len(checkpoints) == 2

    # Should raise error if updating without as_node
    with pytest.raises(InvalidUpdateError):
        graph.bulk_update_state(
            config,
            [
                [
                    StateUpdate(values={"foo": "error"}, as_node=None),
                    StateUpdate(values={"bar": "error"}, as_node=None),
                ]
            ],
        )

    # Should raise if no updates are provided
    with pytest.raises(ValueError, match="No supersteps provided"):
        graph.bulk_update_state(config, [])

    # Should raise if no updates are provided
    with pytest.raises(ValueError, match="No updates provided"):
        graph.bulk_update_state(config, [[], []])

    # Should raise if __end__ or __copy__ update is applied in bulk
    with pytest.raises(InvalidUpdateError):
        graph.bulk_update_state(
            config,
            [
                [
                    StateUpdate(values=None, as_node="__end__"),
                    StateUpdate(values=None, as_node="__copy__"),
                ],
            ],
        )


def test_update_as_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        foo: str

    def agent(state: State) -> State:
        return {"foo": "agent"}

    def tool(state: State) -> State:
        return {"foo": "tool"}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("tool", tool)
        .add_edge(START, "agent")
        .add_edge("agent", "tool")
        .compile(checkpointer=sync_checkpointer)
    )

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "tool"}

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {"foo": "tool"}

    def map_snapshot(i: StateSnapshot) -> dict:
        return {
            "values": i.values,
            "next": i.next,
            "step": i.metadata.get("step"),
        }

    history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "1"}})
    ]

    graph.bulk_update_state(
        {"configurable": {"thread_id": "2"}},
        [
            # First turn
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent"}, "agent")],
            [StateUpdate({"foo": "tool"}, "tool")],
            # Second turn
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent"}, "agent")],
            [StateUpdate({"foo": "tool"}, "tool")],
        ],
    )

    state = graph.get_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "tool"}

    new_history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert [new_history[0], new_history[4]] == history


def test_batch_update_as_input(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    class State(TypedDict):
        foo: str
        tasks: Annotated[list[int], operator.add]

    def agent(state: State) -> State:
        return {"foo": "agent"}

    def map(state: State) -> Command["task"]:
        return Command(
            goto=[
                Send("task", {"index": 0}),
                Send("task", {"index": 1}),
                Send("task", {"index": 2}),
            ],
            update={"foo": "map"},
        )

    def task(state: dict) -> State:
        return {"tasks": [state["index"]]}

    graph = (
        StateGraph(State)
        .add_node("agent", agent)
        .add_node("map", map)
        .add_node("task", task)
        .add_edge(START, "agent")
        .add_edge("agent", "map")
        .compile(checkpointer=sync_checkpointer)
    )

    assert graph.invoke(
        {"foo": "input"},
        {"configurable": {"thread_id": "1"}},
        durability=durability,
    ) == {
        "foo": "map",
        "tasks": [0, 1, 2],
    }

    def map_snapshot(i: StateSnapshot) -> dict:
        return {
            "values": i.values,
            "next": i.next,
            "step": i.metadata.get("step"),
            "tasks": [t.name for t in i.tasks],
        }

    history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "1"}})
    ]

    graph.bulk_update_state(
        {"configurable": {"thread_id": "2"}},
        [
            [StateUpdate({"foo": "input"}, "__input__")],
            [StateUpdate({"foo": "input"}, "__start__")],
            [StateUpdate({"foo": "agent", "tasks": []}, "agent")],
            [
                StateUpdate(
                    Command(
                        goto=[
                            Send("task", {"index": 0}),
                            Send("task", {"index": 1}),
                            Send("task", {"index": 2}),
                        ],
                        update={"foo": "map"},
                    ),
                    "map",
                )
            ],
            [
                StateUpdate({"tasks": [0]}, "task"),
                StateUpdate({"tasks": [1]}, "task"),
                StateUpdate({"tasks": [2]}, "task"),
            ],
        ],
    )

    state = graph.get_state({"configurable": {"thread_id": "2"}})
    assert state.values == {"foo": "map", "tasks": [0, 1, 2]}

    new_history = [
        map_snapshot(s)
        for s in graph.get_state_history({"configurable": {"thread_id": "2"}})
    ]

    if durability != "exit":
        assert new_history == history
    else:
        assert new_history[:1] == history


def test_multi_resume(
    sync_checkpointer: BaseCheckpointSaver
) -> None:
    class ChildState(TypedDict):
        prompt: str
        human_input: str
        human_inputs: list[str]

    def get_human_input(state: ChildState):
        human_input = interrupt(state['prompt'])

        return {
            'human_input': human_input,
            'human_inputs': [human_input],
        }

    child_graph = (
        StateGraph(ChildState)
        .add_node("get_human_input", get_human_input)
        .add_edge(START, "get_human_input")
        .add_edge("get_human_input", END)
        .compile(checkpointer=sync_checkpointer)
    )

    class ParentState(TypedDict):
        prompts: list[str]
        human_inputs: Annotated[list[str], operator.add]

    def assign_workers(state: ParentState) -> list[Send]:
        return [
            Send(
                "child_graph",
                {'prompt': prompt},
            )
            for prompt in state['prompts']
        ]

    def cleanup(state: ParentState):
        assert len(state['human_inputs']) == len(state["prompts"])

    parent_graph = (
        StateGraph(ParentState)
        .add_node("child_graph", child_graph)
        .add_node("cleanup", cleanup)
        .add_conditional_edges(START, assign_workers, ["child_graph"])
        .add_edge("child_graph", "cleanup")
        .add_edge("cleanup", END)
        .compile(checkpointer=sync_checkpointer)
    )

    thread_config: RunnableConfig = {
        'configurable': {
            'thread_id': uuid.uuid4(),
        },
    }

    prompts = ['a', 'b', 'c', 'd', 'e']

    events = parent_graph.invoke(
        {'prompts': prompts},
        thread_config,
        stream_mode='values'
    )

    assert len(events['__interrupt__']) == len(prompts)
    interrupt_values = {i.value for i in events['__interrupt__']}
    assert interrupt_values == set(prompts)

    resume_map: dict[str, str] = {
        i.id: f"human input for prompt {i.value}"
        for i in parent_graph.get_state(thread_config).interrupts
    }

    result = parent_graph.invoke(Command(resume=resume_map), thread_config)
    assert result == {
        'prompts': prompts,
        'human_inputs': [
            f"human input for prompt {prompt}"
            for prompt in prompts
        ],
    }


def test_entrypoint_stateful(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    assert foo.invoke({"a": "1"}, config) == {"current": {"a": "1"}, "previous": None}
    assert foo.invoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": None},
    }
    assert foo.invoke({"a": "3"}, config) == {
        "current": {"a": "3"},
        "previous": {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": None},
        },
    }
    assert states == [
        None,
        {"current": {"a": "1"}, "previous": None},
        {"current": {"a": "2"}, "previous": {"current": {"a": "1"}, "previous": None}},
    ]

    # Test stream
    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "2"}}
    items = [item for item in foo.stream({"a": "1"}, config)]
    assert items == [{"foo": {"current": {"a": "1"}, "previous": None}}]


def test_entrypoint_stateful_update_state(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Test stateful entrypoint invoke."""

    # Test invoke
    states = []

    @entrypoint(checkpointer=sync_checkpointer)
    def foo(inputs, *, previous: Any) -> Any:
        states.append(previous)
        return {"previous": previous, "current": inputs}

    config = {"configurable": {"thread_id": "1"}}

    # assert print(foo.input_channels)
    foo.update_state(config, {"a": "-1"})
    assert foo.invoke({"a": "1"}, config) == {
        "current": {"a": "1"},
        "previous": {"a": "-1"},
    }
    assert foo.invoke({"a": "2"}, config) == {
        "current": {"a": "2"},
        "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
    }
    assert foo.invoke({"a": "3"}, config) == {
        "current": {"a": "3"},
        "previous": {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
        },
    }

    # update state
    foo.update_state(config, {"a": "3"})

    # Test stream
    assert [item for item in foo.stream({"a": "1"}, config)] == [
        {"foo": {"current": {"a": "1"}, "previous": {"a": "3"}}}
    ]
    assert states == [
        {"a": "-1"},
        {"current": {"a": "1"}, "previous": {"a": "-1"}},
        {
            "current": {"a": "2"},
            "previous": {"current": {"a": "1"}, "previous": {"a": "-1"}},
        },
        {"a": "3"},
    ]


def test_imp_exception(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    @task()
    def my_task(number: int):
        time.sleep(0.1)
        return number * 2

    @task()
    def task_with_exception(number: int):
        time.sleep(0.1)
        raise Exception("This is a test exception")

    @entrypoint(checkpointer=sync_checkpointer)
    def my_workflow(number: int):
        my_task(number).result()
        try:
            task_with_exception(number)
        except Exception as e:
            print(f"Exception caught: {e}")
        my_task(number).result()
        return "done"

    thread1 = {"configurable": {"thread_id": "1"}}
    assert my_workflow.invoke(1, thread1) == "done"

    assert [c for c in my_workflow.stream(1, thread1)] == [
        {"my_task": 2},
        {"my_task": 2},
        {"my_workflow": "done"},
    ]


@pytest.mark.parametrize("with_timeout", [False, "inner", "outer", "both"])
@pytest.mark.parametrize("subgraph_persist", [True, False])
def test_parent_command_goto(
    sync_checkpointer: BaseCheckpointSaver,
    subgraph_persist: bool,
    with_timeout: Literal[False, "inner", "outer", "both"],
) -> None:
    class State(TypedDict):
        dialog_state: Annotated[list[str], operator.add]

    def node_a_child(state):
        return {"dialog_state": ["a_child_state"]}

    def node_b_child(state):
        return Command(
            graph=Command.PARENT,
            goto="node_b_parent",
            update={"dialog_state": ["b_child_state"]},
        )

    sub_builder = StateGraph(State)
    sub_builder.add_node(node_a_child)
    sub_builder.add_node(node_b_child)
    sub_builder.add_edge(START, "node_a_child")
    sub_builder.add_edge("node_a_child", "node_b_child")
    sub_graph = sub_builder.compile(checkpointer=subgraph_persist)
    if with_timeout in ("inner", "both"):
        sub_graph.step_timeout = 1

    def node_b_parent(state):
        return {"dialog_state": ["node_b_parent"]}

    main_builder = StateGraph(State)
    main_builder.add_node(node_b_parent)
    main_builder.add_edge(START, "subgraph_node")
    main_builder.add_node("subgraph_node", sub_graph, destinations=("node_b_parent",))

    main_graph = main_builder.compile(sync_checkpointer, name="parent")
    if with_timeout in ("outer", "both"):
        main_graph.step_timeout = 1

    config = {"configurable": {"thread_id": 1}}

    assert main_graph.invoke(input={"dialog_state": ["init_state"]}, config=config) == {
        "dialog_state": ["init_state", "b_child_state", "node_b_parent"]
    }


@pytest.mark.parametrize("with_timeout", [True, False])
def test_timeout_with_parent_command(
    sync_checkpointer: BaseCheckpointSaver, with_timeout: bool
) -> None:
    """Test that parent commands are properly propagated during timeouts."""

    class State(TypedDict):
        value: str

    def parent_command_node(state: State) -> State:
        time.sleep(0.1)  # Add some delay before raising
        return Command(graph=Command.PARENT, goto="test_cmd", update={"key": "value"})

    builder = StateGraph(State)
    builder.add_node("parent_cmd", parent_command_node)
    builder.set_entry_point("parent_cmd")
    graph = builder.compile(checkpointer=sync_checkpointer)
    if with_timeout:
        graph.step_timeout = 1

    # Should propagate parent command, not timeout
    thread1 = {"configurable": {"thread_id": "1"}}
    with pytest.raises(ParentCommand) as exc_info:
        graph.invoke({"value": "start"}, thread1)
    assert exc_info.value.args[0].goto == "test_cmd"
    assert exc_info.value.args[0].update == {"key": "value"}


def test_fork_and_update_task_results(sync_checkpointer: BaseCheckpointSaver) -> None:
    """Test forking and updating task results with state history."""

    def checkpoint(values: dict[str, Any]):
        return ("checkpoint", {"values": values})

    def task(name: str, result: Any):
        return ("task", {"name": name, "result": result})

    def get_tree(history: list[StateSnapshot]) -> list:
        """Build a tree structure from state history for comparison."""
        if not history:
            return []

        # Build a tree structure similar to renderForks
        node_map: dict[str, dict] = {}
        root_nodes: list[dict] = []

        # Second pass: establish parent-child relationships
        for item in reversed(history):
            checkpoint_id = item.config["configurable"]["checkpoint_id"]
            parent_checkpoint_id = (
                item.parent_config["configurable"]["checkpoint_id"]
                if item.parent_config
                else None
            )
            node_map[checkpoint_id] = {"item": item, "children": []}

            parent = node_map.get(parent_checkpoint_id)
            (parent["children"] if parent else root_nodes).append(
                node_map[checkpoint_id]
            )

        def node_to_tree(node: dict) -> list:
            """Convert a node to tree structure."""
            result = [
                checkpoint(node["item"].values),
            ] + [
                task(task_info.name, task_info.result)
                for task_info in node["item"].tasks
            ]

            if len(node["children"]) > 1:
                branches = [node_to_tree(child) for child in node["children"]]
                return result + [branches]
            elif len(node["children"]) == 1:
                return result + node_to_tree(node["children"][0])
            else:
                return result

        if len(root_nodes) == 1:
            # Process all root nodes
            return node_to_tree(root_nodes[0])

        elif len(root_nodes) > 1:
            # Multiple root nodes - treat as branches
            branches = [node_to_tree(node) for node in root_nodes]
            return branches
        else:
            return []

    class State(TypedDict):
        name: Annotated[str, lambda a, b: " > ".join([a, b]) if a else b]

    # Define the graph with a sequence of nodes
    def one(state: State) -> Command:
        return Command(goto=[Send("two", {})], update={"name": "one"})

    def two(state: State) -> State:
        return {"name": "two"}

    def three(state: State) -> State:
        return {"name": "three"}

    graph = (
        StateGraph(State)
        .add_node("one", one)
        .add_node("two", two)
        .add_node("three", three)
        .add_edge(START, "one")
        .add_edge("one", "two")
        .add_edge("two", "three")
        .compile(checkpointer=sync_checkpointer)
    )

    config = {"configurable": {"thread_id": "1"}}
    history: list[StateSnapshot] = []

    # Initial run
    graph.invoke({"name": "start"}, config)
    history = list(graph.get_state_history(config))

    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        checkpoint({"name": "start > one"}),
        task("two", {"name": "two"}),
        task("two", {"name": "two"}),
        checkpoint({"name": "start > one > two > two"}),
        task("three", {"name": "three"}),
        checkpoint({"name": "start > one > two > two > three"}),
    ]

    # Update the start state
    graph.invoke(
        None,
        graph.update_state(
            history[4].config,
            values=[StateUpdate(values={"name": "start*"}, as_node="__start__")],
            as_node="__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start"}),
            checkpoint({"name": "start"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start > one > two > two > three"}),
        ],
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start*"}),
            checkpoint({"name": "start*"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start* > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start* > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start* > one > two > two > three"}),
        ],
    ]

    # Fork from task "one"
    # Start from the checkpoint that has the task "one"
    assert history[3].values == {"name": "start*"}
    assert len(history[3].tasks) == 1
    assert history[3].tasks[0].name == "one"

    graph.invoke(
        None,
        graph.update_state(
            history[3].config,
            [StateUpdate(values={"name": "one*"}, as_node="one")],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start"}),
            checkpoint({"name": "start"}),
            task("one", {"name": "one"}),
            checkpoint({"name": "start > one"}),
            task("two", {"name": "two"}),
            task("two", {"name": "two"}),
            checkpoint({"name": "start > one > two > two"}),
            task("three", {"name": "three"}),
            checkpoint({"name": "start > one > two > two > three"}),
        ],
        [
            checkpoint({"name": ""}),
            task("__start__", {"name": "start*"}),
            [
                [
                    checkpoint({"name": "start*"}),
                    task("one", {"name": "one"}),
                    checkpoint({"name": "start* > one"}),
                    task("two", {"name": "two"}),
                    task("two", {"name": "two"}),
                    checkpoint({"name": "start* > one > two > two"}),
                    task("three", {"name": "three"}),
                    checkpoint({"name": "start* > one > two > two > three"}),
                ],
                [
                    checkpoint({"name": "start*"}),
                    task("one", {"name": "one*"}),
                    checkpoint({"name": "start* > one*"}),
                    task("two", {"name": "two"}),
                    checkpoint({"name": "start* > one* > two"}),
                    task("three", {"name": "three"}),
                    checkpoint({"name": "start* > one* > two > three"}),
                ],
            ],
        ],
    ]

    config = {"configurable": {"thread_id": "2"}}

    # Initialize the thread once again
    graph.invoke({"name": "start"}, config)
    history = list(graph.get_state_history(config))

    # Fork from task "two"
    # Start from the checkpoint that has the task "two"
    assert history[2].values == {"name": "start > one"}

    graph.invoke(
        None,
        graph.update_state(
            history[2].config,
            [
                StateUpdate(values={"name": "two"}, as_node="two"),
                StateUpdate(values={"name": "two"}, as_node="two"),
            ],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
        ],
    ]

    # Fork task three
    assert history[1].values == {"name": "start > one > two > two"}
    assert len(history[1].tasks) == 1
    assert history[1].tasks[0].name == "three"

    graph.invoke(
        None,
        graph.update_state(
            history[1].config,
            [StateUpdate(values={"name": "three*"}, as_node="three")],
            "__copy__",
        ),
    )

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                [
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three*"}),
                        checkpoint({"name": "start > one > two > two > three*"}),
                    ],
                ],
            ],
        ],
    ]

    # Regenerate task three
    assert history[3].values == {"name": "start > one > two > two"}
    assert len(history[3].tasks) == 1
    assert history[3].tasks[0].name == "three"

    graph.invoke(None, graph.update_state(history[3].config, None, "__copy__"))

    history = list(graph.get_state_history(config))
    assert get_tree(history) == [
        checkpoint({"name": ""}),
        task("__start__", {"name": "start"}),
        checkpoint({"name": "start"}),
        task("one", {"name": "one"}),
        [
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                checkpoint({"name": "start > one > two > two"}),
                task("three", {"name": "three"}),
                checkpoint({"name": "start > one > two > two > three"}),
            ],
            [
                checkpoint({"name": "start > one"}),
                task("two", {"name": "two"}),
                task("two", {"name": "two"}),
                [
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three*"}),
                        checkpoint({"name": "start > one > two > two > three*"}),
                    ],
                    [
                        checkpoint({"name": "start > one > two > two"}),
                        task("three", {"name": "three"}),
                        checkpoint({"name": "start > one > two > two > three"}),
                    ],
                ],
            ],
        ],
    ]
