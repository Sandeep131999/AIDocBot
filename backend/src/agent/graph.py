"""
src/graph.py — Enterprise LangGraph Agentic RAG
─────────────────────────────────────────────────
Patterns implemented:
  • Corrective RAG (CRAG): retrieve → grade → rewrite/generate
  • ReAct: LLM decides when to use tools
  • Reflection / self-correction: reflect node after generation
  • Human-in-the-loop: interrupt_before for sensitive operations
  • Multi-provider fallback baked into every node
    • Checkpointer: memory or PostgreSQL
  • Streaming: all nodes support astream_events

Graph topology:
  START
    └─→ input_guard_node
          ├─→ [blocked]  → END
          └─→ [continue] → generate_query_or_respond
                ├─→ [no tools] → output_guard_node → END
                └─→ [tools]    → ToolNode (retrieve)
                                    └─→ grade_documents_node
                                          ├─→ [relevant] → generate_answer_node
                                          │                   └─→ reflect_node
                                          │                         └─→ output_guard_node → END
                                          └─→ [rewrite]  → rewrite_question_node
                                                             └─→ generate_query_or_respond (loop)

Install:
  pip install -U langgraph langgraph-checkpoint-postgres "psycopg[binary,pool]" \

Startup usage (FastAPI lifespan example):
  checkpointer, cleanup = await create_checkpointer()
  graph = build_agentic_rag_graph(checkpointer=checkpointer)
  ...
  # on shutdown
  if cleanup:
      await cleanup()
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Literal, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from src.agent.nodes import (
    generate_answer_node,
    generate_query_or_respond,
    grade_documents_node,
    input_guard_node,
    output_guard_node,
    reflect_node,
    rewrite_question_node,
)
from src.agent.state import AgentState
from src.config import Config
from src.routing.tools import get_all_tools

logger = logging.getLogger(__name__)

# Sentinel so that checkpointer=None can mean "no checkpointer" (needed for
# subgraphs) while "not passed" means "use the default in-memory saver".
_DEFAULT = object()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer factory (ASYNC — call with `await` once at app startup)
# ─────────────────────────────────────────────────────────────────────────────

async def create_checkpointer() -> Tuple[Any, Optional[Callable[[], Awaitable[None]]]]:
    """
    Build the checkpointer selected by Config.CHECKPOINTER_BACKEND.

    Returns:
        (checkpointer, cleanup)
        cleanup is an async callable to run on shutdown, or None.

    Backends:
      - memory:   in-process (dev/test)
      - postgres: persistent (production) — needs POSTGRES_URL
    """
    backend = Config.CHECKPOINTER_BACKEND

    if backend == "postgres" and Config.POSTGRES_ENABLED:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            conninfo = Config.POSTGRES_URL.replace("+asyncpg", "").replace("+psycopg", "")
            pool = AsyncConnectionPool(
                conninfo=conninfo,
                max_size=10,
                kwargs={"autocommit": True, "row_factory": dict_row},
                open=False,
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)
            await saver.setup()  # creates tables (idempotent)
            logger.info("[Checkpointer] Using Postgres async saver")
            return saver, pool.close
        except Exception as e:
            logger.warning(f"[Checkpointer] Postgres failed ({e}), falling back to memory")

    logger.info("[Checkpointer] Using in-memory saver")
    return MemorySaver(), None


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_guard(state: AgentState) -> str:
    return "end" if state.get("guard_blocked") else "continue"


def _route_after_grading(state: AgentState) -> str:
    """
    relevant docs / context → generate
    otherwise rewrite, until the rewrite cap is hit, then generate anyway.

    NOTE: requires `rewrite_count: int` in AgentState, incremented by
    rewrite_question_node (return {"rewrite_count": state.get("rewrite_count", 0) + 1, ...}).
    """
    relevant = state.get("relevant_docs", [])
    context = state.get("context", "")
    rewrite_count = state.get("rewrite_count", 0)

    if relevant or context:
        return "generate_answer"
    if rewrite_count >= Config.MAX_QUERY_REWRITES:
        return "generate_answer"
    return "rewrite"


def _route_after_answer(state: AgentState) -> str:
    """Route to reflect, or skip straight to output_guard."""
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 2)
    has_context = bool(state.get("context", ""))
    has_answer = bool(state.get("answer", ""))

    if has_answer and has_context and reflection_count < max_reflections:
        return "reflect"
    return "output_guard"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_agentic_rag_graph(
    enable_human_in_loop: bool = False,
    checkpointer: Any = _DEFAULT,
):
    """
    Build and compile the enterprise agentic RAG LangGraph.

    Args:
        enable_human_in_loop: If True, pause before `generate_answer` for review.
        checkpointer:
            - not passed  → in-memory saver
            - a saver     → use it (get one from `await create_checkpointer()`)
            - None        → no checkpointer (REQUIRED when used as a subgraph;
                            the parent graph's checkpointer is propagated)

    Returns:
        Compiled LangGraph ready for ainvoke / astream_events
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode(get_all_tools()))
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("output_guard", output_guard_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    workflow.set_entry_point("input_guard")

    # After input guard: blocked → END, else → LLM
    workflow.add_conditional_edges(
        "input_guard",
        _route_after_guard,
        {"end": END, "continue": "generate_query_or_respond"},
    )

    # ReAct: LLM calls a tool or responds directly
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: "output_guard"},
    )

    # After ToolNode → grade the retrieved documents
    workflow.add_edge("retrieve", "grade_documents")

    # After grading: relevant → generate, irrelevant → rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"generate_answer": "generate_answer", "rewrite": "rewrite_question"},
    )

    # Rewrite loops back to the LLM
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # After answer: reflect (if enabled) or output_guard
    workflow.add_conditional_edges(
        "generate_answer",
        _route_after_answer,
        {"reflect": "reflect", "output_guard": "output_guard"},
    )

    # Reflect goes to output_guard (one reflection pass)
    workflow.add_edge("reflect", "output_guard")

    # Output guard → END
    workflow.add_edge("output_guard", END)

    # ── Compile ────────────────────────────────────────────────────────────
    cp = MemorySaver() if checkpointer is _DEFAULT else checkpointer
    compile_kwargs: Dict[str, Any] = {}

    if cp is not None:
        compile_kwargs["checkpointer"] = cp

    # Human-in-the-loop: pause before generating the answer so a human can review
    if enable_human_in_loop:
        compile_kwargs["interrupt_before"] = ["generate_answer"]
        logger.info("[Graph] Human-in-the-loop enabled: interrupt_before=generate_answer")
        if cp is None:
            logger.warning(
                "[Graph] HITL without a checkpointer only works if the parent graph provides one"
            )

    graph = workflow.compile(**compile_kwargs)

    logger.info(
        f"[Graph] Compiled | checkpointer={type(cp).__name__ if cp else None} "
        f"| HITL={enable_human_in_loop}"
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor / multi-agent pattern
# ─────────────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    agent: Literal["rag_agent", "general_agent", "end"]
    reason: str


def build_supervisor_graph(checkpointer: Any = _DEFAULT):
    """
    Supervisor pattern: a router LLM decides which sub-agent to invoke.
    Sub-agents: rag_agent (document Q&A), general_agent (direct LLM).

    Args:
        checkpointer: same semantics as build_agentic_rag_graph. The checkpointer
                      belongs on this PARENT graph only.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.routing.multi_llm import get_fast_llm

    router_llm = get_fast_llm(structured_output_schema=RouterDecision)

    async def supervisor_node(state: AgentState) -> Dict:
        """Route to the appropriate sub-agent."""
        question = state.get("question", "")
        try:
            decision: RouterDecision = await router_llm.ainvoke([
                SystemMessage(content=(
                    "You are a routing supervisor. Decide which agent should handle this request:\n"
                    "- rag_agent: questions about uploaded documents, specific knowledge\n"
                    "- general_agent: general questions, greetings, clarifications\n"
                    "- end: no action needed\n"
                    "Return your routing decision."
                )),
                HumanMessage(content=question),
            ])
            agent, reason = decision.agent, decision.reason
        except Exception as e:
            logger.warning(f"[Supervisor] Structured output failed ({e}); defaulting to rag_agent")
            agent, reason = "rag_agent", "fallback"

        logger.info(f"[Supervisor] Route → {agent}: {reason}")
        return {"node_trace": [f"supervisor:{agent}"], "question": question}

    async def general_agent_node(state: AgentState) -> Dict:
        """Handle non-RAG queries directly."""
        question = state.get("question", "")
        response = await get_fast_llm().ainvoke([
            SystemMessage(content="You are a helpful assistant. Answer the question directly."),
            HumanMessage(content=question),
        ])
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "answer": answer,
            "messages": [response],
            "node_trace": ["general_agent"],
        }

    def route_supervisor(state: AgentState) -> str:
        for entry in reversed(state.get("node_trace", [])):
            if entry.startswith("supervisor:"):
                return entry.split(":", 1)[1]
        return "rag_agent"

    # Subgraph: NO own checkpointer — the parent's is propagated to it.
    rag_graph = build_agentic_rag_graph(checkpointer=None)

    wf = StateGraph(AgentState)
    wf.add_node("supervisor", supervisor_node)
    wf.add_node("rag_agent", rag_graph)
    wf.add_node("general_agent", general_agent_node)
    wf.add_node("output_guard", output_guard_node)

    wf.set_entry_point("supervisor")
    wf.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"rag_agent": "rag_agent", "general_agent": "general_agent", "end": END},
    )
    # rag_agent already ends with its own output_guard, so go straight to END
    wf.add_edge("rag_agent", END)
    wf.add_edge("general_agent", "output_guard")
    wf.add_edge("output_guard", END)

    cp = MemorySaver() if checkpointer is _DEFAULT else checkpointer
    return wf.compile(checkpointer=cp) if cp is not None else wf.compile()