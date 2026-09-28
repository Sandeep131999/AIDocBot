"""
src/graph.py — Enterprise LangGraph Agentic RAG
─────────────────────────────────────────────────
Patterns implemented:
  • Corrective RAG (CRAG): retrieve → grade → rewrite/generate
  • ReAct: LLM decides when to use tools
  • Reflection / self-correction: reflect node after generation
  • Human-in-the-loop: interrupt_before for sensitive operations
  • Multi-provider fallback baked into every node
  • Checkpointer: memory (default) or Postgres (production)
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
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.nodes import (
    input_guard_node,
    generate_query_or_respond,
    grade_documents_node,
    rewrite_question_node,
    generate_answer_node,
    reflect_node,
    output_guard_node,
    route_after_guard,
    route_after_grading,
    route_after_answer,
)
from src.tools import get_retriever_tool, get_all_tools
from src.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_checkpointer():
    """
    Build the appropriate checkpointer based on config.
    - memory: in-process (dev/test)
    - postgres: persistent (production) — requires POSTGRES_URL
    - redis: fast ephemeral (staging)
    """
    backend = Config.CHECKPOINTER_BACKEND

    if backend == "postgres" and Config.POSTGRES_ENABLED:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            logger.info("[Checkpointer] Using Postgres async saver")
            # Note: caller must call .setup() after creating the graph
            return AsyncPostgresSaver.from_conn_string(
                Config.POSTGRES_URL.replace("+asyncpg", "")
            )
        except Exception as e:
            logger.warning(f"[Checkpointer] Postgres failed ({e}), falling back to memory")

    if backend == "redis" and Config.REDIS_ENABLED:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            logger.info("[Checkpointer] Using Redis saver")
            return RedisSaver.from_conn_string(Config.REDIS_URL)
        except Exception as e:
            logger.warning(f"[Checkpointer] Redis failed ({e}), falling back to memory")

    logger.info("[Checkpointer] Using in-memory saver")
    return MemorySaver()


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_guard(state: AgentState) -> str:
    return "end" if state.get("guard_blocked") else "continue"


def _route_tools_or_end(state: AgentState) -> str:
    """After generate_query_or_respond: check if tool calls were made."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"
    return END


def _route_after_grading(state: AgentState) -> str:
    relevant = state.get("relevant_docs", [])
    context = state.get("context", "")
    reflection_count = state.get("reflection_count", 0)

    if relevant or context:
        return "generate_answer"
    if reflection_count >= Config.MAX_QUERY_REWRITES:
        return "generate_answer"
    return "rewrite"


def _route_after_answer(state: AgentState) -> str:
    """Route to reflect or skip straight to output_guard."""
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
    checkpointer=None,
):
    """
    Build and compile the enterprise agentic RAG LangGraph.

    Args:
        enable_human_in_loop: If True, add interrupt_before on risky nodes
        checkpointer: Custom checkpointer (default: auto-selected by config)

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

    # Entry point
    workflow.set_entry_point("input_guard")

    # After input guard: blocked → END, else → LLM
    workflow.add_conditional_edges(
        "input_guard",
        _route_after_guard,
        {"end": END, "continue": "generate_query_or_respond"},
    )

    # ReAct: LLM calls tool or responds directly
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
        {
            "generate_answer": "generate_answer",
            "rewrite": "rewrite_question",
        },
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
    cp = checkpointer if checkpointer is not None else _build_checkpointer()

    compile_kwargs: Dict[str, Any] = {"checkpointer": cp}

    # Human-in-the-loop: pause before generating answer so a human can review
    if enable_human_in_loop:
        compile_kwargs["interrupt_before"] = ["generate_answer"]
        logger.info("[Graph] Human-in-the-loop enabled: interrupt_before=generate_answer")

    graph = workflow.compile(**compile_kwargs)

    logger.info(
        f"[Graph] Compiled | checkpointer={type(cp).__name__} | HITL={enable_human_in_loop}"
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor / multi-agent pattern
# ─────────────────────────────────────────────────────────────────────────────

def build_supervisor_graph():
    """
    Supervisor pattern: a router LLM decides which sub-agent to invoke.
    Sub-agents: rag_agent (document Q&A), general_agent (direct LLM).

    Used when you want to route between specialized agents rather than
    always going through the RAG pipeline.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.multi_llm import get_fast_llm
    from pydantic import BaseModel
    from typing import Literal

    class RouterDecision(BaseModel):
        agent: Literal["rag_agent", "general_agent", "end"]
        reason: str

    router_llm = get_fast_llm().with_structured_output(RouterDecision)

    async def supervisor_node(state: AgentState) -> Dict:
        """Route to the appropriate sub-agent."""
        question = state.get("question", "")
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
        logger.info(f"[Supervisor] Route → {decision.agent}: {decision.reason}")
        return {"node_trace": [f"supervisor:{decision.agent}"], "question": question}

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
        trace = state.get("node_trace", [])
        for entry in reversed(trace):
            if entry.startswith("supervisor:"):
                return entry.split(":")[1]
        return "rag_agent"

    # Build sub-graph for RAG
    rag_graph = build_agentic_rag_graph()

    supervisor_workflow = StateGraph(AgentState)
    supervisor_workflow.add_node("supervisor", supervisor_node)
    supervisor_workflow.add_node("rag_agent", rag_graph)
    supervisor_workflow.add_node("general_agent", general_agent_node)
    supervisor_workflow.add_node("output_guard", output_guard_node)

    supervisor_workflow.set_entry_point("supervisor")
    supervisor_workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"rag_agent": "rag_agent", "general_agent": "general_agent", "end": END},
    )
    supervisor_workflow.add_edge("rag_agent", "output_guard")
    supervisor_workflow.add_edge("general_agent", "output_guard")
    supervisor_workflow.add_edge("output_guard", END)

    cp = _build_checkpointer()
    return supervisor_workflow.compile(checkpointer=cp)
