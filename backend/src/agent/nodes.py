"""
src/nodes.py — Enterprise LangGraph Node Functions
────────────────────────────────────────────────────
All nodes are async. Each returns a dict that partially updates AgentState.

Node pipeline:
  input_guard_node
    → generate_query_or_respond   (LLM decides: retrieve or answer directly)
    → retrieve_node               (hybrid search via ToolNode or direct)
    → grade_documents_node        (LLM grader, structured output)
    → [relevant] → reflect_node   (self-correction, optional)
              → generate_answer_node → output_guard_node → END
    → [irrelevant] → rewrite_question_node → generate_query_or_respond (loop)
    → [blocked] → END

Patterns demonstrated:
  • ReAct (generate_query_or_respond with tool binding)
  • Plan-and-execute (reflection node proposes corrections)
  • Reflection / self-correction (reflect_node)
  • Structured outputs (GradeResult, ReflectionResult)
  • Prompt versioning via Config.get_prompt()
  • Token usage tracking
  • LangSmith run naming
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.config import Config
from src.routing.multi_llm import get_llm, get_fast_llm, get_grader_llm
from src.agent.prompts import PromptLoader
from src.agent.state import AgentState, GuardResult, TokenUsage
from src.routing.tools import get_retriever_tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic structured output schemas
# ─────────────────────────────────────────────────────────────────────────────

class GradeResult(BaseModel):
    relevant: bool = Field(description="Is the document relevant to the question?")
    confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
    reason: str = Field(description="Brief explanation for the grade")


class ReflectionResult(BaseModel):
    needs_correction: bool = Field(description="Does the answer need improvement?")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    improved_answer: str = Field(default="", description="Improved answer, or empty if fine")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: extract text from multi-modal LLM content (Gemini 2.5 etc.)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", str(p)))
            else:
                parts.append(getattr(p, "text", str(p)))
        return "\n".join(parts)
    return str(content)


def _track_usage(response: Any, provider: str = "") -> TokenUsage:
    """Extract token usage metadata from LLM response if available."""
    usage = TokenUsage()
    meta = getattr(response, "response_metadata", {}) or {}
    if "usage" in meta:
        u = meta["usage"]
        usage.prompt_tokens = u.get("prompt_tokens", u.get("input_tokens", 0))
        usage.completion_tokens = u.get("completion_tokens", u.get("output_tokens", 0))
        usage.total_tokens = u.get("total_tokens", usage.prompt_tokens + usage.completion_tokens)
    elif hasattr(response, "usage_metadata"):
        u = response.usage_metadata or {}
        usage.prompt_tokens = u.get("input_tokens", 0)
        usage.completion_tokens = u.get("output_tokens", 0)
        usage.total_tokens = u.get("total_tokens", 0)
    from src.routing.multi_llm import estimate_cost
    usage.estimated_cost_usd = estimate_cost(provider, usage.prompt_tokens, usage.completion_tokens)
    return usage


# ─────────────────────────────────────────────────────────────────────────────
# Lazy tool binding (avoids import-time LLM init)
# ─────────────────────────────────────────────────────────────────────────────

_llm_with_tools = None

def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        tool = get_retriever_tool()
        # Use get_llm_with_tools which binds tools to base LLMs before
        # wrapping with resilience/fallbacks (RunnableLambda/with_fallbacks
        # don't have bind_tools method)
        from src.routing.multi_llm import get_llm_with_tools
        _llm_with_tools = get_llm_with_tools([tool])
    return _llm_with_tools


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Input Guard
# ─────────────────────────────────────────────────────────────────────────────

async def input_guard_node(state: AgentState) -> Dict:
    """
    OWASP LLM01/LLM06: Validate input before any LLM call.
    Runs fast regex checks + optional LLM guard check.
    Short-circuits the graph if blocked.
    """
    if not Config.GUARDRAIL_ENABLED:
        return {"node_trace": ["input_guard:skipped"]}

    from src.safety.guardrails import check_input_guard
    question = state.get("question", "")
    if not question and state.get("messages"):
        question = _extract_text(state["messages"][-1].content)

    start = time.perf_counter()
    result = await check_input_guard(question)
    latency = (time.perf_counter() - start) * 1000

    logger.info(f"[InputGuard] safe={result.safe} violation={result.violation} ({latency:.0f}ms)")

    updates: Dict = {
        "input_guard": result,
        "node_trace": [f"input_guard:{result.violation}"],
    }
    if not result.safe:
        updates.update({
            "guard_blocked": True,
            "guard_reason": result.reason,
            "answer": f"Request blocked: {result.reason}",
            "messages": [AIMessage(content=f"Request blocked: {result.reason}")],
        })
    if result.pii_entities:
        updates["pii_detected"] = True
        updates["pii_entities"] = result.pii_entities

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Generate Query or Respond (ReAct pattern)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_query_or_respond(state: AgentState) -> Dict:
    """
    Core ReAct node: LLM decides whether to call retrieve_documents tool
    or respond directly from conversation context / memory.
    """
    question = state.get("question", "")
    memory_ctx = state.get("memory_context", "")

    system_content = Config.get_prompt("GENERATE_QUERY_SYSTEM_PROMPT")
    if memory_ctx:
        system_content += f"\n\nRelevant conversation history:\n{memory_ctx}"

    messages = [SystemMessage(content=system_content)] + list(state["messages"])

    start = time.perf_counter()
    llm = _get_llm_with_tools()
    response = await llm.ainvoke(messages)
    latency = (time.perf_counter() - start) * 1000

    provider = type(llm).__name__
    usage = _track_usage(response, provider)

    logger.info(f"[GenerateQueryOrRespond] tool_calls={len(getattr(response, 'tool_calls', []))} ({latency:.0f}ms)")

    return {
        "messages": [response],
        "node_trace": ["generate_query_or_respond"],
        "token_usage": usage,
        "llm_provider_used": provider,
        "latency_ms": latency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Grade Documents
# ─────────────────────────────────────────────────────────────────────────────

async def grade_documents_node(state: AgentState) -> Dict:
    """
    Grade retrieved documents for relevance.
    Uses fast structured-output LLM. Updates relevant_docs and context.
    """
    question = state.get("question", "")
    if not question and state.get("messages"):
        question = _extract_text(state["messages"][0].content)

    # Extract docs from the latest ToolMessage
    tool_msg_content = ""
    retrieved_docs_raw: List[Document] = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_msg_content = _extract_text(msg.content)
            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, list):
                retrieved_docs_raw = [doc for doc in artifact if isinstance(doc, Document)]
            break

    all_docs = list(state.get("retrieved_docs", []))
    all_docs.extend(retrieved_docs_raw)

    grader = get_grader_llm()
    relevant: List[Document] = []
    context_parts: List[str] = []

    if all_docs:
        for doc in all_docs:
            try:
                prompt = PromptLoader.grading(question, doc.page_content[:1500])
                grade: GradeResult = await grader.ainvoke([{"role": "user", "content": prompt}])
                if grade.relevant and grade.confidence >= 0.5:
                    relevant.append(doc)
                    context_parts.append(doc.page_content)
                    logger.debug(f"[Grader] RELEVANT (conf={grade.confidence:.2f}): {doc.metadata.get('filename','?')}")
                else:
                    logger.debug(f"[Grader] IRRELEVANT: {doc.metadata.get('filename','?')}: {grade.reason}")
            except Exception as e:
                logger.warning(f"[Grader] Error grading doc: {e} — including by default")
                relevant.append(doc)
                context_parts.append(doc.page_content)
    elif tool_msg_content:
        # Fallback: grade the raw tool message content
        try:
            grade: GradeResult = await grader.ainvoke([{
                "role": "user",
                "content": PromptLoader.grading(question, tool_msg_content[:2000])
            }])
            if grade.relevant:
                context_parts.append(tool_msg_content)
        except Exception as e:
            logger.warning(f"[Grader] Fallback grading error: {e}")
            context_parts.append(tool_msg_content)

    context = "\n\n---\n\n".join(context_parts) if context_parts else tool_msg_content
    is_relevant = len(relevant) > 0 or bool(context_parts)

    logger.info(f"[GradeDocuments] {len(relevant)}/{len(all_docs or [1])} relevant | context={len(context)} chars")

    return {
        "retrieved_docs": retrieved_docs_raw,
        "relevant_docs": relevant,
        "context": context,
        "node_trace": [f"grade_documents:{'relevant' if is_relevant else 'irrelevant'}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Rewrite Question
# ─────────────────────────────────────────────────────────────────────────────

async def rewrite_question_node(state: AgentState) -> Dict:
    """
    Query rewriting: rephrase the question for better vector store matching.
    Caps reflection loops at Config.MAX_QUERY_REWRITES.
    """
    question = state.get("question", "")
    reflection_count = state.get("reflection_count", 0)

    if reflection_count >= Config.MAX_QUERY_REWRITES:
        logger.warning(f"[Rewrite] Max rewrites ({Config.MAX_QUERY_REWRITES}) reached — aborting")
        return {
            "guard_blocked": True,
            "guard_reason": "Could not find relevant documents after multiple retrieval attempts.",
            "answer": "I could not find relevant information to answer your question.",
            "messages": [AIMessage(content="I could not find relevant information to answer your question after multiple attempts.")],
            "node_trace": ["rewrite_question:max_exceeded"],
        }

    prompt = PromptLoader.rewrite(question)
    response = await get_fast_llm().ainvoke([{"role": "user", "content": prompt}])
    rewritten = _extract_text(response.content).strip()

    logger.info(f"[Rewrite] '{question[:60]}' → '{rewritten[:60]}'")

    return {
        "rewritten_question": rewritten,
        "question": rewritten,
        "reflection_count": reflection_count + 1,
        "messages": [HumanMessage(content=rewritten)],
        "node_trace": [f"rewrite_question:attempt_{reflection_count + 1}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Generate Answer
# ─────────────────────────────────────────────────────────────────────────────

async def generate_answer_node(state: AgentState) -> Dict:
    """
    Final answer synthesis from graded context.
    Uses quality LLM for best output.
    """
    question = state.get("question", "")
    if not question and state.get("messages"):
        question = _extract_text(state["messages"][0].content)

    context = state.get("context", "")
    if not context:
        # Fallback: use last tool message
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                context = _extract_text(msg.content)
                break

    if not context:
        answer = "I don't have enough information in the knowledge base to answer this question."
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
            "node_trace": ["generate_answer:no_context"],
        }

    prompt = PromptLoader.generation(question, context)

    start = time.perf_counter()
    from src.routing.multi_llm import get_quality_llm
    llm = get_quality_llm()
    response = await llm.ainvoke([
        SystemMessage(content="You are a precise, helpful assistant. Answer only from the provided context."),
        HumanMessage(content=prompt),
    ])
    latency = (time.perf_counter() - start) * 1000

    answer = _extract_text(response.content)
    usage = _track_usage(response, type(llm).__name__)

    logger.info(f"[GenerateAnswer] {len(answer)} chars ({latency:.0f}ms)")

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "token_usage": usage,
        "latency_ms": latency,
        "node_trace": ["generate_answer"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — Reflect / Self-correction
# ─────────────────────────────────────────────────────────────────────────────

async def reflect_node(state: AgentState) -> Dict:
    """
    Reflection pattern: evaluate the generated answer for quality,
    hallucinations, and completeness. Optionally replace with improved answer.
    """
    if state.get("reflection_count", 0) >= state.get("max_reflections", 2):
        return {"node_trace": ["reflect:skipped_max"]}

    question = state.get("question", "")
    context = state.get("context", "")
    answer = state.get("answer", "")

    if not all([question, context, answer]):
        return {"node_trace": ["reflect:skipped_missing_data"]}

    reflection_prompt = Config.get_prompt("REFLECTION_PROMPT").format(
        question=question,
        context=context[:3000],
        answer=answer[:2000],
    )

    try:
        reflection_llm = get_fast_llm(structured_output_schema=ReflectionResult)
        result: ReflectionResult = await reflection_llm.ainvoke([{
            "role": "user", "content": reflection_prompt
        }])

        updates: Dict = {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "node_trace": [f"reflect:needs_correction={result.needs_correction}"],
        }

        if result.needs_correction and result.improved_answer:
            logger.info(f"[Reflect] Correcting answer. Issues: {result.issues}")
            updates["answer"] = result.improved_answer
            updates["messages"] = [AIMessage(content=result.improved_answer)]
            updates["correction_notes"] = result.issues
        else:
            logger.debug("[Reflect] Answer accepted as-is")

        return updates

    except Exception as e:
        logger.warning(f"[Reflect] Reflection failed: {e} — keeping original answer")
        return {"node_trace": ["reflect:error"]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Output Guard
# ─────────────────────────────────────────────────────────────────────────────

async def output_guard_node(state: AgentState) -> Dict:
    """
    Output validation: check for PII leakage, hallucinations,
    and system prompt disclosure in the generated answer.
    """
    if not Config.GUARDRAIL_ENABLED:
        return {"node_trace": ["output_guard:skipped"]}

    from src.safety.guardrails import check_output_guard
    question = state.get("question", "")
    context = state.get("context", "")
    answer = state.get("answer", "")

    if not answer:
        return {"node_trace": ["output_guard:no_answer"]}

    result = await check_output_guard(question, context, answer)

    logger.info(f"[OutputGuard] safe={result.safe} hallucination={result.hallucination_score:.2f}")

    updates: Dict = {
        "output_guard": result,
        "node_trace": [f"output_guard:safe={result.safe}"],
    }

    if not result.safe:
        safe_answer = "I cannot provide this response due to content policy restrictions."
        updates.update({
            "guard_blocked": True,
            "guard_reason": result.reason,
            "answer": safe_answer,
            "messages": [AIMessage(content=safe_answer)],
        })
    elif result.hallucination_score > Config.GUARDRAIL_HALLUCINATION_THRESHOLD:
        logger.warning(f"[OutputGuard] High hallucination score: {result.hallucination_score:.2f}")
        updates["correction_notes"] = [f"Potential hallucination detected (score={result.hallucination_score:.2f})"]

    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions (used in graph.py conditional edges)
# ─────────────────────────────────────────────────────────────────────────────

def route_after_guard(state: AgentState) -> str:
    """Route: if blocked → END, else → generate_query_or_respond."""
    if state.get("guard_blocked"):
        return "blocked"
    return "continue"


def route_after_retrieval(state: AgentState) -> str:
    """Route: grade retrieved docs, decide next step."""
    context = state.get("context", "")
    relevant_docs = state.get("relevant_docs", [])
    if relevant_docs or context:
        return "relevant"
    return "irrelevant"


def route_after_grading(state: AgentState) -> str:
    """Route based on grading result and rewrite limits."""
    relevant = state.get("relevant_docs", [])
    context = state.get("context", "")
    reflection_count = state.get("reflection_count", 0)

    if relevant or context:
        return "generate_answer"
    if reflection_count >= Config.MAX_QUERY_REWRITES:
        return "generate_answer"   # answer with "no info found"
    return "rewrite"


def route_after_answer(state: AgentState) -> str:
    """Route: if reflection enabled and not at max → reflect, else → output_guard."""
    if (
        Config.GUARDRAIL_ENABLED
        and state.get("reflection_count", 0) < state.get("max_reflections", 2)
        and state.get("answer")
        and state.get("context")
    ):
        return "reflect"
    return "output_guard"
