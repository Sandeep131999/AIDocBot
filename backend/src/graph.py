from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.state import AgentState
from src.nodes import generate_query_or_respond, grade_documents, rewrite_question, generate_answer
from src.tools import get_retriever_tool

def build_agentic_rag_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([get_retriever_tool()]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("generate_query_or_respond")

    # Edge 1: Does LLM want to retrieve?
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END}
    )

    # Edge 2: Grade docs after retrieve
    def grade_router(state):
        # call grader logic inside grade_documents node, but routing here
        from src.nodes import grader_llm, GRADE_PROMPT
        q = state.get("question")
        ctx = state["messages"][-1].content
        grade = grader_llm.invoke([{"role":"user","content": GRADE_PROMPT.format(question=q, document=ctx[:2000])}])
        return "generate_answer" if grade.relevant else "rewrite_question"

    workflow.add_conditional_edges(
        "retrieve",
        grade_router,
        {"generate_answer": "generate_answer", "rewrite_question": "rewrite_question"}
    )

    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

# usage:
# graph = build_agentic_rag_graph()
# graph.invoke({"messages": [{"role":"user","content":"How to reset password?"}]})