from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from src.multi_llm import build_llm
from src.tools import get_retriever_tool

llm = build_llm()
retriever_tool = get_retriever_tool()
llm_with_tools = llm.bind_tools([retriever_tool])

# Node 1: generate_query_or_respond - decides to retrieve or answer directly
def generate_query_or_respond(state):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

GRADE_PROMPT = "You are a relevance grader. Question: {question}\nDocument: {document}\nIs document relevant? Return JSON {{'relevant': true/false}}"
class Grade(BaseModel):
    relevant: bool = Field(description="Is document relevant to question")

grader_llm = build_llm().with_structured_output(Grade)

# Node 2: grade_documents
def grade_documents(state):
    question = state["messages"][0].content
    last_msg = state["messages"][-1].content
    # last message is ToolMessage with docs
    result = grader_llm.invoke([{"role": "user", "content": GRADE_PROMPT.format(question=question, document=last_msg[:2000])}])
    return {"context": last_msg, "question": question, "next": "relevant" if result.relevant else "irrelevant"}

# Node 3: rewrite_question
def rewrite_question(state):
    prompt = f"Rewrite this question to be better for vectorstore retrieval. Original: {state['question']}. Return only rewritten question."
    resp = llm.invoke([{"role": "user", "content": prompt}])
    return {"rewritten_question": resp.content, "messages": [{"role": "user", "content": resp.content}]}

# Node 4: generate_answer
def generate_answer(state):
    question = state.get("question") or state["messages"][0].content
    context = state.get("context") or state["messages"][-1].content
    prompt = f"Use this context to answer. If not in context, say you don't have info.\nContext:\n{context}\n\nQuestion: {question}"
    resp = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}