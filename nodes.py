from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from agents import (
    classify, classify_prompt,
    coder, coder_prompt,
    thinker, thinker_prompt,
    image, image_prompt,
    final, final_output_prompt,
    audit, audit_prompt
)
from research_paper_rag import retriever
from typing import TypedDict, Annotated, List
import operator
import json
import re


# ──────────────────────────────────────────────
# STATE SCHEMA
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    user_query:      str
    target_model:    str
    category:        str
    scaffold:        dict
    complexity:      str
    mcq_answer:      dict
    final_response:  str
    iteration_count: int
    revision_count:  Annotated[List[str], operator.add]


# ──────────────────────────────────────────────
# SHARED PARSER
# ──────────────────────────────────────────────

json_parser = JsonOutputParser()


# ──────────────────────────────────────────────
# NODES
# ──────────────────────────────────────────────

def classify_node(state: AgentState):
    chain    = classify_prompt | classify | json_parser
    response = chain.invoke({"query": state["user_query"]})
    return {
        "category":     response["prompt_type"],
        "target_model": response["target_model"],
        "complexity":   response.get("complexity", "moderate"),
    }


def coder_node(state: AgentState):
    chain    = coder_prompt | coder | json_parser
    response = chain.invoke({
        "user_query":   state["user_query"],
        "target_model": state["target_model"],
    })
    return {"scaffold": response}


def image_node(state: AgentState):
    chain    = image_prompt | image | json_parser
    response = chain.invoke({
        "user_query":   state["user_query"],
        "target_model": state["target_model"],
    })
    return {"scaffold": response}


def thinker_node(state: AgentState):
    chain    = thinker_prompt | thinker | json_parser
    response = chain.invoke({
        "user_query":   state["user_query"],
        "target_model": state["target_model"],
    })
    return {"scaffold": response}


def final_output_node(state: AgentState):
    """
    FIX: The original code called prompt.format({dict}) which passes a
    single positional dict — that is NOT how ChatPromptTemplate works and
    raises: TypeError: 'takes 1 positional argument but 2 were given'.

    The correct call is prompt.format_messages(**kwargs), which substitutes
    each named template variable individually.
    """

    search_query = f"Optimized prompts for, Model = {state["target_model"]} and prompt = {state['scaffold']}"

    docs = retriever.invoke(search_query)
    research_content = "\n\n".join([doc.page_content for doc in docs])
    """
    # ── 1. Render prompt template with keyword arguments ──────────────
    messages_lc = final_output_prompt.format_messages(
        collected_data=state.get("scaffold", {}),
        mcq_answers=state.get("mcq_answer", {}),
        target_model=state["target_model"],
        complexity=state.get("complexity", "moderate"),
        research_data = research_content
    )

    # ── 2. Convert LangChain message objects → Ollama-compatible dicts ─
    # LangChain: msg.type = "human" | "ai" | "system"
    # Ollama:    role     = "user"  | "assistant" | "system"
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    messages = [
        {"role": role_map.get(msg.type, "user"), "content": msg.content}
        for msg in messages_lc
    ]

    # ── 3. Call the custom Ollama client ──────────────────────────────
    response   = ollama_custom_client.chat(
        model="gemma4:31b-cloud",
        messages=messages,
        stream=False,
    )
    final_text = response["message"]["content"]
    """
    chain = final_output_prompt | final | StrOutputParser()
    
    final_text = chain.invoke({
        "collected_data" : state.get("scaffold", {}),
        "mcq_answers" : state.get("mcq_answer", {}),
        "target_model" : state["target_model"],
        "complexity" : state.get("complexity", "moderate"),
        "research_data" : research_content
    })

    return {"final_response": final_text}

import ast

def audit_node(state: AgentState):
    """
    Uses StrOutputParser first, then manually extracts JSON from the raw
    string to handle models that wrap JSON in markdown fences or use
    single quotes instead of double quotes.
    """
    chain = audit_prompt | audit | StrOutputParser()

    try:
        raw_response = chain.invoke({"data": state["scaffold"]})

        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON block found in audit response")

        # clean_json = json_match.group(0).replace("'", '"')
        clean_json = json_match.group(0).strip()
        # response   = json.loads(clean_json)
        try:
            response = json.loads(clean_json)
        except:
            response = ast.literal_eval(clean_json)

    except Exception as e:
        print(f"[audit_node] Parse error: {e}. Using fallback.")
        response = {"score": 5, "reason": "Audit failed to parse correctly."}

    return {
        "iteration_count": state["iteration_count"] + 1,
        "revision_count":  [response.get("reason", "N/A")],
    }