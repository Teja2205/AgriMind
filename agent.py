from langgraph.graph import StateGraph, END
from typing import TypedDict
import re
from sentence_transformers import CrossEncoder
import json
from openai import OpenAI
from dotenv import load_dotenv 
import os
import chromadb
import asyncio
import hashlib
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class AgentState(TypedDict):
    image_url: str
    crop: str
    check_result: str
    context: str
    sources: list[str]
    diagnosis: dict

async def call_farmos_tool(tool_name: str, arguments: dict) -> dict:
    """Call a FarmOS MCP tool and return the result."""
    server_params = StdioServerParameters(
        command="/Users/tejaguduguntla/ai-fullstack-course/farmos-mcp/.venv/bin/python",
        args=["/Users/tejaguduguntla/ai-fullstack-course/farmos-mcp/server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="crop_diseases")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_cache_key(crop: str, image_url: str) -> str:
    key = f"{crop}:{image_url}"
    return hashlib.md5(key.encode()).hexdigest()

cache_store = {}

def check_image(image_url: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"you act as agronomist and use the image url{image_url} and check if the given image is a plant or not and if it is a plant you must check if it has a disease or not based on your observation you must give response saying DISEASED_PLANT, HEALTHY_PLANT, or NOT_A_PLANT"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a plant with disease? Reply only: DISEASED_PLANT, HEALTHY_PLANT, or NOT_A_PLANT"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )
    return response.choices[0].message.content.strip()

def check_image_node(state: AgentState) -> dict:
    """Simple node that checks the image with error handling."""
    try:
        result = check_image(state["image_url"])
        return {"check_result": result}
    except Exception as e:
        print(f"check_image_node failed: {e}")
        return {"check_result": "ERROR"}

def get_context_node(state: AgentState) -> dict:
    """Queries FarmOS MCP and ChromaDB to build full field context."""

    # Step 1 — FarmOS MCP
    try:
        farmos_context = asyncio.run(call_farmos_tool(
            "get_field_context",
            {
                "lat": 40.7128,
                "lon": -74.0060,
                "crop": state["crop"],
                "state": "California"
            }
        ))
    except Exception as e:
        farmos_context = f"Field context unavailable: {str(e)}"

    # Step 2 — ChromaDB
    try:
        query_text = f"{state['crop']} common diseases symptoms treatment"
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        ).data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )
        docs = results['documents'][0]
        pairs = [[query_text, doc] for doc in docs]
        scores = reranker.predict(pairs)
        metadatas = results['metadatas'][0]
        ranked = sorted(zip(scores, docs, metadatas), reverse=True)
        top_docs = [doc for score, doc, meta in ranked[:3]]
        rag_context = "\n\n".join(top_docs)
        sources = [meta.get('source', 'unknown') if meta else 'unknown' for score, doc, meta in ranked[:3]]
    except Exception as e:
        print(f"ChromaDB query failed: {e}")
        rag_context = "Disease reference unavailable"
        sources = []

    # Step 3 — combine
    combined_context = f"FIELD CONDITIONS:\n{farmos_context}\n\nDISEASE REFERENCE:\n{rag_context}"
    return {"context": combined_context, "sources": sources}

def diagnose_node(state: AgentState) -> dict:
    """Calls GPT-4o with RAG context to diagnose the crop disease."""
    cache_key = get_cache_key(state["crop"], state["image_url"])
    if cache_key in cache_store:
        print("CACHE HIT — returning cached diagnosis")
        return {"diagnosis": cache_store[cache_key]}

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert agronomist. Use this reference material to inform your diagnosis:\n\n{state['context']}\n\nRespond ONLY with a JSON object with exactly these fields: disease_name (string), severity (integer 1-5), description (string), treatment_steps (array of strings). No markdown, no explanation, just raw JSON."
                },
                {
                    "role": "user",
                    "content": (
                        [
                            {
                                "type": "text",
                                "text": f"look at the image of a {state['crop']} and analyse the condition of the crop and give what are the best pesticides and minimal quantity of pesticide that i can use for the crop to have max yield"
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": state['image_url']}
                            }
                        ]
                        if state.get("image_url")
                        else f"Analyse the condition of a {state['crop']} crop and give the best pesticides and minimal quantity to use for maximum yield."
                    )
                }
            ],
            max_tokens=1000
        )
        raw = response.choices[0].message.content
        print("RAW:", repr(raw))
        raw = raw.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw = match.group() if match else raw
        parsed = json.loads(raw)
        cache_store[cache_key] = parsed
        return {"diagnosis": parsed}
    except Exception as e:
        print(f"diagnose_node failed: {e}")
        return {"diagnosis": {
            "disease_name": "Diagnosis unavailable",
            "severity": 0,
            "description": f"Service temporarily unavailable: {str(e)}",
            "treatment_steps": ["Please try again in 30 seconds"]
        }}

def route_after_check(state: AgentState) -> str:
    """Conditional edge — decides which node to go to after image check."""
    if state["check_result"] == "ERROR":
        return END
    if state["check_result"] == "DISEASED_PLANT":
        return "get_context"
    return END

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("check_image", check_image_node)
workflow.add_node("get_context", get_context_node)
workflow.add_node("diagnose", diagnose_node)

workflow.set_entry_point("check_image")
workflow.add_conditional_edges("check_image", route_after_check)
workflow.add_edge("get_context", "diagnose")
workflow.add_edge("diagnose", END)

def run_eval(question: str, crop: str) -> dict:
    """Runs evaluation without image check — for RAGAS testing only."""
    context_result = get_context_node({
        "image_url": "",
        "crop": crop,
        "check_result": "DISEASED_PLANT",
        "context": "",
        "sources": [],
        "diagnosis": {}
    })
    diagnose_result = diagnose_node({
        "image_url": "",
        "crop": crop,
        "check_result": "DISEASED_PLANT",
        "context": context_result["context"],
        "sources": context_result["sources"],
        "diagnosis": {}
    })
    return {
        "answer": str(diagnose_result["diagnosis"]),
        "contexts": context_result["sources"]
    }

def stream_diagnosis(state: AgentState):
    """Streams GPT-4o diagnosis token by token for real-time response."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert agronomist. Use this reference material to inform your diagnosis:\n\n{state['context']}\n\nProvide a clear, detailed diagnosis including: disease name, severity, description, and treatment steps with specific pesticide quantities. Write in plain English, not JSON."
            },
            {
                "role": "user",
                "content": (
                    [
                        {
                            "type": "text",
                            "text": f"look at the image of a {state['crop']} and analyse the condition of the crop and give what are the best pesticides and minimal quantity of pesticide that i can use for the crop to have max yield"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": state['image_url']}
                        }
                    ]
                    if state.get("image_url")
                    else f"Analyse the condition of a {state['crop']} crop and give the best pesticides and minimal quantity to use for maximum yield."
                )
            }
        ],
        stream=True,
        max_tokens=1000
    )
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield token

# Compile
agent = workflow.compile()