from langgraph.graph import StateGraph, END
from typing import TypedDict
import re
from sentence_transformers import CrossEncoder
import json
from openai import OpenAI
from dotenv import load_dotenv 
import os
import chromadb
import hashlib

load_dotenv()
class AgentState(TypedDict):
     image_url: str
     crop: str
     check_result: str
     context: str
     sources: list[str]
     diagnosis: dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path = "./chroma_db")
collection = chroma_client.get_collection(name = "crop_diseases")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def get_cache_key(crop: str, image_url: str) -> str:
    key = f"{crop}:{image_url}"
    return hashlib.md5(key.encode()).hexdigest()
cache_store = {}

def check_image(image_url: str) -> str:
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages= [{
            "role": "system",
            "content": f"you act as agronomist and use the image url{image_url} and check if the given image is a plant or not and if it is a plant you must check if it has a disease or not based on you observation you must give response saying DISEASED_PLANT, HEALTHY_PLANT, or NOT_A_PLANT "
            },
            {
                "role": "user",
                "content":[
                     {"type": "text", "text": "Is this a plant with disease? Reply only: DISEASED_PLANT, HEALTHY_PLANT, or NOT_A_PLANT"},
                     {"type": "image_url", "image_url": {"url": image_url}}
                ]
                
            }]
    )
    return response.choices[0].message.content.strip()

def check_image_node(state : AgentState) -> dict:
    """ Simple node that checks the image and gives a respose wether the given image is plant or not and it also give if the plant is diseased or healthy"""  
    result = check_image(state["image_url"])  # call the function
    return {"check_result": result}   
    
def get_context_node(state : AgentState) -> dict:
    """Queries ChromaDB with the crop type to retrieve relevant disease context and sources."""
    query_text = f"{state['crop']} common diseases symptoms treatment"
    query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query_text
    ).data[0].embedding
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10)
    docs = results['documents'][0]
    pairs = [[query_text, doc] for doc in docs]
    scores = reranker.predict(pairs)
    metadatas = results['metadatas'][0]
    ranked = sorted(zip(scores, docs, metadatas), reverse=True)
    top_docs = [doc for score, doc, meta in ranked[:3]]
    context = "\n\n".join(top_docs)
    sources = [meta.get('source', 'unknown') if meta else 'unknown' for score, doc, meta in ranked[:3]]
    return {"context": context, "sources": sources}
    
def diagnose_node(state : AgentState) -> dict :
    
    cache_key = get_cache_key(state["crop"], state["image_url"])
    if cache_key in cache_store:
        print("CACHE HIT — returning cached diagnosis")
        return {"diagnosis": cache_store[cache_key]}
    if state['image_url']:
        user_content = [
        {"type": "text", "text": f"look at the image of a {state['crop']} and analyse the condition"},
        {"type": "image_url", "image_url": {"url": state['image_url']}}
    ]
    else:
        user_content = [
        {"type": "text", "text": f"Based ONLY on the reference material provided, diagnose common diseases in {state['crop']} and provide treatment steps. Do not use any knowledge outside the provided context."}
    ]
    
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You are an expert agronomist. Use this reference material to inform your diagnosis:\n\n{state['context']}\n\nRespond ONLY with a JSON object with exactly these fields: disease_name (string), severity (integer 1-5), description (string), treatment_steps (array of strings). No markdown, no explanation, just raw JSON."   # your expert agronomist instruction here
        },
        {
            "role": "user",
            "content": (
                [
                    {
                        "type": "text",
                        "text": f"look at the image of a {state['crop']} and analyse the condition of the crop and give what the are best pesticides and minimal quantity of pesticide that i can use for the crop to have max crop"
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

def route_after_check(state: AgentState) -> str:
    """Conditional edge — decides which node to go to after image check."""
    if state["check_result"] == "DISEASED_PLANT":
        return "get_context"
    return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("check_image", check_image_node)
workflow.add_node("get_context", get_context_node)
workflow.add_node("diagnose", diagnose_node)

# Add edges
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
    
def stream_diagnosis(state : AgentState):
    """Streams GPT-4o diagnosis token by token for real-time response."""
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You are an expert agronomist. Use this reference material to inform your diagnosis:\n\n{state['context']}\n\nProvide a clear, detailed diagnosis including: disease name, severity, description, and treatment steps with specific pesticide quantities. Write in plain English, not JSON."   # your expert agronomist instruction here
        },
        {
            "role": "user",
            "content": (
                [
                    {
                        "type": "text",
                        "text": f"look at the image of a {state['crop']} and analyse the condition of the crop and give what the are best pesticides and minimal quantity of pesticide that i can use for the crop to have max crop"
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
    stream = True,
    max_tokens=1000
)
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            yield token

    

# Compile
agent = workflow.compile()
    