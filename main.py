from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv 
from openai import OpenAI
import os
import json
import re
import chromadb


load_dotenv()
app = FastAPI()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="crop_diseases")


class Diagnose(BaseModel):
    image_url: str
    crop: str
    
    
class DiagnoseResponse(BaseModel):
    disease_name : str
    severity : int
    description : str
    treatment_steps : list[str]
    sources: list[str]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"Status":"ArigMind Test Api"}

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
@app.post("/diagnose")
def diagnose(dig:Diagnose):
    check_result = check_image(dig.image_url)
    if check_result == "NOT_A_PLANT":
        return {"error": "Image is not a plant. Please upload a crop photo."}
    if check_result == "HEALTHY_PLANT":
        return {"error": "Plant appears healthy. No disease detected."}
    
    query_text = f"{dig.crop} common diseases symptoms treatment"
    query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query_text
    ).data[0].embedding
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3)
    context = "\n\n".join(results['documents'][0])
    sources = results['metadatas'][0]
    sources = [s.get('source', 'unknown') if s else 'unknown' for s in sources]
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": f"You are an expert agronomist. Use this reference material to inform your diagnosis:\n\n{context}\n\nRespond ONLY with a JSON object with exactly these fields: disease_name (string), severity (integer 1-5), description (string), treatment_steps (array of strings). No markdown, no explanation, just raw JSON."   # your expert agronomist instruction here
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"look at the image  of a {dig.crop} and analyse the condition of the crop and give what the are best pesticides and minimal quantity of pesticide that i can use for the crop to have max crop"  # your sentence using dig.crop
                },
                {
                    "type": "image_url",
                    "image_url": {"url": dig.image_url}  # use dig.image_url here
                }
            ]
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
    return DiagnoseResponse(**parsed, sources = sources)



