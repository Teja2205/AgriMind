from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv 
from openai import OpenAI
import os
import json
import re


load_dotenv()
app = FastAPI()

class Diagnose(BaseModel):
    image_url: str
    crop: str
    
    
class DiagnoseResponse(BaseModel):
    disease_name : str
    severity : int
    description : str
    treatment_steps : list[str]
    

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"Status":"ArigMind Test Api"}


@app.post("/diagnose")
def diagnose(dig:Diagnose):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content":"You are an expert agronomist. Respond ONLY with a JSON object, no other text. The JSON must have exactly these fields: disease_name (string), severity (integer 1-5), description (string), treatment_steps (array of strings). No markdown, no explanation, just the raw JSON object."   # your expert agronomist instruction here
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
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    raw = match.group() if match else raw
    parsed = json.loads(raw)
    return DiagnoseResponse(**parsed)

