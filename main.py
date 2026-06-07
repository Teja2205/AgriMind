from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv 
from openai import OpenAI
import os


load_dotenv()
app = FastAPI()

class Diagnose(BaseModel):
    image_url: str
    crop: str
    

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
            "content": "You are an expert agronomist who have knowledge on all the crops, the desieases that crops get and based on the crops disease stage tell what pestiside to use and why"   # your expert agronomist instruction here
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
    return {"diagnosis": response.choices[0].message.content}
