from langgraph.graph import StateGraph, END
from typing import TypedDict
import main
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv()
class AgentState(TypedDict):
     image_url: str
     crop: str
     check_result: str
     context: str
     sources: list[str]
     diagnosis: dict

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    """ Simple node that checks the image and gives a respose wether the given image is plant or not and it also give if the plant is diseased o healthy"""
    
    