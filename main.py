from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv 
from agent import agent, AgentState


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
    sources: list[str]


@app.get("/")
def root():
    return {"Status":"ArigMind Test Api"}

@app.post("/diagnose")
def diagnose(dig: Diagnose):
    result = agent.invoke({
        "image_url": dig.image_url,
        "crop": dig.crop,
        "check_result": "",
        "context": "",
        "sources": [],
        "diagnosis": {}
    })
    
    if "error" in result:
        return result
    
    diagnosis = result["diagnosis"]
    diagnosis["sources"] = result["sources"]
    return DiagnoseResponse(**diagnosis)



