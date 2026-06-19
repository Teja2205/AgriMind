from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv 
from fastapi.responses import StreamingResponse
from agent import agent, stream_diagnosis, AgentState, check_image, get_context_node
from fastapi import Header, HTTPException, Depends
import os

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

def verify_api_key(x_api_key: str = Header(...)):
    key = os.getenv("AGRIMIND_API_KEY")
    if x_api_key == key:
        return True
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
@app.get("/")
def root():
    return {"Status":"ArigMind Test Api"}

@app.post("/diagnose")
def diagnose(dig: Diagnose, api_key: bool = Depends(verify_api_key)):
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

@app.post("/diagnose/stream")
def diagnose_stream(dig: Diagnose, api_key: bool = Depends(verify_api_key)):
    
    check_result = check_image(dig.image_url)
    
    if check_result == "NOT_A_PLANT":
        return {"error": "Image is not a plant. Please upload a crop photo."}
    
    if check_result == "HEALTHY_PLANT":
        return {"error": "Plant appears healthy. No disease detected."}
    
    context_result = get_context_node({
        "image_url": dig.image_url,
        "crop": dig.crop,
        "check_result": check_result,
        "context": "",
        "sources": [],
        "diagnosis": {}
    })
    
    state = {
        "image_url": dig.image_url,
        "crop": dig.crop,
        "check_result": check_result,
        "context": context_result["context"],
        "sources": context_result["sources"],
        "diagnosis": {}
    }
    
    return StreamingResponse(
        stream_diagnosis(state),
        media_type="text/plain"
    )



