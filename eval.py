from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from datasets import Dataset
from agent import run_eval
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
golden_dataset = [
    { 
     "question" : " why is wicth broom casued in blue berries and how can the disease be cured" ,
     "ground_truth":"Witches' broom is caused by a rust fungus that infects both blueberry bushes and fir trees. This disease causes clusters of small branches known as witches' brooms to form at the base of the plant. Branches with witches' broom have small leaves and spongy, reddish bar and it can be cured by Infected blueberry plants should be dug up and burned, buried or composted." ,
     "contexts":[],
     "crop": "Blueberry"
     
    },
    {
        "question": "What disease is occured when a tomato plant is bring sunlight for more time and how can it be avoided?",      
        "ground_truth": "Sunscald occurs on tomato fruit that have been exposed to too much sun. This is common in plants that have lost leaves from a leaf spot disease or insect feeding, but can also occur on plants that are over pruned or on fruit that are otherwise exposed to the sun.",  
        "contexts": [],
        "crop":"Tomato"
    },
    {
    "question": "What disease affects apple trees and how should it be treated?",
    "ground_truth": "Apple scab is caused by a fungus that infects leaves and fruit. Treatment includes planting disease-resistant varieties and applying fungicides with proper timing.",
    "contexts": [],
    "crop":"Apple"
    
    }
]
results = []

for item in golden_dataset:
    crop = item["question"].split()[0]  # extracts "why", "What", "What" — wrong  # extract first word as crop hint
    result = run_eval(item["question"], item["crop"])
    
    results.append({
        "question": item["question"],
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": item["ground_truth"]
    })
dataset = Dataset.from_list(results)

scores = evaluate(
    dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
    llm=llm,
    embeddings = embeddings
)

print(scores)