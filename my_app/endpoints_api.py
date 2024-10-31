from fastapi import FastAPI
from pydantic import BaseModel
from uuid import UUID
from typing import List
from my_app.app_module import controller

app = FastAPI()

# Define a Pydantic model for request body
class AnalysisRequest(BaseModel):
    analysisId: UUID
    model: str
    filePaths: List[str]


class StorageContent(BaseModel):
    filePaths: List[str]

    
class TestContent(BaseModel):
    model_conf: str
    dataset: str
    csv_file_id:str

class TestContentMetrics(BaseModel):
    dataset: str
    csv_file_id:str

@app.post("/model/run")
async def analyze_files(request: AnalysisRequest):
    # Extract the received data
    analysis_id = request.analysisId
    selected_model = request.model
    file_paths = request.filePaths

    # You can add processing logic here
    results = controller.predict_audios(analysis_id, selected_model, file_paths)
    return results


@app.post("/model/storage")
async def analyze_files(request: StorageContent):
    # Extract the received data
    file_paths = request.filePaths

    # You can add processing logic here
    results = controller.storage_content(file_paths)
    return results


@app.post("/model/eval_dataset")
async def test_dataset(request: TestContent):
    # Extract the received data
    model_conf = request.model_conf
    dataset = request.dataset
    results = controller.eval_dataset(dataset = dataset, model_conf = model_conf, output_csv = request.csv_file_id)
    return results

    # You can add processing logic here

@app.post("/model/eval_metrics")
async def test_dataset(request: TestContentMetrics):
    dataset = request.dataset

    # You can add processing logic here
    results = controller.eval_metrics(dataset = dataset, output_csv = request.csv_file_id)

    return results

# For running the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("endpoints_api:app", host="0.0.0.0", port=8000, reload=True)