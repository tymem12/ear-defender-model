from typing import List, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from my_app.app_module import controller
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

app = FastAPI()

# Define a Pydantic model for request body
class AnalysisRequest(BaseModel):
    analysisId: str
    model: str
    files: List[Dict[str, str]]


class StorageContent(BaseModel):
    filePaths: List[str]

    
class TestContent(BaseModel):
    model_conf: str
    dataset: str
    csv_file_id:str

class TestContentMetrics(BaseModel):
    dataset: str
    csv_file_id:str

# Thread pool executor setup
executor = ThreadPoolExecutor(max_workers=1)
executor_lock = threading.Lock()

# Placeholder controller class (Replace with actual implementation)


@app.post("/model/run")
async def analyze_files(request: AnalysisRequest, authorization: str = Header(...)):
    analysis_id = request.analysisId
    selected_model = request.model
    files = request.files

    # Validate the model parameters
    status, info = controller.evaluate_parameters_model_run(selected_model, files)

    if not status:
        raise HTTPException(status_code=400, detail=info)

    # Store the token associated with this analysis_id
    controller.store_token(analysis_id, authorization)

    # Immediate response to the sender
    response = {
        "status": "accepted",
        "info": f"Request for analysis {analysis_id} accepted",
        "analysis_id": analysis_id,
        "files": len(files),
    }

    # Start the task using ThreadPoolExecutor
    with executor_lock:
        executor.submit(controller.predict_audios, analysis_id, selected_model, files)

    return response



@app.post("/model/storage")
async def analyze_files(request: StorageContent):
    # Extract the received data
    file_paths = request.filePaths

    # You can add processing logic here
    results = controller.storage_content(file_paths)
    return results


@app.post("/model/eval_dataset")
async def eval_dataset(request: TestContent, background_tasks: BackgroundTasks):
    # Extract the received data
    model_conf = request.model_conf
    dataset = request.dataset
    status, info = controller.eval_params_eval_dataset(dataset, model_conf)
    if not status:
        raise HTTPException(status_code=400, detail=info)

    background_tasks.add_task(controller.eval_dataset,dataset = dataset, model_conf = model_conf, output_csv = request.csv_file_id)
    return info

    # You can add processing logic here

@app.post("/model/eval_metrics")
async def test_metrics(request: TestContentMetrics):
    dataset = request.dataset
    # You can add processing logic here
    results = controller.eval_metrics(dataset = dataset, output_csv = request.csv_file_id)
    return results

@app.get("/config/model")
async def test_metrics():
    # You can add processing logic here
    results = controller.get_models()
    return results


# For running the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("endpoints_api:app", host="0.0.0.0", port=8000, reload=True)