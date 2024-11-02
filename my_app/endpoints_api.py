from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from pydantic import BaseModel
from uuid import UUID
from typing import List
from my_app.app_module import controller
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline

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
async def analyze_files(request: AnalysisRequest, background_tasks: BackgroundTasks, authorization: str = Header(...)):
    analysis_id = request.analysisId
    selected_model = request.model
    file_paths = request.filePaths

    # Validate the model parameters
    status, info = controller.evaluate_parameters_model_run(selected_model, file_paths)
    if not status:
        raise HTTPException(status_code=400, detail=info)

    # Store the token associated with this analysis_id
    controller.store_token(analysis_id, authorization)

    # Immediate response to the sender
    response = {
        "status": "accepted",
        "info": f"Request for analysis {analysis_id} accepted",
        "analysis_id": analysis_id,
        "files": len(file_paths)
    }

    # Start the background task to run the analysis
    background_tasks.add_task(controller.predict_audios, analysis_id, selected_model, file_paths)

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