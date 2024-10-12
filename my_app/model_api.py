from fastapi import FastAPI
from pydantic import BaseModel
from uuid import UUID
from typing import List
from app import controller

app = FastAPI()

# Define a Pydantic model for request body
class AnalysisRequest(BaseModel):
    analysisId: UUID
    model: str
    filePaths: List[str]

@app.post("/model")
async def analyze_files(request: AnalysisRequest):
    # Extract the received data
    analysis_id = request.analysisId
    selected_model = request.model
    file_paths = request.filePaths

    # You can add processing logic here
    results = controller.predict_audios(analysis_id, selected_model, file_paths)
    return results


    # Example: Return a response with the data received
    # return {
    #     "message": "Analysis started",
    #     "analysis_id": analysis_id,
    #     "model": selected_model,
    #     "filePaths": file_paths
    # }

# For running the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_api:app", host="0.0.0.0", port=8000, reload=True)