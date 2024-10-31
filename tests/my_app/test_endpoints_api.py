from fastapi.testclient import TestClient
from uuid import uuid4
from my_app.app_module import controller
from my_app.endpoints_api import app
from dotenv import load_dotenv

client = TestClient(app)

def test_analyze_files_model_endpoint_mesonet():
    # Prepare test data
    test_analysis_id = uuid4()
    test_model = "mesonet"
    test_file_paths = ["path/to/file1.wav", "path/to/file2.wav"]

    # Expected data
    request_data = {
        "analysisId": str(test_analysis_id),
        "model": test_model,
        "filePaths": test_file_paths,
    }
    load_dotenv()

    # Mock the controller's predict_audios method if necessary
    # to return a known response
    controller.predict_audios = lambda *args, **kwargs: {"status": "success", "model": "mesonet"}

    # Send POST request to /model
    response = client.post("/model/run", json=request_data)

    # Assert response status code and content
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    assert response_json["model"] == "mesonet"


def test_analyze_files_model_endpoint_wav_to_vec():
    # Prepare test data
    test_analysis_id = uuid4()
    test_model = "wav2vec"
    test_file_paths = ["path/to/file1.wav", "path/to/file2.wav"]

    # Expected data
    request_data = {
        "analysisId": str(test_analysis_id),
        "model": test_model,
        "filePaths": test_file_paths,
    }
    load_dotenv()

    # Mock the controller's predict_audios method if necessary
    # to return a known response
    controller.predict_audios = lambda *args, **kwargs: {"status": "success", "model": "wav2vec"}

    # Send POST request to /model
    response = client.post("/model/run", json=request_data)

    # Assert response status code and content
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    assert response_json["model"] == "wav2vec"
