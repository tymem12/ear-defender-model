from fastapi.testclient import TestClient
from uuid import uuid4
from my_app.app_module import controller
from my_app.endpoints_api import app
from dotenv import load_dotenv
import pytest

client = TestClient(app)

def test_analyze_files_model_endpoint_mesonet():
    # Prepare test data
    test_analysis_id = uuid4()
    test_model = "mesonet"
    test_file_paths = [ {
        'filePath': "file.jpg",
        'link': "link.com"
        }, {
        'filePath': "file2.jpg",
        'link': "link2.com"
        }]

    # Expected data
    request_data = {
        "analysisId": str(test_analysis_id),
        "model": test_model,
        "files": test_file_paths,
    }
    load_dotenv()

    # Mock the controller's predict_audios and evaluate_parameters_model_run methods if necessary
    controller.predict_audios = lambda *args, **kwargs: {"status": "accepted", "model": "mesonet"}
    controller.evaluate_parameters_model_run = lambda *args, **kwargs: (True, "Valid parameters")

    # Send POST request to /model with authorization header
    response = client.post("/model/run", json=request_data, headers={"authorization": "Bearer test_token"})

    # Print the response content for debugging
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Assert response status code and content
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "accepted"
    assert response_json["info"].startswith("Request for analysis")
    assert response_json["analysis_id"] == str(test_analysis_id)
    assert response_json["files"] == len(test_file_paths)


def test_analyze_files_model_endpoint_wav_to_vec():
    # Prepare test data
    test_analysis_id = uuid4()
    test_model = "wav2vec"
    test_file_paths = [ {
        'filePath': "file.jpg",
        'link': "link.com"
        }, {
        'filePath': "file2.jpg",
        'link': "link2.com"
        }]

    # Expected data
    request_data = {
        "analysisId": str(test_analysis_id),
        "model": test_model,
        "files": test_file_paths,
    }
    load_dotenv()

    # Mock the controller's predict_audios and evaluate_parameters_model_run methods if necessary
    controller.predict_audios = lambda *args, **kwargs: {"status": "accepted", "model": "wav2vec"}
    controller.evaluate_parameters_model_run = lambda *args, **kwargs: (True, "Valid parameters")

    # Send POST request to /model with authorization header
    response = client.post("/model/run", json=request_data, headers={"authorization": "Bearer test_token"})

    # Print the response content for debugging
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())

    # Assert response status code and content
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "accepted"
    assert response_json["info"].startswith("Request for analysis")
    assert response_json["analysis_id"] == str(test_analysis_id)
    assert response_json["files"] == len(test_file_paths)


