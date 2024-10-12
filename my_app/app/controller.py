from uuid import UUID
import requests
from datetime import datetime
from model_module.evaluate_dataset import predict
import os
import json


def predict_audios(analysis_id: UUID, selected_model: str, file_paths: list[str]):
    # get mo

    start_analysis = datetime.now().isoformat()
    
    for link in file_paths:
        _,segments_list, labels = predict(selected_model, [link])
        # update database to load to the specific document 

        if len(segments_list) != len(labels):
            raise ValueError("Segments and labels lists must have the same length.")
        
        # Prepare modelPredictions
        model_predictions = [{"segmentNumber": segment, "label": label} for segment, label in zip(segments_list, labels)]
        
        # Prepare the payload for the request
        





def connector_create_predictions(link: str, model: str, model_predictions: list[dict]):
    # Ensure the length of segments and labels match
    
    payload = {
            "link": link,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "modelPredictions": model_predictions
        }
    # Get the URL from environment variables
    connector_url = os.getenv('CONNECTOR_URL')
    
    if connector_url is None:
        raise ValueError("CONNECTOR_URL environment variable is not set.")
    
    # Send the request
    headers = {'Content-Type': 'application/json'}
    response = requests.post(connector_url, data=json.dumps(payload), headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()



# to finisz
def connector_update_analysis(analysis_id: str, status: str, start_timestamp: str, duration: int, 
                              starting_point: str, depth: int, max_files: int, model: str, 
                              file_count: int, predictions: list[dict]):
    """
    Sends an update to the connector with analysis results.

    :param analysis_id: Unique ID of the analysis
    :param status: Current status of the analysis (DOWNLOADING/PROCESSING/FINISHED)
    :param start_timestamp: Timestamp of when the analysis started (ISO 8601 format)
    :param duration: Duration of the analysis in seconds
    :param starting_point: Initial URL of the scraper
    :param depth: Depth of the scraper's search
    :param max_files: Maximum number of files the scraper can scrape
    :param model: Model used for analysis
    :param file_count: Number of files analyzed
    :param predictions: List of predictions for each file, with each entry being a dict
                        that contains link, timestamp, model, and modelPredictions
    """

    # Prepare inputParams
    input_params = {
        "startingPoint": starting_point,
        "depth": depth,
        "maxFiles": max_files,
        "model": model
    }

    # Prepare the payload for the request
    payload = {
        "id": analysis_id,
        "status": status,
        "timestamp": start_timestamp,  # ISO 8601 timestamp for when the analysis started
        "duration": duration,
        "inputParams": input_params,
        "fileCount": file_count,
        "predictions": predictions
    }

    # Get the URL from environment variables
    connector_url = os.getenv('CONNECTOR_URL')
    
    if connector_url is None:
        raise ValueError("CONNECTOR_URL environment variable is not set.")
    
    # Send the request
    headers = {'Content-Type': 'application/json'}
    response = requests.post(connector_url, data=json.dumps(payload), headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()