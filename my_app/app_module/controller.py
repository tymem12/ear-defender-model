from uuid import UUID
import requests
from datetime import datetime
from my_app.model_module.evaluate_audios import predict
from my_app.model_module.model_manager import ModelManager
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline

import os
import json
from typing import List, Dict
import logging

def predict_audios(analysis_id : UUID, selected_model: str, file_paths: List[str]):
    # get mo

    start_analysis = datetime.now().isoformat()
    predictions = []
    # model_manager = ModelManager(selected_model)



    prediction_pipline = PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    
    for link in file_paths:
        # return_dict =  {'file_path' : link}        

        files ,segments_list, labels = predict(prediction_pipline, [link])

        # update database to load to the specific document 

        if len(segments_list) != len(labels):
            raise ValueError("Segments and labels lists must have the same length.")
        
        # Prepare modelPredictions
        model_predictions = [{"segmentNumber": segment, "label": label} for segment, label in zip(segments_list, labels)]
        # return_dict['model_predictions'] = model_predictions
        # return_list.append(return_dict)

        payload = {
            "link": link,
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "modelPredictions": model_predictions
        }

        # print(connector_create_predictions(payload))
        predictions.append(payload)
    
    
    status = 'FINISHED'
    end_analysis = datetime.now().isoformat()

    file_count = len(predictions)
    # connector_update_analysis(analysis_id=analysis_id, status=status, finishTimestamp=end_analysis, predictionResults=predictions)


    # return return_list
    return predictions

        


def storage_content(file_paths: List[str]):
    results = {}
    folder_path = os.getenv('AUDIO_STORAGE')
    for file_path in file_paths:
        audio_path = os.path.join(folder_path, file_path)
        if not os.path.exists(audio_path):
            logging.info(f"File {audio_path} doesn't exist.")
            # Get the number of files in the base directory and print
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            logging.info(f"Number of files in the folder '{folder_path}': {num_files}")
            results[file_path] = False
        else:
            results[file_path] = True
    return results






def connector_create_predictions(payload: Dict):
    # Ensure the length of segments and labels match
    
    # Get the URL from environment variables
    connector_url = os.getenv('CONNECTOR_ADDRESS') + ':' + os.getenv('CONNECTOR_PORT') +'/predictions'
    
    if connector_url is None:
        raise ValueError("CONNECTOR_URL environment variable is not set.")
    
    # Send the request
    headers = {'Content-Type': 'application/json'}
    response = requests.post(connector_url, data=json.dumps(payload), headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json(), 



def connector_update_analysis(analysis_id: str, status: str, finishTimestamp: str, predictionResults : List[Dict]):

    # Prepare the payload for the request
    payload = {
        "status": status,
        "finishTimestamp": finishTimestamp,  # ISO 8601 timestamp for when the analysis started
        "predictionResults": predictionResults
    }

    # Get the URL from environment variables
    connector_url = os.getenv('CONNECTOR_ADDRESSL') + ':' + os.getenv('CONNECTOR_PORT') + '/analyses/' + analysis_id + '/'
    
    if connector_url is None:
        raise ValueError("CONNECTOR_URL environment variable is not set.")
    
    # Send the request
    headers = {'Content-Type': 'application/json'}
    response = requests.put(connector_url, data=json.dumps(payload), headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()