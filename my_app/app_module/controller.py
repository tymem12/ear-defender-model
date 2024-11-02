from uuid import UUID
import requests
from datetime import datetime
from my_app.model_module.evaluate_audios import predict
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline, ModelFactory
from my_app.app_module import client_API
from my_app import utils
from my_app.model_module import metrics

import os
from typing import List, Dict
import logging

TOKENS = {}

def store_token(analysis_id, token):
    TOKENS[analysis_id] = token

def evaluate_parameters_model_run(selected_model: str, file_paths):
    try:
        PredictionPipeline(selected_model)
    except ValueError as e:
        return False, f'Unknown model: {selected_model}'
    
    storage_path = os.getenv('AUDIO_STORAGE')
    for file in file_paths:
        if not os.path.isfile(f'{storage_path}/{file}'):
            return False, f'File does not exists in storage'
        
    return True, f'{len(file_paths)} passed to analysis'    
    
def get_models():
    return ModelFactory.get_available_models()



def predict_audios(analysis_id : UUID, selected_model: str, file_paths: List[str]):
    # get mo
    can_access_connector = True
    start_analysis = datetime.now().isoformat()
    predictions = []
    token = TOKENS[analysis_id]
    try:
        prediction_pipeline = PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    except ValueError as e:
        # Log error as the response has already been sent
        print(f"Error initializing PredictionPipeline: {str(e)}")
        return  # Exit if model initialization fails
    
    for link in file_paths:
        files ,segments_list, labels = predict(prediction_pipeline, [link])

        if len(segments_list) != len(labels):
            print("Segments and labels lists must have the same length.")
            continue  # Log and skip this file if there's an inconsistency
        model_predictions = [{"segmentNumber": segment, "label": label} for segment, label in zip(segments_list, labels)]
        payload = {
            "link": link,
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "modelPredictions": model_predictions
        }
        if can_access_connector:
            create_pred_res = client_API.connector_create_predictions(payload, token=token) #methods that send rtequest to diffrent API
            if create_pred_res.get('status', 'success') == 'failure': 
                can_access_connector = False 
        predictions.append(payload)
    
    
    status = 'FINISHED'
    end_analysis = datetime.now().isoformat()

    file_count = len(predictions)
    if can_access_connector:
        upate_analysis_res = client_API.connector_update_analysis(analysis_id=analysis_id, status=status, finishTimestamp=end_analysis, predictionResults=predictions, token =token ) #methods that send rtequest to diffrent API

    #     return {
    #             "status": "success",
    #             "info": f'{file_count} files were analyzed',
    #             "files_count" : file_count,
    #             'analysis' : analysis_id,
    #             'timestamp_start' : start_analysis,
    #             'timestamp_end' : end_analysis,  # For example, "Unknown model: <model_name>"
    #             'model': selected_model,
    #             'predictions': predictions
    #         }
    # return {
    #             "status": "failure",
    #             "info": f'Could not access connector, analysis not saved',
    #             "files_count" : file_count,
    #             'analysis' : analysis_id,
    #             'timestamp_start' : start_analysis,
    #             'timestamp_end' : end_analysis,  # For example, "Unknown model: <model_name>"
    #             'model': selected_model,
    #             'predictions': predictions
    #         }

        

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


def eval_params_eval_dataset(dataset:str, model_conf: str, output_csv = str):
    config_path = os.getenv('CONFIG_FOLDER') + '/' +model_conf
    available_datasets = ['deep_voice', 'fake_audio', 'my_eng', 'my_pol', 'release_in_the_wild', 'test_dataset', 'example']
    available_configs = ['config_mesonet.yaml', 'config_mesonet_finetuned.yaml', 'config_wav2vec.yaml']

    if dataset not in available_datasets:
         return False, f'Not correct dataset_name. You passed {dataset} but available_datasets are: {available_datasets}' 
    if model_conf not in available_configs:
         return False, f'Not correct config_path. You passed {model_conf} but available_datasets are: {available_configs}' 

    return True, f'Analysis started for dataset {dataset}. Files are generating '
def eval_dataset(dataset:str, model_conf: str, output_csv : str):
    config_path = os.getenv('CONFIG_FOLDER') + '/' +model_conf

    # config_path = os.getenv('CONFIG_FOLDER') + '/' +model_conf
    # available_datasets = ['deep_voice', 'fake_audio', 'my_eng', 'my_pol', 'release_in_the_wild', 'test_dataset', 'example']
    # if dataset not in available_datasets:
    #      return {
    #         "status": "failure",
    #         "info": 'Not correct dataset_name. available_datasets are: ' + available_datasets,
    #     }
        
    
    
    model_name = utils.load_config(config_path)['model']['name']
    try:
        files_fake, folder_path_fake = utils.get_files_to_predict(dataset = dataset, status = 'fake')
        files_real, folder_path_real = utils.get_files_to_predict(dataset = dataset, status = 'real')
    except FileExistsError as e:
        return {
            "status": "failure",
            "info": str(e),
        }
    # try:
    prediction_pipline = PredictionPipeline(model_name, config_path = config_path,return_labels = True, return_scores = True)
    # except ValueError as e:
    #     # Handle the case where an unknown model name is provided
    #     return {
    #         "status": "failure",
    #         "info": str(e)  # For example, "Unknown model: <model_name>"
    #     }
    results_folder = os.getenv('RESULTS_CSV')
    results_csv_path_fake = f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'
    for file in files_fake:
        files, fragments, results = predict(prediction_pipline, file_paths=[file], base_dir= folder_path_fake)
        utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_fake)

    results_csv_path_real = f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'
    for file in files_real:
        files, fragments, results = predict(prediction_pipline, file_paths=[file], base_dir= folder_path_real)
        utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_real)
    
    return {
            "status": "success",
            "info": f"files saved in {results_csv_path_fake} and {results_csv_path_real}"
        }



def eval_metrics(dataset:str, output_csv):
    results_folder = os.getenv('RESULTS_CSV')
    try:
        predictions, labels = utils.get_labels_and_predictions_from_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
    except FileNotFoundError as e:
        return{
        "status": "failure",
        "info": str(e),
        }
    eer = metrics.calculate_eer_from_labels(predictions, labels)
    # results = metrics.calculate_eer_from_labels_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
    return {
        "status": "success",
        "info": 'err calculated',
        "results" : eer  # For example, "Unknown model: <model_name>"}
    }











