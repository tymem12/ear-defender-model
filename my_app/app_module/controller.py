# from uuid import UUID
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
    token_elem = token.split()
    TOKENS[analysis_id] = token_elem[-1]

def evaluate_parameters_model_run(selected_model: str, file_paths):
    # if not ModelStore.get(selected_model):
    try:
        PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    except ValueError as e:
        logging.info(f"Error initializing PredictionPipeline: {str(e)}")
        return False, f'Unknown model: {selected_model}'
    
    storage_path = os.getenv('AUDIO_STORAGE')
    for file in file_paths:
        if not os.path.isfile(f'{storage_path}/{file}'):
            return False, f'File does not exists in storage'
        
    return True, f'{len(file_paths)} passed to analysis'    
    
def get_models():
    return ModelFactory.get_available_models()



def predict_audios(analysis_id : str, selected_model: str, file_paths: List[str]):
    logging.info(f'analysis {analysis_id} started')

    can_access_connector = True
    predictions = []
    token = TOKENS[analysis_id]
    prediction_pipeline = PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    
    for link in file_paths:
        files ,segments_list, labels = predict(prediction_pipeline, [link])

        if len(segments_list) != len(labels):
            logging.info("Segments and labels lists must have the same length.")
            continue  # Log and skip this file if there's an inconsistency
        model_predictions = [{"segmentNumber": segment, "label": label} for segment, label in zip(segments_list, labels)]
        payload = {
            "link": link,
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "modelPredictions": model_predictions
        }
            # logging.info(client_API.connector_create_predictions(analysis_id = analysis_id, payload= payload, token=token)) #methods that send rtequest to diffrent API
        response_code, info = client_API.connector_create_predictions(analysis_id = analysis_id, payload= payload, token=token) #methods that send rtequest to diffrent API
        utils.delate_file_from_storage(link)
        if response_code != 200:
            logging.info(f": {link} predictions not send to the connector: {info}" )
            client_API.connector_abort_analysis(analysis_id, token)
            can_access_connector = False
            break
        else:
            predictions.append(payload)
            logging.info(f"analysis {analysis_id} updated with new prediction for file: {link} " )

    if can_access_connector:
        response_code, info = client_API.connector_end_analysis(analysis_id=analysis_id,  token =token ) #methods that send rtequest to diffrent API
        if response_code == 200:
            logging.info(f'analysis {analysis_id} finished')
        else:
            client_API.connector_abort_analysis(analysis_id, token)
            
    else:
        logging.info(f'analysis {analysis_id} was aborted')


        

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
    available_configs = ['config_mesonet.yaml', 'config_mesonet_finetuned.yaml', 'config_mesonet_ft_asp.yaml','config_wav2vec.yaml']

    if dataset not in available_datasets:
         return False, f'Not correct dataset_name. You passed {dataset} but available_datasets are: {available_datasets}' 
    if model_conf not in available_configs:
         return False, f'Not correct config_path. You passed {model_conf} but available_datasets are: {available_configs}' 

    return True, f'Analysis started for dataset {dataset}. Files are generating '
def eval_dataset(dataset:str, model_conf: str, output_csv : str):
    config_path = os.getenv('CONFIG_FOLDER') + '/' +model_conf


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
    results_folder = os.getenv('RESULTS_CSV')

    results_csv_path_fake = f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'
    for file in files_fake:
        files, fragments, results = predict(prediction_pipline, file_paths=[file], base_dir= folder_path_fake)
        utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_fake)
        logging.info(file + " saved")

    results_csv_path_real = f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'
    for file in files_real:
        files, fragments, results = predict(prediction_pipline, file_paths=[file], base_dir= folder_path_real)
        utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_real)
        logging.info(file + " saved")

    logging.info(f"files saved in {results_csv_path_fake} and {results_csv_path_real}")
    return {
            "status": "success",
            "info": f"files saved in {results_csv_path_fake} and {results_csv_path_real}"
        }



# def eval_metrics(dataset:str, output_csv):
#     results_folder = os.getenv('RESULTS_CSV')
#     try:
#         predictions, labels = utils.get_labels_and_predictions_from_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
#     except FileNotFoundError as e:
#         return{
#         "status": "failure",
#         "info": str(e),
#         }
#     eer = metrics.calculate_eer_from_labels(predictions, labels)
#     return {
#         "status": "success",
#         "info": 'err calculated',
#         "results" : eer  # For example, "Unknown model: <model_name>"}
#     }
def eval_metrics(dataset:str, output_csv):
    results_folder = os.getenv('RESULTS_CSV')
    try:
        scores_spoof, scores_real = utils.get_scores_from_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
    except FileNotFoundError as e:
        return{
        "status": "failure",
        "info": str(e),
        }
    eer , _= metrics.calculate_eer_from_scores(scores_spoof, scores_real)
    if eer:
        return {
            "status": "success",
            "info": 'err calculated',
            "results" : eer  # For example, "Unknown model: <model_name>"}
        }
    else:
        return {
            "status": "failure",
            "info": _,
        }









