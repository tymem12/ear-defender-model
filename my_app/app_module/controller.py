from datetime import datetime
import logging
import os
import time
from typing import List, Dict
from my_app.model_module.evaluate_audios import predict
from my_app.model_module.prediction_pipeline.model_factory import PredictionPipeline, ModelFactory
from my_app.app_module import client_API
from my_app import utils
from my_app.model_module import metrics
import httpx
import asyncio
from collections import defaultdict
from asyncio import Lock

TOKENS = defaultdict(str)
TOKEN_LOCK = Lock()

async def store_token(analysis_id, token):
    async with TOKEN_LOCK:
        TOKENS[analysis_id] = token[-1]

async def evaluate_parameters_model_run(selected_model: str, files: List[Dict[str, str]]):

    try:
        PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    except ValueError as e:
        logging.info(f"Error initializing PredictionPipeline: {str(e)}")
        return False, f'Unknown model: {selected_model}'
    
    storage_path = os.getenv('AUDIO_STORAGE')
    valid_files = []

    for file in files:
        file_path = file['filePath']
        if os.path.isfile(f'{storage_path}/{file_path}'):
            valid_files.append(file)
        else:
            logging.warning(f"File not in storage. File removed: {file}")

    if not valid_files:
        return False, "No valid files found in storage"

    return True, f'{len(valid_files)} files passed to analysis' 
    
def get_models():
    return ModelFactory.get_available_models()

async def predict_audios(analysis_id: str, selected_model: str, files: List[Dict[str, str]]):
    logging.info(f'analysis {analysis_id} started')
    file_paths = [file['filePath'] for file in files]
    file_links = [file['link'] for file in files]

    can_access_connector = True
    predictions = []
    token = TOKENS[analysis_id]
    prediction_pipeline = PredictionPipeline(selected_model, return_labels=True, return_scores=False)
    
    async def process_file(file_path, file_link):
        try:
            # Perform predictions
            files, segments_list, labels = predict(prediction_pipeline, [file_path])
            if len(segments_list) != len(labels):
                raise ValueError("Segments and labels lists must have the same length.")

            # Prepare the payload
            model_predictions = [{"segmentNumber": segment, "label": label} for segment, label in zip(segments_list, labels)]
            model_score, model_label = metrics.file_score_and_label(model_predictions)
            payload = {
                "link": file_link,
                "timestamp": datetime.now().isoformat(),
                "model": selected_model,
                "modelPredictions": model_predictions,
                "score": model_score,
                "label": model_label,
                "filePath": file_path
            }

            # Send predictions asynchronously
            status_code, message = await client_API.connector_create_predictions(analysis_id, payload, token)
            if status_code != 200:
                logging.error(f"Error sending predictions for {file_path}: {message}")
                raise RuntimeError(f"Failed to send predictions for {file_path}")

            # Delete file from storage
            utils.delate_file_from_storage(file_path)
            return payload

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise
    tasks = [process_file(file_path, file_link) for file_path, file_link in zip(file_paths, file_links)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Error processing file: {result}")
            client_API.connector_abort_analysis(analysis_id, token)
            can_access_connector = False
            break
    else:
        predictions.append(result)

    if can_access_connector:
        response_code, info = client_API.connector_end_analysis(analysis_id=analysis_id, token=token)
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


def eval_params_eval_dataset(dataset:str, model_conf: str):
    config_path = os.getenv('CONFIG_FOLDER') + '/' + model_conf
    available_datasets = os.getenv('AVAILABLE_DATASETS', '').split(',')
    available_configs = os.getenv('AVAILABLE_CONFIGS', '').split(',')


    if dataset not in available_datasets:
         return False, f'Not correct dataset_name. You passed {dataset} but available_datasets are: {available_datasets}' 
    if model_conf not in available_configs:
         return False, f'Not correct config_path. You passed {model_conf} but available_datasets are: {available_configs}' 

    return True, f'Analysis started for dataset {dataset}. Files are generating '
def eval_dataset(dataset:str, model_conf: str, output_csv : str):
    config_path = os.getenv('CONFIG_FOLDER') + '/' +model_conf


    model_name = utils.load_config(config_path)['model']['name']
    # try:
    files_fake, folder_path_fake = utils.get_files_to_predict(dataset = dataset, status = 'fake')
    files_real, folder_path_real = utils.get_files_to_predict(dataset = dataset, status = 'real')
    # except FileExistsError as e:
    #     return {
    #         "status": "failure",
    #         "info": str(e),
    #     }
    # try:
    prediction_pipeline = PredictionPipeline(model_name, config_path = config_path,return_labels = True, return_scores = True)
    results_folder = os.getenv('RESULTS_CSV')
    duration_list = []

    if files_fake:
        results_csv_path_fake = f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'
        for file in files_fake:
            start_time = time.time()  # Record the start time
            files, fragments, results = predict(prediction_pipeline, file_paths=[file], base_dir=folder_path_fake)
            end_time = time.time()  # Record the end time
            duration = end_time - start_time  # Calculate execution duration
            utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_fake)
            duration_list.append(duration)
            logging.info(file + " saved")

    if files_real:
        results_csv_path_real = f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'
        for file in files_real:
            start_time = time.time()  # Record the start time
            files, fragments, results = predict(prediction_pipeline, file_paths=[file], base_dir=folder_path_real)
            end_time = time.time()  # Record the end time
            duration = end_time - start_time  # Calculate execution duration
            utils.save_results_to_csv(files, fragments, results[0], results[1], results_csv_path_real)
            duration_list.append(duration)
            logging.info(file + " saved")
    combined_files = files_fake + files_real
    utils.save_durations(combined_files, duration_list, f'results_csv/{dataset}/duration_{dataset}_{output_csv}.csv')
    logging.info(f"files saved")
    # return {
    #         "status": "success",
    #         "info": f"files saved in {results_csv_path_fake} and {results_csv_path_real}"
    #     }


def eval_metrics(dataset:str, output_csv:str):
    results_folder = os.getenv('RESULTS_CSV')
    # try:
    scores_spoof, scores_real = utils.get_scores_from_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
    predictions, labels = utils.get_labels_and_predictions_from_csv([f'{results_folder}/{dataset}/{dataset}_fake_{output_csv}.csv'], [f'{results_folder}/{dataset}/{dataset}_real_{output_csv}.csv'])
    # except FileNotFoundError as e:
    #     return{
    #     "status": "failure",
    #     "info": str(e),
    #     }
    acc = metrics.calculate_acc_from_labels(predictions, labels)
    eer , _= metrics.calculate_eer_from_scores(scores_spoof, scores_real)
    if eer and acc:
        return {
            "status": "success",
            "info": 'err calculated',
            "eer" : eer, 
            "acc" : acc
        }
    elif not acc and eer:
        return {
            "status": "success",
            "info": 'Cannot calculate accuracy',
            "eer" : eer, 
            "acc" : 'N/A'
        }
    
    elif acc and not eer:
        return {
            "status": "success",
            "info": 'Cannot calculate eer',
            "eer" : 'N/A', 
            "acc" : acc
        }

    else:
        return {
            "status": "failure",
            "info": 'Cannot calculate eer',
            "eer" : 'N/A' ,
            "acc" : 'N/A'

        }









