import yaml
import csv
import os
from typing import Sequence, Dict
import torch
from torch.utils.data import DataLoader 
from model_module.test_dataset import Dataset_Custom
from model_module.models_manager import get_model


def predict_audios(model_name: str, id_list: Sequence, parameters, base_dir: str, output_csv: str, save_func):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Dataset_Custom(id_list, base_dir=base_dir)
    model, postprocessing = get_model(model_name=model_name, config=parameters, device=device)
    
    data_loader = DataLoader(dataset, batch_size=14, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x, (file_name, idx) in data_loader:
        fname_list = []
        score_list = []
        label_list = []  
        
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        batch_label = postprocessing(batch_out)
        
        batch_score = batch_out.data.cpu().numpy()

        # Add outputs for current batch
        fname_list.extend(file_name)
        score_list.extend(batch_score)
        label_list.extend(batch_label)

        # Call the saving function after each batch
        save_func(fname_list, idx, score_list, label_list, output_csv)

    return file_name, idx, score_list, label_list


def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        config = config['model']
    return config["parameters"]


# Append results for Meso
def save_meso_results(file_names, fragments, scores, labels, output_csv):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file is being created
        if not file_exists:
            writer.writerow(['name', 'fragment', 'score', 'label'])
        
        # Iterate over the data and write to CSV
        for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
            writer.writerow([fname, fragment.item(), score[0], label.item()])


# Append results for Wav2Vec
def save_wav2vec_results(file_names, fragments, scores, labels, output_csv):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file is being created
        if not file_exists:
            writer.writerow(['name', 'fragment', 'score', 'label'])
        
        # Iterate over the data and write to CSV
        for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
            # Convert score and label tensors to lists
            score_tuple = score[1]
            label_tuple = label[1].item()
            writer.writerow([fname, fragment.item(), score_tuple, label_tuple])


if __name__ == '__main__':
    model_meso = 'mesonet'
    model_wav = 'wav2vec'
    list_IDs = ["audio_1.m4a", "audio_2.m4a"]
    meso_config = load_config('config_files/config_mesonet.yaml')
    wav2vec_config = load_config('config_files/config_wav2vec.yaml')
    base_dir = '../datasets/test_dataset'

    meso_output_csv = '../meso_results.csv'
    wav2vec_output_csv = '../wav2vec_results.csv'

    # Predict and save MesoNet results batch by batch
    # predict_audios(model_meso, list_IDs, meso_config, base_dir, meso_output_csv, save_meso_results)

    # Predict and save Wav2Vec results batch by batch
    file_name, idx, score_list, label_list = predict_audios(model_wav, list_IDs, wav2vec_config, base_dir, wav2vec_output_csv, save_wav2vec_results)

