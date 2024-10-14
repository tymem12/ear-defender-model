import os
import csv

from my_app.model_module.evaluate_audios import predict
from my_app.model_module.model_manager import load_config
from my_app.model_module.model_manager import ModelManager


from typing import List
from dotenv import load_dotenv


def save_results_to_csv(file_names, fragments, scores, labels, output_csv):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['name', 'fragment', 'score', 'label'])
        
        # Iterate over the data and write to CSV
        for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
            writer.writerow([fname, fragment, score, label])


# Append results for Wav2Vec
# def save_wav2vec_results(file_names, fragments, scores, labels, output_csv):
#     file_exists = os.path.isfile(output_csv)
#     with open(output_csv, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         # Write the header if the file is being created
#         if not file_exists:
#             writer.writerow(['name', 'fragment', 'score', 'label'])
        
#         # Iterate over the data and write to CSV
#         for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
#             # Convert score and label tensors to lists
#             score_tuple = score[1]
#             label_tuple = label[1].item()
#             writer.writerow([fname, fragment.item(), score_tuple, label_tuple])


def predict_audios_csv(input_csv: List[str], audio_dir: str, configuration_file :str, output_path: str):
    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        list_IDs = [row[0] for row in reader]
    
    config = load_config(configuration_file)
    model_name = config['model']['name']
    model_manager = ModelManager(model_name)
    model_manager.config_file = configuration_file
    print('TUTAJ')
    print(model_manager.config_file)

    for file in list_IDs:
        files, fragments, scores, labels = predict(model_manager=model_manager, file_paths=[file], base_dir= audio_dir)
        save_results_to_csv(files, fragments, scores, labels, output_csv=output_path)



if __name__ == '__main__':
    load_dotenv()
    predict_audios_csv('example.csv', 'test_dataset', 'config_files/config_mesonet.yaml', 'example_results.csv')

