import os
import csv
from typing import List
from dotenv import load_dotenv
from my_app.model_module.evaluate_audios import predict
from my_app.model_module.utils import load_config
from my_app.model_module.prediction_pipline.model_factory import PredictionPipeline



def save_results_to_csv(file_names, fragments, scores, labels, output_csv):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['name', 'fragment', 'score', 'label'])
        
        # Iterate over the data and write to CSV
        for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
            writer.writerow([fname, fragment, score, label])


def predict_audios_csv(input_csv: List[str], audio_dir: str, configuration_file :str, output_path: str):
    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        list_IDs = [row[0] for row in reader]
    
    config = load_config(configuration_file)
    model_name = config['model']['name']

    prediction_pipline = PredictionPipeline(model_name, configuration_file,return_labels = True, return_scores = True)

    for file in list_IDs:
        files, fragments, results = predict(prediction_pipline, file_paths=[file], base_dir= audio_dir)

        save_results_to_csv(files, fragments, results[0], results[1], output_csv=output_path)



if __name__ == '__main__':
    load_dotenv()

    
    # predict_audios_csv('my_pol_data_fake.csv', '../datasets/my_pol/fake', 'config_files/config_mesonet.yaml', 'test_files/my_pol/meso_my_pol_fake.csv')
    # predict_audios_csv('my_pol_data_real.csv', '../datasets/my_pol/real', 'config_files/config_mesonet.yaml', 'test_files/my_pol/meso_my_pol_real.csv')


    # predict_audios_csv('my_pol_data_fake.csv', '../datasets/my_pol/fake', 'config_files/config_wav2vec.yaml', 'test_files/my_pol/wav2vec_my_pol_fake.csv')
    # predict_audios_csv('my_pol_data_real.csv', '../datasets/my_pol/real', 'config_files/config_wav2vec.yaml', 'test_files/my_pol/wav2vec_my_pol_real.csv')


    # predict_audios_csv('my_pol_data_fake.csv', '../datasets/my_pol/fake', 'config_files/config_mesonet_finetuned.yaml', 'test_files/my_pol/meso_ft_my_pol_fake.csv')
    # predict_audios_csv('my_pol_data_real.csv', '../datasets/my_pol/real', 'config_files/config_mesonet_finetuned.yaml', 'test_files/my_pol/meso_ft_my_pol_real.csv')

    predict_audios_csv('example.csv', 'test_dataset', 'config_files/config_mesonet_finetuned.yaml', 'example_results_meso_ft.csv')
    