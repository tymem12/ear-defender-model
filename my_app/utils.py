import csv
import os
from typing import Dict, List, Tuple
import yaml

def load_config(yaml_path: str) -> Dict:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        return config

def get_files_to_predict(dataset: str, status: str) -> Tuple[List[str], str]:
    audio_storage_test = os.getenv('AUDIO_DATASETS')
    list_IDs = []  # Initialize list_IDs as empty by default

    if dataset == 'release_in_the_wild':
        path_to_csv = f'{audio_storage_test}/{dataset}/meta.csv'
        if not os.path.isfile(path_to_csv):
            return [], ''
        status_file = 'spoof' if status == 'fake' else 'bona-fide'
        with open(path_to_csv, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            list_IDs = [row[0] for row in reader if row and row[2] == status_file]
        path_to_folder = f'{audio_storage_test}/{dataset}'

    else:
        path_to_csv = f'{audio_storage_test}/{dataset}/{dataset}_data_{status}.csv'
        if not os.path.isfile(path_to_csv):
            return [], ''
        with open(path_to_csv, mode='r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            list_IDs = [row[0] for row in reader if row]
        path_to_folder = f'{audio_storage_test}/{dataset}/{status}'

    return list_IDs, path_to_folder

def save_results_to_csv(file_names: List[str], fragments: List[int], scores: List[float], labels: List[int], output_csv:str) -> None:
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['name', 'fragment', 'score', 'label'])
        
        for fname, fragment, score, label in zip(file_names, fragments, scores, labels):
            writer.writerow([fname, fragment, score, label])

def get_labels_and_predictions_from_csv(spoof_links: List[str], real_links: List[str]) -> Tuple[List[int], List[int]]:
    predictions = []
    labels = []
    for spoof_link in spoof_links:
        if os.path.isfile(spoof_link):
            with open(spoof_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    predictions.append(int(row[3]))
                    labels.append(0)

    for real_link in real_links:
        if os.path.isfile(real_link):
            with open(real_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    predictions.append(int(row[3]))
                    labels.append(1)

    return predictions, labels



def get_scores_and_predictions_from_csv(spoof_links: List[str], real_links: List[str])-> Tuple[List[float], List[int]]:
    predictions = []
    labels = []
    for spoof_link in spoof_links:
        if os.path.isfile(spoof_link):
            with open(spoof_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    predictions.append(int(row[2]))
                    labels.append(0)

    for real_link in real_links:
        if os.path.isfile(real_link):
            with open(real_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    predictions.append(int(row[2]))
                    labels.append(1)

    return predictions, labels



def get_scores_from_csv(spoof_links: List[str], real_links: List[str])-> Tuple[List[float], List[float]]:
    scores_spoof = []
    scores_real = []

    # Process spoof links
    for spoof_link in spoof_links:
        if os.path.isfile(spoof_link):
            # raise FileNotFoundError(f"Spoof link file not found: {spoof_link}")
            with open(spoof_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    scores_spoof.append(float(row[2]))

    # Process real links
    for real_link in real_links:
        if os.path.isfile(real_link):
            # raise FileNotFoundError(f"Real link file not found: {real_link}")
            with open(real_link, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    scores_real.append(float(row[2]))

    return scores_spoof, scores_real

def delate_file_from_storage(link: str, base_dir = None) -> None:
    if base_dir is None:
        base_dir = os.getenv('AUDIO_STORAGE')
    file_path = f'{base_dir}/{link}'
    os.remove(file_path)

def save_durations(files: List[str], durations: List[float], csv_name: str):
    file_exists = os.path.isfile(csv_name)
    with open(csv_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['fname', 'duration'])
        
        for fname, duration in zip(files, durations):
            writer.writerow([fname, duration])