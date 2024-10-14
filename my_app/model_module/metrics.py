import csv
import numpy as np
from typing import List
from my_app.model_module.models.wav2vec.eval_metrics_DF import compute_eer

def calculate_eer_from_scores(scores_spoof: List[float], scores_real: List[float]):
    scores_spoof = np.array(scores_spoof)
    scores_real = np.array(scores_real)

    eer, threshold = compute_eer(scores_real, scores_spoof)
    print(f"EER: {eer}   threshold: {threshold}")
    return eer, threshold

def calculate_eer_from_scores_csv(spoof_links: List[str], real_links: List[str]):
    scores_spoof = []
    scores_real = []
    for spoof_link in spoof_links:
        with open(spoof_link, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                scores_spoof.append(float(row[2]))  # Append the score that is in the third row

    for real_link in real_links:
        with open(real_link, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                scores_real.append(float(row[2]))  # Append the score that is in the third row
    
    return calculate_eer_from_scores(scores_spoof, scores_real)


def calculate_eer_from_labels(model_predictions: List[int], actual_labels: List[int]):
    predictions = np.array(model_predictions)
    labels = np.array(actual_labels)
    
    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    false_acceptances = np.sum((predictions == 1) & (labels == 0))  # Predicted 1 but actual is 0 (False Positive)
    false_rejections = np.sum((predictions == 0) & (labels == 1))   # Predicted 0 but actual is 1 (False Negative)
    
    total_negatives = np.sum(labels == 0)  # Actual negatives
    total_positives = np.sum(labels == 1)  # Actual positives
    
    # FAR is the proportion of false acceptances among all negatives
    if total_negatives > 0:
        far = false_acceptances / total_negatives
    else:
        far = 0.0  # Avoid division by zero
    
    # FRR is the proportion of false rejections among all positives
    if total_positives > 0:
        frr = false_rejections / total_positives
    else:
        frr = 0.0  # Avoid division by zero
    
    # EER occurs when FAR == FRR (approximately)
    eer = (far + frr) / 2
    print(f'EER: {eer}')
    
    return eer

def calculate_eer_from_labels_csv(spoof_links: List[str], real_links: List[str]):
    predictions = []
    labels = []
    for spoof_link in spoof_links:
        with open(spoof_link, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                predictions.append(int(row[3]))
                labels.append(0)

    for real_link in real_links:
        with open(real_link, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                predictions.append(int(row[3]))
                labels.append(1)
    
    return calculate_eer_from_labels(predictions, labels)
    




