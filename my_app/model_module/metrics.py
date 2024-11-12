import csv
import numpy as np
from typing import List
from my_app.model_module.models.wav2vec.eval_metrics_DF import compute_eer

def calculate_eer_from_scores(scores_spoof: List[float], scores_real: List[float]):
    scores_spoof = np.array(scores_spoof)
    scores_real = np.array(scores_real)

    eer, threshold = compute_eer(scores_real, scores_spoof)
    return eer, threshold


    
    # return calculate_eer_from_scores(scores_spoof, scores_real)


def calculate_eer_from_labels(model_predictions: List[int], actual_labels: List[int]):
    predictions = np.array(model_predictions)
    labels = np.array(actual_labels)

    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    false_acceptances = np.sum((predictions == 1) & (labels == 0))  # Predicted 1 but actual is 0 (False Positive)
    false_rejections = np.sum((predictions == 0) & (labels == 1))   # Predicted 0 but actual is 1 (False Negative)

    total_negatives = np.sum(labels == 0)  # Actual negatives
    total_positives = np.sum(labels == 1)  # Actual positives

    # Calculate FAR and FRR
    far = false_acceptances / total_negatives if total_negatives > 0 else 0.0
    frr = false_rejections / total_positives if total_positives > 0 else 0.0

    # Calculate EER: if FAR == FRR, then that's our EER. Otherwise, return 1 if both rates are maximal.
    if far == frr:
        eer = far
    else:
        eer = min(far, frr)
    
    return eer


    
    




