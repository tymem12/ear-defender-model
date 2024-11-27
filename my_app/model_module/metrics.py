from typing import List, Tuple, Dict
import numpy as np
from my_app.model_module.models.wav2vec.eval_metrics_DF import compute_eer



def calculate_eer_from_scores(scores_spoof: List[float], scores_real: List[float]) -> Tuple[float, float]:
    if not scores_spoof:
        return None, "Warning: Cannot calculate EER. No fi"
    if not scores_real:
        return None, "Warning: Cannot calculate EER. Real examples do not exist"


    # Convert to numpy arrays
    scores_spoof = np.array(scores_spoof)
    scores_real = np.array(scores_real)

    # Calculate EER and threshold using the compute_eer function
    eer, threshold = compute_eer(scores_real, scores_spoof)
    return eer, threshold
    

def calculate_eer_from_labels(model_predictions: List[int], actual_labels: List[int]) -> float:
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

def file_score_and_label(model_predictions: List[Dict[str, int]]) -> Tuple[float, int]:
  
    # Count the total number of segments
    total_segments = len(model_predictions)
    
    if total_segments == 0:
        raise ValueError("The model_predictions list is empty. Cannot calculate proportions.")

    # Count the number of deepfake segments (label == 0)
    deepfake_segments = sum(1 for prediction in model_predictions if prediction["label"] == 0)
    
    # Calculate the proportion of deepfake segments
    proportion_deepfake = deepfake_segments / total_segments
    
    # Assign the file label based on the proportion
    file_label = 1 if proportion_deepfake < 0.5 else 0
    
    return proportion_deepfake, file_label


from typing import List

def calculate_acc_from_labels(model_predictions: List[int], actual_labels: List[int]) -> float:
    # Check if both lists have the same length
    if len(model_predictions) != len(actual_labels):
        return None
    if not actual_labels:
        return None


    # Calculate the number of correct predictions
    correct_predictions = sum(p == a for p, a in zip(model_predictions, actual_labels))

    # Calculate accuracy
    accuracy = correct_predictions / len(actual_labels)

    return accuracy

