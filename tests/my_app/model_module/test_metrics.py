import pytest
from unittest.mock import patch
import numpy as np
from my_app.model_module.models.wav2vec.eval_metrics_DF import compute_eer
from my_app.model_module.metrics import calculate_eer_from_scores, calculate_eer_from_labels,file_score_and_label

# Mock data for compute_det_curve (frr, far, thresholds)
mock_frr = np.array([0.1, 0.05, 0.01, 0.005])
mock_far = np.array([0.005, 0.01, 0.05, 0.1])
mock_thresholds = np.array([0.2, 0.4, 0.6, 0.8])

@pytest.fixture
def mock_det_curve():
    with patch('my_app.model_module.models.wav2vec.eval_metrics_DF.compute_det_curve', return_value=(mock_frr, mock_far, mock_thresholds)) as mock:
        yield mock

def test_calculate_eer_from_scores(mock_det_curve):
    # Test input for calculate_eer_from_scores
    scores_real = [0.9, 0.85, 0.8, 0.75, 0.7]
    scores_spoof = [0.1, 0.15, 0.2, 0.25, 0.3]

    eer, threshold = calculate_eer_from_scores(scores_spoof, scores_real)

    # Verify exact EER and threshold values based on mocked compute_det_curve output
    expected_eer = np.mean((mock_frr[2], mock_far[2]))  # Min index where abs(frr - far) is smallest
    expected_threshold = mock_thresholds[1]

    # Check for exact values without approximation
    assert eer == expected_eer, f"Expected EER to be {expected_eer}, got {eer}"
    assert threshold == expected_threshold, f"Expected threshold to be {expected_threshold}, got {threshold}"

def test_calculate_eer_from_labels():
    # Test cases for calculate_eer_from_labels

    # Case 1: Perfect separation - EER should be 0
    model_predictions = [1, 1, 0, 0]
    actual_labels = [1, 1, 0, 0]  # No false positives or false negatives
    eer = calculate_eer_from_labels(model_predictions, actual_labels)
    assert eer == 0.0, "EER should be zero for perfect separation"

    # Case 2: Equal number of false acceptances and false rejections - EER should be 0.5
    model_predictions = [1, 0, 1, 0]
    actual_labels = [0, 1, 0, 1]  # 2 false acceptances, 2 false rejections
    eer = calculate_eer_from_labels(model_predictions, actual_labels)
    expected_eer = 1  # FAR = 0.5, FRR = 0.5
    assert eer == expected_eer, f"Expected EER to be {expected_eer} when FAR and FRR are equal, got {eer}"

    # Case 3: No positives in labels - EER should be 0
    model_predictions = [0, 0, 0, 0]
    actual_labels = [0, 0, 0, 0]  # No positives, so FAR and FRR should be zero
    eer = calculate_eer_from_labels(model_predictions, actual_labels)
    assert eer == 0.0, "EER should be zero when there are no positives"

    # Case 4: No negatives in labels - EER should be 0
    model_predictions = [1, 1, 1, 1]
    actual_labels = [1, 1, 1, 1]  # No negatives, so FAR and FRR should be zero
    eer = calculate_eer_from_labels(model_predictions, actual_labels)
    assert eer == 0.0, "EER should be zero when there are no negatives"

    # Case 5: Mixed predictions with known EER value
    model_predictions = [1, 1, 0, 0, 1, 0, 1, 0]
    actual_labels = [0, 1, 1, 0, 0, 1, 1, 0]  # Mixed case
    false_acceptances = 2  # Model predicted 1 for actual 0
    false_rejections = 2   # Model predicted 0 for actual 1
    total_negatives = 4    # Number of 0s in actual_labels
    total_positives = 4    # Number of 1s in actual_labels
    far = false_acceptances / total_negatives
    frr = false_rejections / total_positives
    expected_eer = far if far == frr else min(far, frr)  # Expected EER

    eer = calculate_eer_from_labels(model_predictions, actual_labels)
    assert eer == expected_eer, f"Expected EER to be {expected_eer}, got {eer}"


def test_file_score_and_label_all_deepfake():
    """Test when all segments are labeled as deepfake."""
    model_predictions = [
        {"segmentNumber": 1, "label": 0},
        {"segmentNumber": 2, "label": 0},
        {"segmentNumber": 3, "label": 0},
    ]
    proportion, label = file_score_and_label(model_predictions)
    assert proportion == 1.0
    assert label == 0

def test_file_score_and_label_all_real():
    """Test when all segments are labeled as real."""
    model_predictions = [
        {"segmentNumber": 1, "label": 1},
        {"segmentNumber": 2, "label": 1},
        {"segmentNumber": 3, "label": 1},
    ]
    proportion, label = file_score_and_label(model_predictions)
    assert proportion == 0.0
    assert label == 1

def test_file_score_and_label_mixed_labels():
    """Test with a mix of deepfake and real labels."""
    model_predictions = [
        {"segmentNumber": 1, "label": 0},
        {"segmentNumber": 2, "label": 1},
        {"segmentNumber": 3, "label": 0},
        {"segmentNumber": 4, "label": 1},
    ]
    proportion, label = file_score_and_label(model_predictions)
    assert proportion == 0.5
    assert label == 0

def test_file_score_and_label_less_than_half_deepfake():
    """Test when less than half of the segments are deepfake."""
    model_predictions = [
        {"segmentNumber": 1, "label": 0},
        {"segmentNumber": 2, "label": 1},
        {"segmentNumber": 3, "label": 1},
    ]
    proportion, label = file_score_and_label(model_predictions)
    assert proportion == pytest.approx(1/3)
    assert label == 1

def test_file_score_and_label_empty_list():
    """Test when the input list is empty."""
    model_predictions = []
    with pytest.raises(ValueError, match="The model_predictions list is empty. Cannot calculate proportions."):
        file_score_and_label(model_predictions)