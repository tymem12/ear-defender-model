import pytest
from unittest.mock import Mock
import torch
from my_app.model_module.prediction_pipeline.postprocessing_strategy import Wav2vecPostprocessing, MesoPostprocessing

# Test Data
@pytest.fixture
def mock_predictions():
    """Fixture that returns mock predictions."""
    # Mock predictions as a batch of tensors, each containing scores
    return [torch.tensor([0.2, 0.3, 0.8]), torch.tensor([0.1, 0.4, 0.9])]

@pytest.fixture
def mock_scores():
    """Fixture that returns processed scores."""
    return [0.8, 0.9]


# Tests for Wav2vecPostprocessing
def test_wav2vec_postprocessing_scores(mock_predictions):
    """Test _process_scores in Wav2vecPostprocessing."""
    threshold = 0.5
    postprocessor = Wav2vecPostprocessing(threshold=threshold)
    scores = postprocessor._process_scores(mock_predictions)
    
    # Verify that scores match the last element in each tensor in mock_predictions
    expected_scores = [0.8, 0.9]
    assert scores == pytest.approx(expected_scores), "Scores did not match expected values in Wav2vecPostprocessing."


def test_wav2vec_postprocessing_labels(mock_scores):
    """Test _process_label in Wav2vecPostprocessing based on the threshold."""
    threshold = 0.5
    postprocessor = Wav2vecPostprocessing(threshold=threshold)
    labels = postprocessor._process_label(mock_scores)
    
    # Expected labels: 1 if score > threshold, otherwise 0
    expected_labels = [1, 1]
    assert labels == expected_labels, "Labels did not match expected values in Wav2vecPostprocessing."


def test_wav2vec_postprocessing_process(mock_predictions):
    """Test process method in Wav2vecPostprocessing."""
    threshold = 0.5
    postprocessor = Wav2vecPostprocessing(threshold=threshold)
    
    # Test different combinations of return_scores and return_labels
    scores, labels = postprocessor.process(mock_predictions, return_scores=True, return_labels=True)
    assert scores == pytest.approx([0.8, 0.9])
    assert labels == [1, 1]
    
    only_scores = postprocessor.process(mock_predictions, return_scores=True, return_labels=False)
    assert only_scores == pytest.approx([0.8, 0.9])
    
    only_labels = postprocessor.process(mock_predictions, return_scores=False, return_labels=True)
    assert only_labels == [1, 1]


# Tests for MesoPostprocessing
def test_meso_postprocessing_scores(mock_predictions):
    """Test _process_scores in MesoPostprocessing."""
    postprocessor = MesoPostprocessing()
    scores = postprocessor._process_scores(mock_predictions)
    
    # Verify that scores match the first element in each tensor in mock_predictions
    expected_scores = [0.2, 0.1]
    assert scores == pytest.approx(expected_scores), "Scores did not match expected values in MesoPostprocessing."


def test_meso_postprocessing_labels(mock_scores):
    """Test _process_label in MesoPostprocessing."""
    postprocessor = MesoPostprocessing()
    labels = postprocessor._process_label(mock_scores)
    
    # Expected labels after sigmoid and rounding
    expected_labels = [1, 1]  # Assuming values are above 0.5 after sigmoid
    assert labels == expected_labels, "Labels did not match expected values in MesoPostprocessing."


def test_meso_postprocessing_process(mock_predictions):
    """Test process method in MesoPostprocessing."""
    postprocessor = MesoPostprocessing()
    
    # Test different combinations of return_scores and return_labels
    scores, labels = postprocessor.process(mock_predictions, return_scores=True, return_labels=True)
    assert scores == pytest.approx([0.2, 0.1])
    assert labels == [1, 1]  # Assuming sigmoid transforms scores above 0.5
    
    only_scores = postprocessor.process(mock_predictions, return_scores=True, return_labels=False)
    assert only_scores == pytest.approx([0.2, 0.1])
    
    only_labels = postprocessor.process(mock_predictions, return_scores=False, return_labels=True)
    assert only_labels == [1, 1]


# Edge Case Tests
def test_wav2vec_postprocessing_no_scores_or_labels(mock_predictions):
    """Test Wav2vecPostprocessing with both return_scores and return_labels set to False."""
    threshold = 0.5
    postprocessor = Wav2vecPostprocessing(threshold=threshold)
    
    result = postprocessor.process(mock_predictions, return_scores=False, return_labels=False)
    assert result is None, "Expected None when both return_scores and return_labels are False."


def test_meso_postprocessing_no_scores_or_labels(mock_predictions):
    """Test MesoPostprocessing with both return_scores and return_labels set to False."""
    postprocessor = MesoPostprocessing()
    
    result = postprocessor.process(mock_predictions, return_scores=False, return_labels=False)
    assert result is None, "Expected None when both return_scores and return_labels are False."
