import pytest
import torch
from abc import ABC, abstractmethod
from my_app.model_module.prediction_pipline.postprocessing_strategy import PostprocessingStrategy, Wav2vecPostprocessing, MesoPostprocessing

# Test for the abstract base class (mainly to ensure it cannot be instantiated directly)
def test_postprocessing_strategy_abstract_methods():
    with pytest.raises(TypeError):
        strategy = PostprocessingStrategy()

# Test the Wav2vecPostprocessing class
class TestWav2vecPostprocessingCorrect:
    @pytest.fixture
    def strategy(self):
        return Wav2vecPostprocessing(threshold=0.5)

    @pytest.fixture
    def prediction(self):
        return [torch.tensor([0.1, 0.6]), torch.tensor([0.2, 0.4]), torch.tensor([0.8, 0.9])]

    def test_process_return_scores_and_labels(self, strategy, prediction):
        scores, labels = strategy.process(prediction, return_scores=True, return_labels=True)
        assert scores == pytest.approx([0.6, 0.4, 0.9], rel=1e-5), "Unexpected scores output"
        assert labels == [1, 0, 1], "Unexpected labels output"

    def test_process_return_scores_only(self, strategy, prediction):
        scores = strategy.process(prediction, return_scores=True, return_labels=False)
        assert scores == pytest.approx([0.6, 0.4, 0.9], rel=1e-5), "Unexpected scores output when only scores are returned"

    def test_process_return_labels_only(self, strategy, prediction):
        labels = strategy.process(prediction, return_scores=False, return_labels=True)
        assert labels == [1, 0, 1], "Unexpected labels output when only labels are returned"

    def test_process_no_scores_no_labels(self, strategy, prediction):
        result = strategy.process(prediction, return_scores=False, return_labels=False)
        assert result is None, "Expected None when both return_scores and return_labels are False"

    def test_process_custom_threshold(self, prediction):
        strategy = Wav2vecPostprocessing(threshold=0.7)
        labels = strategy.process(prediction, return_scores=False, return_labels=True)
        assert labels == [0, 0, 1], "Unexpected labels with a custom threshold"

# Test the MesoPostprocessing class
class TestMesoPostprocessingCorrect:
    @pytest.fixture
    def strategy(self):
        return MesoPostprocessing()

    @pytest.fixture
    def prediction(self):
        return [torch.tensor([0.8]), torch.tensor([0.5]), torch.tensor([0.4])]

    def test_process_return_scores_and_labels(self, strategy, prediction):
        scores, labels = strategy.process(prediction, return_scores=True, return_labels=True)
        assert scores == pytest.approx([0.8, 0.5, 0.4], rel=1e-5), "Unexpected scores output"
        assert labels == [1, 1, 1], "Unexpected labels output with sigmoid applied"

    def test_process_return_scores_only(self, strategy, prediction):
        scores = strategy.process(prediction, return_scores=True, return_labels=False)
        assert scores == pytest.approx([0.8, 0.5, 0.4], rel=1e-5), "Unexpected scores output when only scores are returned"

    def test_process_return_labels_only(self, strategy, prediction):
        labels = strategy.process(prediction, return_scores=False, return_labels=True)
        assert labels == [1, 1, 1], "Unexpected labels output when only labels are returned"

    def test_process_no_scores_no_labels(self, strategy, prediction):
        result = strategy.process(prediction, return_scores=False, return_labels=False)
        assert result is None, "Expected None when both return_scores and return_labels are False"