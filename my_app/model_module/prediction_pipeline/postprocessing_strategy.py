from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import torch
from torch import Tensor

class PostprocessingStrategy(ABC):
    def process(
        self, 
        prediction: List[Tensor],  # Assuming predictions are lists of tensors
        return_scores: bool = False, 
        return_labels: bool = True
    ) -> Union[List[int], List[float], Tuple[List[float], List[int]]]:
        scores = self._process_scores(prediction)
        labels = self._process_label(scores)
        if return_scores and return_labels:
            return scores, labels
        elif return_scores:
            return scores
        elif return_labels:
            return labels

    @abstractmethod
    def _process_label(self, scores: List[float]) -> List[int]:
        pass

    @abstractmethod
    def _process_scores(self, prediction: List[Tensor]) -> List[float]:
        pass
    
# Postprocessing strategy for ModelA
class Wav2vecPostprocessing(PostprocessingStrategy):

    def __init__(self, threshold:float):
        self.threshold : float = threshold
    
    def _process_scores(self, prediction: List[Tensor]) -> List[float]:
        scores_list = [t[-1].item() for t in prediction]
        return scores_list
    
    def  _process_label(self, scores: List[float]) -> List[int]:
        return [int(val > self.threshold) for val in scores]
        

# Postprocessing strategy for ModelB
class MesoPostprocessing(PostprocessingStrategy):
    
    def _process_scores(self, prediction: List[torch.Tensor]) -> List[float]:
        scores_list = [t[0].item() for t in prediction]
        return scores_list
    
    def  _process_label(self, scores: List[float]) -> List[int]:
        prediction = torch.sigmoid(torch.tensor(scores))
        batch_pred_label = (prediction + 0.5).int()
        return batch_pred_label.tolist()
    