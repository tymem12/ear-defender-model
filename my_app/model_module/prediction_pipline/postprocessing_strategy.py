from abc import ABC, abstractmethod
import torch

class PostprocessingStrategy(ABC):
    def process(self, prediction, return_scores = False, return_labels = True):
        scores = self._process_scores(prediction)
        labels = self._process_label(scores)
        if return_scores and return_labels:
            return scores, labels
        elif return_scores:
            return scores
        elif return_labels:
            return labels

    @abstractmethod
    def _process_label(self, scores):
        pass

    @abstractmethod
    def _process_scores(self, predictions ):
        pass
    
    
# Postprocessing strategy for ModelA
class Wav2vecPostprocessing(PostprocessingStrategy):

    def __init__(self, threshold):
        self.threshold : float = threshold
    
    def _process_scores(self, prediction):
        # print(f"Postprocessed for wav2vec")
    
        scores_list = [t[-1].item() for t in prediction]
        return scores_list
    
    def _process_label(self, scores):
        return [int(val > self.threshold) for val in scores]
        

# Postprocessing strategy for ModelB
class MesoPostprocessing(PostprocessingStrategy):

    
    def _process_scores(self, prediction):
        # print(f"Postprocessed for mesonet")
        # elements = [ scor[1] for scor in prediction]
        # label_list = [((tensor > float(self.threshold)).int()).item() for tensor in elements ]
        scores_list = [t[0].item() for t in prediction]

        return scores_list
    
    def _process_label(self, scores):
        prediction = torch.sigmoid(torch.tensor(scores))
        batch_pred_label = (prediction + 0.5).int()
        return batch_pred_label.tolist()
    