from abc import ABC, abstractmethod

class PostprocessingStrategy(ABC):
    @abstractmethod
    def process(self, prediction):
        pass

# Postprocessing strategy for ModelA
class Wav2vecPostprocessing(PostprocessingStrategy):

    def __init__(self, threshold):
        self.threshold : float = threshold
        
    def process(self, prediction):
        return f"Postprocessed {prediction} for ModelA"

# Postprocessing strategy for ModelB
class MesoPostprocessing(PostprocessingStrategy):
    def process(self, prediction):
        return f"Postprocessed {prediction} for ModelB"
