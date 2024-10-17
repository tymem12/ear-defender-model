from abc import ABC, abstractmethod

# Initialization Strategy Interface
class InitializationStrategy(ABC):
    @abstractmethod
    def initialize(self, config_path):
        pass

# Concrete Initialization Strategy for Mesonet Model
class MesonetInitialization(InitializationStrategy):
    def initialize(self, config_path):
        print(f"Initializing Mesonet model with config: {config_path}")

# Concrete Initialization Strategy for Wav2vec Model
class Wav2vecInitialization(InitializationStrategy):
    def initialize(self, config_path):
        print(f"Initializing Wav2vec model with config: {config_path}")
