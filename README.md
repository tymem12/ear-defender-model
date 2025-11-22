# 🧠 EarDefender – Detector Module

**DeepFake Audio Detection Engine**

## 🚀 Overview

The **Detector** Module is responsible for running DeepFake audio analysis within the EarDefender system.
It evaluates audio samples, processes datasets, computes metrics, and exposes REST endpoints for real-time detection.

This service operates as an independent Docker container and integrates with the Connector Module through a secure API.



## ⚙️ Setup Instructions

1. **Clone the repository**

  `git clone https://github.com/tymem12/ear-defender-model.git`

2. **Initialize submodules**

  `git submodule update --init --recursive`

3. **Build & run the service**
   
  `docker compose up`

5. **Access the API**

The detector runs on port 7000.

### API Docs (Swagger/FastAPI)
👉 http://127.0.0.1:7000/docs


## 🔌 Key Endpoints

### `/model/run`

Main Detection Endpoint

- Used directly by the Connector Module

- Requires Bearer Token authorization

- Performs the core DeepFake detection pipeline

- Returns model prediction results for provided audio




### `model/eval_dataset`
Runs a chosen model on an entire dataset

Results are saved to:

`results_csv/{dataset_name}/`

Requires manually preparing the following structure:

```
datasets/
  <dataset_name>/
    audio_files.wav
```

**Dataset structure reference:**

👉 https://drive.google.com/drive/folders/1ZpGWf4Y9DVYWxHGfkRimII0-m6LvZFPz


### `model/eval_metrics`
Computes metrics (e.g., EER) using previously saved predictions

No dataset files needed — only CSV results from `results_csv/`

These metrics correspond to those referenced in the research article

### Postman

**Postman collections** (included in the repo) contain ready-to-use request examples.

**Only** main endpoint is used by the **Connector** during regular EarDefender operation.



## 🧪 Tests

**Run all tests inside the container:**

`bash -c "source activate SSL_Spoofing && pytest tests"`

**Run tests with coverage:**

`bash -c "source activate SSL_Spoofing && pytest --cov=my_app tests/"`

Coverage includes tests for the embedded fairseq submodule.



## 📚 References

### Submodules & Implementations

https://github.com/TakHemlata/SSL_Anti-spoofing

https://github.com/piotrkawa/deepfake-whisper-features


### Datasets Used

**In_the_wild** — https://arxiv.org/abs/2203.16263

**MLAAD** — https://arxiv.org/abs/2401.09512

**Deep_voice** — https://arxiv.org/abs/2308.12734

