# Ear Defender - Detector Module

This repository contains the **Detector Module** for the [**Ear Defender**](https://github.com/tymem12/ear-defender) project, designed to analyze audio datasets and return metrics and results. This README provides setup instructions and an overview of the key endpoints for the Detector Module, which operates as a separate Docker-based service within the larger Ear Defender application ecosystem.

---

## Setup Instructions

To set up and launch the Detector Module, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tymem12/ear-defender-model.git
   ```
   
2. **Initialize and Update Submodules**:
   After cloning the repository, update submodules to ensure all necessary components are available.
   ```bash
   git submodule update --init --recursive
   ```

3. **Build and Run the Docker Container**:
   Use Docker Compose to build and start the container:
   ```bash
   docker compose up
   ```
   This will build the container image and launch the Detector Module.

4. **Accessing the Service**:
   Once the container is up and running, the Detector Module will be accessible on **port 7000**.

   - **Documentation**: To view detailed API documentation, navigate to:
     ```
     http://127.0.0.1:7000/docs
     ```

---

## Key Endpoints

The Detector Module provides several endpoints, which are accessible via the API documentation. Here are the main endpoints of interest:

1. **`/model/run`** - **Main Detection Endpoint**:
   - This endpoint is used by the connector module of the Ear Defender application to initiate the core detection process.
   - **Requires Authorization**: A Bearer token is required for access.
   - **Usage**: This is the primary endpoint for the core functionality of the Detector Module, interacting with the connector to perform detection tasks.

2. **Additional Open Endpoints**:
   - Other endpoints are accessible to users without authentication. These endpoints are designed to:
     - Run a specified model for the dataset mentioned in the article (`model/eval_dataset`) and save the results in the `results_csv/{dataset_name}` folder. For this method to work, the `datasets` folder must be created, containing the necessary audio files. All audio files need to be placed inside `datasets`, and the directory structure can be found here: [Google Drive Folder](https://drive.google.com/drive/folders/1ZpGWf4Y9DVYWxHGfkRimII0-m6LvZFPz?usp=sharing).
      - Access metrics based on datasets mentioned in the related research article (`model/eval_metrics`), from the files stored in the `results_csv/{dataset_name}`. To obtain the metrics referenced in the article, it is not necessary to download the datasets.
   - The Postman collections contain examples of requests (including their body) that can be used to run predictions on given dataset with the models or calculate metrics from the predictions.
   - While these endpoints are open to users, they are not utilized by the connector module.

---

## Tests:
- `bash -c "source activate SSL_Spoofing && pytest tests"` - Used to run the tests in the container from the console.
- `bash -c "source activate SSL_Spoofing && pytest --cov=my_app tests/"` - Used to test the coverage of the tests (this command includes coverage for the tests belonging to the used submodule `fairseq`).

---

## References:
- Submodules and implementations used from:
   - [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
   - [deepfake-whisper-features](https://github.com/piotrkawa/deepfake-whisper-features)

- Datasets used for evaluation:
   - [In_the_wild](https://arxiv.org/abs/2203.16263)
   - [MLAAD](https://arxiv.org/abs/2401.09512)
   - [Deep_voice](https://arxiv.org/abs/2308.12734)

