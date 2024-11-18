# Ear Defender - Detector Module

This repository contains the **Detector Module** for the **Ear Defender** project, designed to analyze audio datasets and return metrics and results. This README provides setup instructions and an overview of the key endpoints for the Detector Module, which operates as a separate Docker-based service within the larger Ear Defender application ecosystem.

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
     - Retrieve metrics and performance insights
     - Access results based on datasets mentioned in the related research article
   - While these endpoints are open for users, they are not utilized by the connector module.

---

## tests:
bash -c "source activate SSL_Spoofing && pytest tests"
pytest --cov=your_module_or_directory tests/

