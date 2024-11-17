# Start with a Python 3.7 base image (since the model needs Python 3.7)
FROM python:3.7

# Set a working directory inside the container
WORKDIR /app

# Install necessary system dependencies and Conda
RUN apt-get update && apt-get install -y wget bzip2 ffmpeg curl \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $HOME/miniconda \
    && rm ~/miniconda.sh \
    && $HOME/miniconda/bin/conda init bash

# Add Conda to PATH
ENV PATH=/root/miniconda/bin:$PATH

# Create the Conda environment for SSL_Spoofing
RUN conda create -n SSL_Spoofing python=3.7 -y

# Activate the environment and install dependencies
# Install PyTorch and other dependencies
RUN /bin/bash -c "source activate SSL_Spoofing && \
   pip install torch==1.11.0+cu113 torchvision torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html"

# Install gdown to handle Google Drive downloads
RUN /bin/bash -c "source activate SSL_Spoofing && pip install gdown"

# Copy the model1 directory containing fairseq into the container
COPY . /app

# Navigate to fairseq directory and install it
WORKDIR /app/my_app/model_module/models/wav2vec/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
RUN /bin/bash -c "source activate SSL_Spoofing && pip install --editable ./"

# Copy the requirements.txt file into the container and install additional dependencies
COPY requirements.txt /app/requirements.txt
RUN /bin/bash -c "source activate SSL_Spoofing && pip install -r /app/requirements.txt"

# Use gdown to download pretrained models from Google Drive
RUN mkdir -p /app/pretrained && \
    if [ ! -f /app/pretrained/Best_LA_model_for_DF.pth ]; then \
        /bin/bash -c "source activate SSL_Spoofing && gdown --id 1DMVH5l34MmOMc_Y2--qaE8YETBCYW7SL -O /app/pretrained/Best_LA_model_for_DF.pth"; \
    fi

RUN if [ ! -f /app/xlsr2_300m.pt ]; then \
        /bin/bash -c "source activate SSL_Spoofing && gdown --id 1hNPENe7LctmpLoC2R1nBa7cL1gCP4f6r -O /app/xlsr2_300m.pt"; \
    fi

# Expose the application port
EXPOSE 7000

# Set the working directory back to /app
WORKDIR /app

# Copy the environment file
COPY .env /app/.env

# Define the default command to run the application
CMD ["/bin/bash", "-c", "source activate SSL_Spoofing && python -m dotenv -f /app/.env run uvicorn my_app.endpoints_api:app --host 0.0.0.0 --port 7000 --reload"]
