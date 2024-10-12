# Start with a Python 3.7 base image (since the model needs Python 3.7)
FROM python:3.7

# Set a working directory inside the container
WORKDIR /app

# Install Conda (you can use Miniconda for a lightweight version of Conda)
RUN apt-get update && apt-get install -y wget bzip2 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $HOME/miniconda \
    && rm ~/miniconda.sh \
    && $HOME/miniconda/bin/conda init bash

# Add conda to PATH
ENV PATH=/root/miniconda/bin:$PATH

# Create the Conda environment for SSL_Spoofing
RUN conda create -n SSL_Spoofing python=3.7 -y

# Activate the environment and install dependencies
# Install PyTorch and other dependencies
RUN /bin/bash -c "source activate SSL_Spoofing && \
    pip install torch==1.11.0+cu113 torchvision torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html"

# Copy the model1 directory containing fairseq into the container
COPY . /app


# Ensure that git is installed in the container
RUN apt-get install -y git

# Initialize and update submodules
RUN git submodule init && git submodule update


# Navigate to fairseq directory and install it
WORKDIR /app/models/wav2vec/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
RUN /bin/bash -c "source activate SSL_Spoofing && pip install --editable ./"

# Copy the requirements.txt file into the container and install additional dependencies
COPY requirements.txt /app/requirements.txt
RUN /bin/bash -c "source activate SSL_Spoofing && pip install -r /app/requirements.txt"

WORKDIR /app

COPY .env /app/.env

# Modify the CMD to automatically load environment variables on start
CMD ["/bin/bash", "-c", "source activate SSL_Spoofing && python -m dotenv -f /app/.env run exec bash"]




# YOU NEED TO ADD LOADING DOTEN
# from dotenv import load_dotenv
# import os

# # Load environment variables from the .env file
# load_dotenv(