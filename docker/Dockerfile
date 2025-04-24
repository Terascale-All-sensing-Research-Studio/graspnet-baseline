FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set the timezone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install apt dependancies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        nano \
        wget \
        virtualenv \
	ffmpeg \
        libsm6 \
        libxext6

# Enables installing all versions of python
RUN apt-get install -y \
    software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install python dependancies
RUN apt-get install -y \
    python3-pip python3.8 python3.8-dev python3.8-distutils

# Add the GraspNet Baseline directory
ADD .. /graspnet-baseline

# Greate the virtual environment, and install PyTorch and requirements.txt
RUN cd /graspnet-baseline && \
    virtualenv -p python3.8 .venv && \
    . /graspnet-baseline/.venv/bin/activate && \
    pip install torch torchvision torchaudio && \
    pip install -r requirements.txt

# Install pointnet2
RUN cd /graspnet-baseline/pointnet2 && \
    . /graspnet-baseline/.venv/bin/activate && \
    TORCH_CUDA_ARCH_LIST="8.0" pip install .

# Install knn
RUN cd /graspnet-baseline/knn && \
    . /graspnet-baseline/.venv/bin/activate && \
    TORCH_CUDA_ARCH_LIST="8.0" pip install .

# Install graspnetAPI
RUN cd /graspnet-baseline/graspnetAPI && \
    . /graspnet-baseline/.venv/bin/activate && \
    pip install -e .

# Include the virtual environment to the PATH
RUN echo "source /graspnet-baseline/.venv/bin/activate" >> ~/.bashrc

# Set the working directory to the application directory
WORKDIR /graspnet-baseline
