FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04

# Install Poetry
RUN apt-get update && apt-get install -y git curl python3 python3-pip build-essential
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /root

RUN pip3 install torch torchvision torchaudio
RUN git clone https://github.com/fishaudio/fish-diffusion.git && cd fish-diffusion && poetry install
