FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS fish-diffusion

# Install Poetry
RUN apt-get update && apt-get install -y git curl python3 python3-pip build-essential ffmpeg libsm6 libxext6
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /root

RUN git clone https://github.com/fishaudio/fish-diffusion.git && cd fish-diffusion && poetry install

WORKDIR /root/fish-diffusion
RUN python3 tools/download_nsf_hifigan.py --agree-license
