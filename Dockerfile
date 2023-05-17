# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS fish-diffusion
FROM python:3.10-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y git curl build-essential \
    ffmpeg libsm6 libxext6 openssh-server sudo && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install fish user and ssh
RUN useradd -rm -d /home/fish -s /bin/bash -g root -G sudo -u 1000 fish && \
    echo 'fish:fish' | chpasswd && \
    ln -s /home/fish /app && \
    systemctl enable ssh && \
    mkdir /run/sshd
EXPOSE 22

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /app

RUN git clone https://github.com/fishaudio/fish-diffusion.git --depth 1 && cd fish-diffusion && \
    poetry install && rm -rf ~/.cache/pypoetry

WORKDIR /app/fish-diffusion
RUN python3 tools/download_nsf_hifigan.py --agree-license

# Entry point
CMD ["/usr/sbin/sshd", "-D"]
