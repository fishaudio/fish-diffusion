FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS fish-diffusion

# Install system dependencies
RUN apt-get update && apt-get install -y git curl python3 python3-pip build-essential ffmpeg libsm6 libxext6 openssh-server sudo
RUN useradd -rm -d /home/fish -s /bin/bash -g root -G sudo -u 1000 fish 
RUN echo 'fish:fish' | chpasswd
RUN ln -s /home/fish /app
RUN systemctl enable ssh
RUN mkdir /run/sshd
EXPOSE 22

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /app

RUN git clone https://github.com/fishaudio/fish-diffusion.git && cd fish-diffusion && poetry install

WORKDIR /app/fish-diffusion
RUN python3 tools/download_nsf_hifigan.py --agree-license

# Entry point
CMD ["/usr/sbin/sshd", "-D"]
