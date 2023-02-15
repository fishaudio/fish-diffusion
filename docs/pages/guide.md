# Quick FishSVC Guide

> Updated: 01/02/2023
> Made by: Kangarroar

First you need to install conda on your PC, I recommend installing Miniconda if you don't want it to eat a lot of your disk space.

The link for Miniconda is here: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Make sure you choose Miniconda Python 3.10 otherwise you won't be able to install Fish!
After installing Miniconda, open "Anaconda", in my case I have Windows so I press the key Windows and type Anaconda, smh.

Then you will need to type

    conda create --name Fish

  Once you have done that, a environment called Fish will be created, to access it you need to type

    conda activate Fish

  

Then you will be already on your Fish environment and you can proceed installation of FishSVC!







***
## Preparing the environment

  First you need to install PyTorch to be able to train, inference and etc with this command:

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

 After that you need to run

     pip install poetry

 Once you have finished the base to set up the environment, proceed to download FishSVC from the [GitHub](https://github.com/fishaudio/fish-diffusion)

Click code and then "Download as zip", then you decompress the folder wherever you want.

On your conda environment, point to the folder where you have all the files for fish, just click on the file explorer bar and copy the full path, on conda run the command

    cd C:/Users//NAME/Documents/fish-difussion (example)

  

And then run the command

    poetry install

  
 Fish Diffusion requires the OPENVPI 441khz NSF-HiFiGAN vocoder to generate audio, there is an automatic download for it, just run the command

     python tools/download_nsf_hifigan.py

It will start downloading the vocoder automatically and will put it on the checkpoints folder,, wait until it's done or you can do a manual download for it. [Hifigan Link](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1)

  

FishSVC is installed!



***

## Dataset Preparation

You need to put the dataset into the dataset directory in the following file structure

```shell
dataset
├───train
│   ├───xxx1-xxx1.wav
│   ├───...
│   ├───Lxx-0xx8.wav
│   └───speaker0 (Subdirectory is also supported)
│       └───xxx1-xxx1.wav
└───valid
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
All the wav files need to be inside the train folder, not in subfolder or otherwise it will fail when preprocessing, unless you are doing a multi-speaker model.

_I STRONGLY RECOMMEND FOR LOCAL TRAINING, THAT THE WAVS AREN'T HIGHER THAN 10 SECONDS LONG IF YOU HAVE 4GB OF VRAM_

To start the preprocessing you need to run the following command

    python tools/preprocessing/extract_features.py --config configs/svc_hubert_soft.py --path dataset --clean

 It should take about 2 to 5 minutes to finish preproccesing, once that is done run the following command

    python tools/preprocessing/generate_stats.py --input-dir dataset/train --output-file dataset/stats.json

  Once that is done you have your data prepared to start training!


***

## Training time!

You need to change your batchsize accordingly to your GPU resources, which can be found in `configs/_base_/datasets/audio_folder.py`

![](https://lh4.googleusercontent.com/0q_FDH7QqyHOW1dicuEgVrU2MFPkTP8xU1jL7-NcdwgmmMzYHGo8P-E76p_Ecvh55v6J2jZXN271F0bDW7NnmfvhlIO17MIhjFzGIxJtabOtN6q9kzUPB1Caeamz3_XNGAmNY2K0znkI4A63VH64zMo)

The value you can change is batch size, and num_workers

In my case I have a CPU with 6 cores and a GTX 1650 with 4GB VRAM, I set my number of workers to my number of cores, and my batch size is set to 12, it may vary for you if you have a better GPU, but a GTX 1650 is the baseline to be able to train.

If you get CUDA out of memory try lowering the batch size.
You may want to change every how many steps you want to save the ckpt! By default it saves the checkpoints every 10.000 steps, the config file for it is in:

    configs/_base_/trainers/base.py

![](https://lh3.googleusercontent.com/cxFONBMCwnwmMASjRCkacF6OzQBFH28Ctzrx7FfMrSgE8WjjTBd11g7kTnisUZYJk692wdO_yXK2w9IktDX86RSU4QXm6GiMpq5_xJjeooRsVmg29GvAEni0lT2FW5kML9cL2-uuzJ9ptwiUkGBxN9s)

The line you want to change is the line 20, "every_n_train_steps=10000", change the number to just "1000".
To start training, you need to run the following command:

    python train.py --config configs/svc_hubert_soft.py

It will ask for a wandb account to look at your graphs and audios, it works similar to tensorboard, just follow the steps the terminal will show to you if you want to use W&B or not

Training should start after setting that up!

***


## Inference

Inferencing with FishSVC is fully terminal code right now, in my case I have a folder called audios where I drop the wav to inference, the command to run inference is:
```bash
python inference.py --config configs/svc_hubert_soft.py \
    --checkpoint [checkpoint] \
    --input [input audio] \
    --output [output audio]
```
If you want to change the key, after everything add: –pitch_adjust 0 (Change 0 to your liking, like DiffSVC, 6 means 6 notes higher, -6 means 6 notes lower.

 Your checkpoints will be in the folder logs. In output audio you need to put the name you want your rendered file to be, including .wav at the end

In my case, the command I run is:

    python inference.py –config configs/svc_hubert_soft.py – checkpoint logs/8liv3s/epoches….=0.2.ckpt –input audios/test.wav output – audios/render/test.wav

It should start rendering and it's pretty fast actually! So in less than 2 minutes your audio should be rendered.

Done! ^^
(I'm actually developing a GUI to run inference without having to put commands, here's a sneak peak of it)

![](https://lh4.googleusercontent.com/McIUfOcaaF6kffVUurQqmkJT_0lXvDSZEDbyFjLTPdnWXW-NKViOSpq7E8c5KihumzayZq7JakCsi7m3E7uVBBkOlu3AoLuVpipVUYbhi8GwxPvkoVrixSLbwDdc38HRiOVvA91jDJLvtyAVRVuUoZ4)

  


***


## DIFF SVC CONVERSION TO FISH SVC

This is pretty straightforward, conversion takes 2 minutes to run and it allows inference inmediatly!
You first need to copy your DiffSVC checkpoint of the model you want to convert, in my case I created a specific folder called "conversion"

 
To run the conversion you need to use the following command:
```bash
python tools/diff_svc_converter.py --config configs/svc_hubert_soft_diff_svc.py \
    --input-path [DiffSVC ckpt] \
    --output-path [Fish Diffusion ckpt]
```

In my case, the command I will run it's:

    python tools/diff_svc_converter.py –config configs/svc_hubert_soft_diff_svc.py –input-path conversion/model_train_400000.ckpt –output-path checkpoints/NAME.ckpt
It will start running the conversion, if you run into a PyTorch error of size mismatch, you may need to change residual channels on your svc_hubert_soft_diff_svc.py

  
***


## Inferencing with a DiffSVC converted model

It's the same as a normal FishSVC model, just that instead you will need to use: svc_hybert_soft_diff_svc.py config, instead of the normal one.
```bash
python inference.py --config configs/svc_hubert_soft_diff_svc.py \
    --checkpoint [checkpoint] \
    --input [input audio] \
    --output [output audio]
```

In my case the command I will run it's:

    python inference.py –config configs/svc_hubert_soft_diff_svc.py – checkpoint checkpoint/NAME.ckpt –input audios/test.wav output – audios/render/test.wav
And done! You are inferencing with FishSVC from a DiffSVC ckpt!
