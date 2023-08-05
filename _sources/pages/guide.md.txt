# Quick FishSVC Guide

> Updated: March 03 2023 (Lengyue)  
> Made & Updated: Kangarroar (Feb 01 2023)
> Updated: May 17 2023 (OOPPEENN)

If you don't want to install the environment manually or don't have a powerful GPU, using [Google Colab](https://colab.research.google.com/github/fishaudio/fish-diffusion/blob/main/notebooks/train.ipynb) for training is a good option to get started

## Preparing the environment
1. You need to install conda on your PC, I recommend installing Miniconda if you don't want it to eat a lot of your disk space.

   The link for Miniconda is here: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   
   For Chinese users, you may need to get miniconda through the mirror source, and change the miniconda source according to the guide of the mirror station.
   
   The link for Miniconda is here: [https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)  
   The link for guide is here: [https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

2. After installing Miniconda, open "Anaconda", then you will need to type
   ```
   conda create --name Fish python=3.10
   ```
   Once you have done that, a environment called Fish will be created, to access it you need to type
   ```
   conda activate Fish
   ```
3. Make sure that the next steps are in the virtual environment of Fish

   Install pdm to manage project dependencies
   
   Windows:
   ```
   curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -
   ```
   Linux:
   ```
   curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -
   ```
   For Chinese users, you may need to get pdm through the mirror source, and change the pdm source according to the guide of the mirror station.  
   The link for guide is here: [https://mirrors.tuna.tsinghua.edu.cn/help/pypi/](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

 4. Once you have finished the base to set up the environment, proceed to download FishSVC from the [GitHub](https://github.com/fishaudio/fish-diffusion). You can either

- Click code and then "Download as zip", then you decompress the folder wherever you want.
- Or you can clone the repository with the command `git clone https://github.com/fishaudio/fish-diffusion` if you have git installed.

   In your conda environment, point to the folder where you have all the files for fish, just click on the file explorer bar and copy the full path, on conda run the command
   ```
   cd C:/Users/NAME/Documents/fish-difussion (example)
   ```

  
5. Run this command to update the dependencies and install the project
   ```
   pdm sync
   ```
  
6. Fish Diffusion requires the [FishAudio NSF-HiFiGAN](https://github.com/fishaudio/fish-diffusion/releases/tag/v2.0.0) vocoder to generate audio, there is an automatic download for it, just run the command
   ```
   python tools/download_nsf_hifigan.py
   ```

   It will start downloading the vocoder automatically and will put it on the checkpoints folder, wait until it's done or you can do a manual download for it. [Hifigan Link](https://github.com/openvpi/vocoders/releases/tag/nsf-hifigan-v1)

  

7.  FishSVC is installed!

## Dataset Preparation (Single Speaker)

You need to put the dataset into the dataset directory in the following file structure

```shell
dataset
├───train
│   ├───xxx1-xxx1.wav
│   ├───...
│   ├───Lxx-0xx8.wav
└───valid
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

> You should only put few wav files on the valid folder (like 10-20), it's only used to check the audio quality.

_I STRONGLY RECOMMEND FOR LOCAL TRAINING, THAT THE WAVS AREN'T HIGHER THAN 10 SECONDS LONG IF YOU HAVE 4GB OF VRAM_

To start the preprocessing you need to run the following command

```bash
# Process the training data (with augmentation if included in the config)
python tools/preprocessing/extract_features.py --config configs/svc_content_vec.py --path dataset/train --clean

# Process the validation data without augmentation
python tools/preprocessing/extract_features.py --config configs/svc_content_vec.py --path dataset/valid --clean --no-augmentation
```

  Once that is done you have your data prepared to start training!

> It's recommended to copy the config file to a new one, so you don't mess up the original one.

## Dataset Preparation (Multi Speaker)
Reference to [Dataset Configuration](https://fishaudio.github.io/fish-diffusion/pages/config.html#dataset). 

## Training time!

You may need to change your batchsize accordingly to your GPU resources, which can be found in `configs/_base_/datasets/naive_svc.py`.

```python
dataloader = dict(
    train=dict(
        batch_size=20,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    ),
    valid=dict(
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    ),
)
```

The value you can change is batch size, and num_workers

In my case I have a CPU with 6 cores and a GTX 1650 with 4GB VRAM, I set my number of workers to my number of cores, and my batch size is set to 12, it may vary for you if you have a better GPU, but a GTX 1650 is the baseline to be able to train.

If you get CUDA out of memory try lowering the batch size.
You may want to change every how many steps you want to save the ckpt! By default it saves the checkpoints every 10,000 steps, the config file for it is in:

    configs/_base_/trainers/base.py

```python
    log_every_n_steps=10,
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    max_steps=300000,
    # Warning: If you are training the model with fs2 (and see nan), you should either use bf16 or fp32
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=10000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
```

The line you want to change is the line 20, "every_n_train_steps=10000", change the number to just "1000" if you want to save every 1,000 steps (it will also takes more space on your hard drive).

To start training, you need to run the following command:

    python tools/diffusion/train.py --config configs/svc_content_vec.py

It will ask for a wandb account to look at your graphs and audios, it works similar to tensorboard, just follow the steps the terminal will show to you if you want to use W&B or not

Training should start after setting that up!

***


## Inference

Inferencing with FishSVC is fully terminal code right now, in my case I have a folder called audios where I drop the wav to inference, the command to run inference is:
```bash
python tools/diffusion/inference.py --config configs/svc_content_vec.py \
    --checkpoint [checkpoint file] \
    --input [input audio] \
    --output [output audio]
```
If you want to change the key, after everything add: –pitch_adjust 0 (Change 0 to your liking, like DiffSVC, 6 means 6 notes higher, -6 means 6 notes lower.

 Your checkpoints will be in the folder logs. In output audio you need to put the name you want your rendered file to be, including .wav at the end

In my case, the command I run is:

```bash
python tools/diffusion/inference.py –-config configs/svc_hubert_soft.py \
    –-checkpoint logs/8liv3s/epoches….=0.2.ckpt \
    –-input audios/test.wav \
    --output audios/render/test.wav
```

It should start rendering and it's pretty fast actually! So in less than 2 minutes your audio should be rendered.

Done! ^^

If you want to start a web-ui to render your audios, you can use the following command:

```bash
python tools/diffusion/inference.py –-config configs/svc_hubert_soft.py \
    –-checkpoint logs/8liv3s/epoches….=0.2.ckpt \
    --gradio
```


***


## DIFF SVC CONVERSION TO FISH SVC

This is pretty straightforward, conversion takes 2 minutes to run and it allows inference immediately!
You first need to copy your DiffSVC checkpoint of the model you want to convert, in my case I created a specific folder called "conversion"

 
To run the conversion you need to use the following command:
```bash
python tools/diffusion/diff_svc_converter.py --config configs/svc_hubert_soft_diff_svc.py \
    --input-path [DiffSVC ckpt] \
    --output-path [Fish Diffusion ckpt]
```

In my case, the command I will run it's:

```bash
python tools/diffusion/diff_svc_converter.py –-config configs/svc_hubert_soft_diff_svc.py –-input-path conversion/model_train_400000.ckpt -–output-path checkpoints/NAME.ckpt
```

It will start running the conversion, if you run into a PyTorch error of size mismatch, you may need to change residual channels on your svc_hubert_soft_diff_svc.py

  
***


## Inferencing with a DiffSVC converted model

It's the same as a normal FishSVC model, just that instead you will need to use: svc_hybert_soft_diff_svc.py config, instead of the normal one.
```bash
python tools/diffusion/inference.py --config configs/svc_hubert_soft_diff_svc.py \
    --checkpoint [checkpoint path] \
    --input [input audio] \
    --output [output audio]
```

In my case the command I will run it's:

```bash
python tools/diffusion/inference.py –-config configs/svc_hubert_soft_diff_svc.py \
    –-checkpoint checkpoint/NAME.ckpt \
    –-input audios/test.wav \
    –-output audios/render/test.wav
```

And done! You are inferencing with FishSVC from a DiffSVC ckpt!
