# FAQ

> Please make sure that you are using the latest version of the library.

## Why training is so slow on my device?
- Please make sure that you are not using the CPU. 
- If you are training a laptop, you may want to use FP32 instead of FP16.

## Why the generated audio is blurry or weird?
- If it sounds noisy, you probably need to wait for more training steps.
- If it sounds blurry, please make sure you preprocessed the dataset using your current config.

## Why I see KeyError 'pytorch-lightning_version'?
This indicates that you are using a converted model from DiffSVC, which is not PyTorch Lightning. Please use `--pretrained` instead of `--resume` to load the model.

## Why I see some error about missing keys when resuming training?
To support the ONNX exporting, we have to change the model structure. To solve this problem, you can use `--pretrained` instead of `--resume` to load the model.
