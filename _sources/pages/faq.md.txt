# FAQ

> Please make sure that you are using the latest version of the library.

## Why training is so slow on my device?
- Please make sure that you are not using the CPU. 
- If you are training a laptop, you may want to use FP32 instead of FP16.

## Why the generated audio is blurry or weird?
- If it sounds noisy, you probably need to wait for more training steps.
- If it sounds blurry, please make sure you preprocessed the dataset using your current config.
