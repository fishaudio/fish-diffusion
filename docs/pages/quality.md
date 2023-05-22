# Enhancing Generation Quality

Generation quality is of paramount importance to us and we strive to deliver the highest quality possible. However, generating high-quality results from small or reverberant (not clean) training data can be challenging and may result in significant artifacts. To address this, we've traditionally recommended the use of the HifiSinger architecture.

We're excited to introduce our latest artifact reduction technology: the shallow diffusion denoiser. If you want to use it, ensure that your fish-diffusion version is 2.3.0 or above, and the denoiser model, [Model](https://github.com/fishaudio/fish-diffusion/releases/download/v2.2.0/denoiser-cn-hubert-large-v1.ckpt), is properly placed in the `checkpoints/denoiser/denoiser-cn-hubert-large-v1.ckpt` directory. You can then run the following command to improve generation quality:

```bash
python tools/diffusion/inference.py --config configs/denoiser_cn_hubert.py \
    --checkpoint checkpoints/denoiser/denoiser-cn-hubert-large-v1.ckpt \
    --input "input.wav" \
    --output "output.wav" \
    --sampler_interval 5 \
    --skip_steps 970
```

The `skip_steps` parameter dictates the behavior of the diffusion denoiser. For example, setting it to 970 will perform only 30 denoising steps, which helps in preserving the accent. As the denoiser is essentially a fish diffusion model, you can experiment with various parameters such as `sampler_interval`, `skip_steps`, `pitch_extractor`, etc., to achieve different outcomes.