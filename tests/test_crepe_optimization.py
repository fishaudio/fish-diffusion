import time

import pytorch_lightning as pl
import torch
import torchaudio
from torchcrepe.filter import mean, median

from fish_diffusion.modules.pitch_extractors.crepe import (
    CrepePitchExtractor,
    MaskedAvgPool1d,
    MaskedMedianPool1d,
)


def test_pools():
    masked_avg_pool = MaskedAvgPool1d(3, 1, 1)
    masked_median_pool = MaskedMedianPool1d(3, 1, 1)

    # Test Pool Optimization
    x = torch.rand(1, 44100 * 10)

    # Random 10% of the elements in the input tensor will be nan
    x[torch.rand_like(x) < 0.1] = float("nan")

    t0 = time.time()
    r0_mean = mean(x, 3)
    r0_median = median(x, 3)
    t0 = time.time() - t0

    t1 = time.time()
    r1_mean = masked_avg_pool(x)
    r1_median = masked_median_pool(x)
    t1 = time.time() - t1

    # Show where the values are different
    r0_mean[torch.isnan(r0_mean)] = 0
    r1_mean[torch.isnan(r1_mean)] = 0

    assert torch.allclose(r0_mean, r1_mean)

    r0_median[torch.isnan(r0_median)] = 0
    r1_median[torch.isnan(r1_median)] = 0

    assert torch.allclose(r0_median, r1_median)

    print(f"---- Pools Test ----")
    print(f"Original: {t0}, Optimized: {t1}, Speedup: {t0 / t1}")


def test_real_world():
    source = "dataset/valid/opencpop/TruE-干音_0000/0002.wav"
    audio, sr = torchaudio.load(source)
    audio = audio.cuda().mean(dim=0, keepdim=True)

    original = CrepePitchExtractor(
        f0_min=40.0, f0_max=2000.0, keep_zeros=False, use_fast_filters=False
    )
    optimized = CrepePitchExtractor(
        f0_min=40.0, f0_max=2000.0, keep_zeros=False, use_fast_filters=True
    )

    pl.seed_everything(0)
    original_time = time.time()
    x0 = original(audio, sr)
    original_time = time.time() - original_time

    pl.seed_everything(0)
    optimized_time = time.time()
    x1 = optimized(audio, sr)
    optimized_time = time.time() - optimized_time

    assert torch.allclose(x0, x1)

    print(f"---- Real World Test ----")
    print(
        f"Original: {original_time}, Optimized: {optimized_time}, Speedup: {original_time / optimized_time}"
    )


if __name__ == "__main__":
    test_pools()
    test_real_world()

    # Run again to avoid the first run being slower
    test_real_world()
