import sys

import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

trainer = dict(
    accelerator="gpu",
    devices=-1,
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    log_every_n_steps=10,
    val_check_interval=5000,
    check_val_every_n_epoch=None,
    max_steps=1_000_000,
    # Warning: If you are training the model with fs2 (and see nan), you should either use bf16 or fp32
    precision="16-mixed",
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.4f}",
            save_on_train_epoch_end=False,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)

# Use DDP for multi-gpu training
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # Use gloo for windows
    process_group_backend = "nccl" if sys.platform != "win32" else "gloo"

    trainer["strategy"] = DDPStrategy(
        process_group_backend=process_group_backend,
        gradient_as_bucket_view=True,
        ddp_comm_hook=default.fp16_compress_hook,
    )
