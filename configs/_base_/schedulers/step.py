optimizer = dict(
    type="AdamW",
    lr=8e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.98),
    eps=1e-9,
)

scheduler = dict(
    type="StepLR",
    step_size=50000,
    gamma=0.5,
)
