optimizer = dict(
    type="AdamW",
    lr=0.0002,
    betas=(0.8, 0.99),
    eps=1e-9,
)

scheduler = dict(type="ExponentialLR", gamma=0.999)  # lr_decay
