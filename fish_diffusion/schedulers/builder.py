from mmengine import Registry
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR

LR_SCHEUDLERS = Registry("lr_schedulers")

LR_SCHEUDLERS.register_module(name="LambdaLR", module=LambdaLR)
LR_SCHEUDLERS.register_module(name="StepLR", module=StepLR)
LR_SCHEUDLERS.register_module(name="ExponentialLR", module=ExponentialLR)
