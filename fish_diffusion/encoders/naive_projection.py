from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class NaiveProjectionEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        use_embedding: bool = False,
        use_neck: bool = False,
        neck_size: int = 8,
        preprocessing=None,
    ):
        """Naive projection encoder.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
            use_embedding (bool, optional): Use embedding. Defaults to False.
            use_neck (bool, optional): Use bottleneck. Defaults to False.
            neck_size (int, optional): Hidden size. Defaults to 8. Only used when use_bottleneck is True.
            preprocessing (function, optional): Preprocessing function. Defaults to None.
        """

        super().__init__()

        self.use_embedding = use_embedding
        self.input_size = input_size
        self.output_size = output_size
        self.preprocessing = preprocessing

        if use_embedding:
            self.embedding = nn.Embedding(input_size, output_size)
        elif use_neck:
            self.projection = nn.Sequential(
                nn.Linear(input_size, neck_size),
                nn.Linear(neck_size, output_size),
            )
        else:
            self.projection = nn.Linear(input_size, output_size)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=m.embedding_dim**-0.5)

    def forward(self, x, *args, **kwargs):
        if self.preprocessing is not None:
            x = self.preprocessing(x)

        return self.embedding(x) if self.use_embedding else self.projection(x)
