from torch import nn
from .builder import ENCODERS


@ENCODERS.register_module()
class NaiveProjectionEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        use_embedding: bool = False,
        use_softmax_bottleneck: bool = False,
        hidden_size=128,
        preprocessing=None,
    ):
        """Naive projection encoder.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
            use_embedding (bool, optional): Use embedding. Defaults to False.
            use_softmax_bottleneck (bool, optional): Use softmax bottleneck. Defaults to False.
            hidden_size (int, optional): Hidden size. Defaults to 128. Only used when use_softmax_bottleneck is True.
            preprocessing (function, optional): Preprocessing function. Defaults to None.
        """

        super().__init__()

        self.use_embedding = use_embedding
        self.input_size = input_size
        self.output_size = output_size
        self.preprocessing = preprocessing

        if use_embedding:
            self.embedding = nn.Embedding(input_size, output_size)
        elif use_softmax_bottleneck:
            self.projection = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Softmax(dim=2),
                nn.Linear(hidden_size, output_size),
            )
        else:
            self.projection = nn.Linear(input_size, output_size)

    def forward(self, x, *args, **kwargs):
        if self.preprocessing is not None:
            x = self.preprocessing(x)

        return self.embedding(x) if self.use_embedding else self.projection(x)
