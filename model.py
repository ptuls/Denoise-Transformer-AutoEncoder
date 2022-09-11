import torch
import numpy as np
from typing import Union


bce_logits = torch.nn.functional.binary_cross_entropy_with_logits
mse = torch.nn.functional.mse_loss


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, feedforward_dim: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm1(x_in + attn_out)
        ff_out = self.linear2(torch.nn.functional.relu(self.linear1(x)))
        x = self.layernorm2(x + ff_out)
        return x


class TransformerAutoEncoder(torch.nn.Module):
    """
    Takes in numerical and categorical features to learn a latent space of the
    tabular dataset.
    """

    def __init__(
        self,
        num_inputs: int,
        n_cats: int,
        n_nums: int,
        num_encoders: int = 3,
        hidden_size: int = 1024,
        num_subspaces: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        feedforward_dim: int = 512,
        emphasis: float = 0.75,
        task_weights: list[float] = [10, 14],
        mask_loss_weight: int = 2,
    ):
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.emphasis = emphasis
        self.task_weights = np.array(task_weights) / sum(task_weights)
        self.mask_loss_weight = mask_loss_weight

        self.excite = torch.nn.Linear(in_features=num_inputs, out_features=hidden_size)
        if num_encoders > 0:
            self.num_encoders = num_encoders
        else:
            raise ValueError("number of encoders has to be at least 1")

        self.encoders = []
        for i in range(self.num_encoders):
            self.encoders.append(TransformerEncoder(embed_dim, num_heads, dropout, feedforward_dim))

        self.mask_predictor = torch.nn.Linear(in_features=hidden_size, out_features=num_inputs)
        self.reconstructor = torch.nn.Linear(
            in_features=hidden_size + num_inputs, out_features=num_inputs
        )

    def divide(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.num_subspaces, self.embed_dim)).permute((1, 0, 2))
        return x

    def combine(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[1]
        x = x.permute((1, 0, 2)).reshape((batch_size, -1))
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        x = torch.nn.functional.relu(self.excite(x))
        enc = []
        x = self.divide(x)
        enc[0] = self.encoders[0](x)
        for i in range(1, self.num_encoders):
            enc[i] = self.encoders[i](enc[i - 1])
        x = self.combine(enc[self.num_encoders - 1])

        predicted_mask = self.mask_predictor(x)
        reconstruction = self.reconstructor(torch.cat([x, predicted_mask], dim=1))
        return enc, (reconstruction, predicted_mask)

    def split(self, t: torch.Tensor) -> torch.Tensor:
        return torch.split(t, [self.n_cats, self.n_nums], dim=1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        attn_outs, _ = self.forward(x)
        return torch.cat([self.combine(x) for x in attn_outs], dim=1)

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, reduction: str = "mean"
    ) -> Union[float, list[float, float]]:
        _, (reconstruction, predicted_mask) = self.forward(x)
        x_cats, x_nums = self.split(reconstruction)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(mask * self.emphasis + (1 - mask) * (1 - self.emphasis))

        cat_loss = self.task_weights[0] * torch.mul(
            w_cats, bce_logits(x_cats, y_cats, reduction="none")
        )
        num_loss = self.task_weights[1] * torch.mul(w_nums, mse(x_nums, y_nums, reduction="none"))

        reconstruction_loss = (
            torch.cat([cat_loss, num_loss], dim=1)
            if reduction == "none"
            else cat_loss.mean() + num_loss.mean()
        )
        mask_loss = self.mask_loss_weight * bce_logits(predicted_mask, mask, reduction=reduction)

        return (
            reconstruction_loss + mask_loss
            if reduction == "mean"
            else [reconstruction_loss, mask_loss]
        )


class SwapNoiseMasker(object):
    def __init__(self, probas: np.array):
        self.probas = torch.from_numpy(np.array(probas))

    def apply(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        should_swap = torch.bernoulli(self.probas.to(x.device) * torch.ones(x.shape).to(x.device))
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x, mask


def test_tf_encoder():
    m = TransformerEncoder(4, 2, 0.1, 16)
    x = torch.rand((32, 8))
    x = x.reshape((32, 2, 4)).permute((1, 0, 2))
    o = m(x)
    assert o.shape == torch.Size([2, 32, 4])


def test_dae_model():
    m = TransformerAutoEncoder(
        5,
        2,
        3,
        num_encoders=3,
        hidden_size=16,
        num_subspaces=4,
        embed_dim=4,
        num_heads=2,
        dropout=0.1,
        feedforward_dim=4,
        emphasis=0.75,
    )

    x = torch.cat([torch.randint(0, 2, (5, 2)), torch.rand((5, 3))], dim=1)
    f = m.feature(x)
    assert f.shape == torch.Size([5, 16 * 3])
    loss = m.loss(x, x, (x > 0.2).float())


def test_swap_noise():
    probas = [0.2, 0.5, 0.8]
    m = SwapNoiseMasker(probas)
    diffs = []
    for i in range(1000):
        x = torch.rand((32, 3))
        noisy_x, _ = m.apply(x)
        diffs.append((x != noisy_x).float().mean(0).unsqueeze(0))

    print("specified : ", probas, " - actual : ", torch.cat(diffs, 0).mean(0))


if __name__ == "__main__":
    test_tf_encoder()
    test_dae_model()
    test_swap_noise()
