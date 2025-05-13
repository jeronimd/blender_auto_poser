from typing import Tuple

import torch as t
import torch.nn.functional as f


class FCBlock(t.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int):
        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.fc_layers = [t.nn.Linear(size_in, layer_width)]
        self.relu_layers = [t.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(t.nn.Dropout(p=dropout))
            self.relu_layers.append(t.nn.Identity())
        self.fc_layers += [t.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [t.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

        self.forward_projection = t.nn.Linear(layer_width, size_out)
        self.backward_projection = t.nn.Linear(size_in, layer_width)
        self.fc_layers = t.nn.ModuleList(self.fc_layers)
        self.relu_layers = t.nn.ModuleList(self.relu_layers)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = t.relu(h + self.backward_projection(x))
        return b, f


class Embedding(t.nn.Module):
    """Implementation of embedding using one hot encoded input and fully connected layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.projection = t.nn.Linear(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def forward(self, e: t.Tensor) -> t.Tensor:
        e_ohe = f.one_hot(e, num_classes=self.num_embeddings).float()
        return self.projection(e_ohe)


class WeightedProtoRes(t.nn.Module):
    def __init__(self, nb_joints):
        super().__init__()

        num_layers_enc = 3
        num_blocks_enc = 3
        layer_width_enc = 1024
        num_layers_stage1 = 3
        num_blocks_stage1 = 3
        layer_width_stage1 = 1024
        num_layers_stage2 = 3
        num_blocks_stage2 = 3
        layer_width_stage2 = 1024
        dropout = 0.01
        embedding_dim = 32
        embedding_size = 64
        embedding_num = 2

        size_in = 7
        size_out = nb_joints * 6
        size_out_stage1 = nb_joints * 3

        self.layer_width_enc = layer_width_enc
        self.num_blocks_stage1 = num_blocks_stage1
        self.num_blocks_stage2 = num_blocks_stage2

        self.embeddings = [Embedding(embedding_size, embedding_dim) for _ in range(embedding_num)]

        self.encoder_blocks = [FCBlock(num_layers_enc, layer_width_enc, dropout, size_in + embedding_dim * embedding_num, layer_width_enc)]
        self.encoder_blocks += [FCBlock(num_layers_enc, layer_width_enc, dropout, layer_width_enc, layer_width_enc) for _ in range(num_blocks_enc - 1)]

        self.stage1_blocks = [
            FCBlock(num_layers_stage1, layer_width_stage1, dropout, layer_width_enc, size_out_stage1)
        ] + [FCBlock(num_layers_stage1, layer_width_stage1, dropout, layer_width_stage1, size_out_stage1) for _ in range(num_blocks_stage1 - 1)]

        self.stage2_blocks = [
            FCBlock(num_layers_stage2, layer_width_stage2, dropout, size_out_stage1 + layer_width_enc, size_out)
        ] + [FCBlock(num_layers_stage2, layer_width_stage2, dropout, layer_width_stage2, size_out) for _ in range(num_blocks_stage2 - 1)]

        self.model = t.nn.ModuleList(self.encoder_blocks + self.stage1_blocks + self.stage2_blocks + self.embeddings)

    def encode(self, x: t.Tensor, weights: t.Tensor, *args) -> t.Tensor:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        weights = weights.unsqueeze(2)
        weights_sum = weights.sum(dim=1, keepdim=True)

        ee = [x]
        for i, v in enumerate(args):
            ee.append(self.embeddings[i](v))
        backcast = t.cat(ee, dim=-1)

        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            # weighted average
            prototype = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum

            backcast = backcast - prototype / (i + 1.0)
            backcast = t.relu(backcast)

        pose_embedding = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum
        pose_embedding = pose_embedding.squeeze(1)
        return pose_embedding

    def decode(self, pose_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = pose_embedding
        stage1_forecast = 0.0
        for block in self.stage1_blocks:
            if self.num_blocks_stage1 > 0:
                backcast, f = block(backcast)
                stage1_forecast = stage1_forecast + f
            else:
                stage1_forecast = block(backcast)

        stage1_forecast_no_hips = stage1_forecast - stage1_forecast[:, 0:3].repeat(1, stage1_forecast.shape[1] // 3)
        backcast = t.cat([stage1_forecast_no_hips, pose_embedding], dim=-1)
        stage2_forecast = 0.0
        for block in self.stage2_blocks:
            if self.num_blocks_stage2 > 0:
                backcast, f = block(backcast)
                stage2_forecast = stage2_forecast + f
            else:
                stage2_forecast = block(backcast)

        return stage1_forecast, stage2_forecast

    def forward(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, *args)
        return self.decode(pose_embedding)
