import random
import sys

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

""" from https://github.com/Pointerformer/Pointerformer/blob/main/revtorch/revtorch.py """


class ReversibleBlock(nn.Module):
    """
    Elementary building block for building (partially) reversible architectures

    Implementation of the Reversible block described in the RevNet paper
    (https://arxiv.org/abs/1707.04585). Must be used inside a :class:`revtorch.ReversibleSequence`
    for autograd support.

    Arguments:
        f_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        g_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        split_along_dim (integer): dimension along which the tensor is split into the two parts requried for the reversible block
        fix_random_seed (boolean): Use the same random seed for the forward and backward pass if set to true
    """

    def __init__(self, f_block, g_block, split_along_dim=1, fix_random_seed=False):
        super(ReversibleBlock, self).__init__()
        self.f_block = f_block
        self.g_block = g_block
        self.split_along_dim = split_along_dim
        self.fix_random_seed = fix_random_seed
        self.random_seeds = {}

    def _init_seed(self, namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace] = random.randint(0, sys.maxsize)
            self._set_seed(namespace)

    def _set_seed(self, namespace):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namespace])

    def forward(self, x):
        """
        Performs the forward pass of the reversible block. Does not record any gradients.
        :param x: Input tensor. Must be splittable along dimension 1.
        :return: Output tensor of the same shape as the input tensor
        """
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        with torch.no_grad():
            self._init_seed("f")
            y1 = x1 + self.f_block(x2)
            self._init_seed("g")
            y2 = x2 + self.g_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim)

    def backward_pass(self, y, dy, retain_graph):
        """
        Performs the backward pass of the reversible block.

        Calculates the derivatives of the block's parameters in f_block and g_block, as well as the inputs of the
        forward pass and its gradients.

        :param y: Outputs of the reversible block
        :param dy: Derivatives of the outputs
        :param retain_graph: Whether to retain the graph on intercepted backwards
        :return: A tuple of (block input, block input derivatives). The block inputs are the same shape as the block outptus.
        """

        # Split the arguments channel-wise
        if y.dtype == torch.half:
            y = y.float()
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        del y
        assert not y1.requires_grad, "y1 must already be detached"
        assert not y2.requires_grad, "y2 must already be detached"
        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_along_dim)
        del dy
        assert not dy1.requires_grad, "dy1 must not require grad"
        assert not dy2.requires_grad, "dy2 must not require grad"

        # Enable autograd for y1 and y2. This ensures that PyTorch
        # keeps track of ops. that use y1 and y2 as inputs in a DAG
        y1.requires_grad = True
        y2.requires_grad = True

        # Ensures that PyTorch tracks the operations in a DAG
        with torch.enable_grad():
            self._set_seed("g")
            gy1 = self.g_block(y1)

            # Use autograd framework to differentiate the calculation. The
            # derivatives of the parameters of G are set as a side effect
            gy1.backward(dy2, retain_graph=retain_graph)

        with torch.no_grad():
            x2 = y2 - gy1  # Restore first input of forward()
            del y2, gy1

            # The gradient of x1 is the sum of the gradient of the output
            # y1 as well as the gradient that flows back through G
            # (The gradient that flows back through G is stored in y1.grad)
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            self._set_seed("f")
            fx2 = self.f_block(x2)

            # Use autograd framework to differentiate the calculation. The
            # derivatives of the parameters of F are set as a side effec
            fx2.backward(dx1, retain_graph=retain_graph)

        with torch.no_grad():
            x1 = y1 - fx2  # Restore second input of forward()
            del y1, fx2

            # The gradient of x2 is the sum of the gradient of the output
            # y2 as well as the gradient that flows back through F
            # (The gradient that flows back through F is stored in x2.grad)
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Undo the channelwise split
            x = torch.cat([x1, x2.detach()], dim=self.split_along_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_along_dim)

        return x, dx


class _ReversibleModuleFunction(torch.autograd.function.Function):
    """
    Integrates the reversible sequence into the autograd framework
    """

    @staticmethod
    def forward(ctx, x, reversible_blocks, eagerly_discard_variables):
        """
        Performs the forward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param x: input tensor
        :param reversible_blocks: nn.Modulelist of reversible blocks
        :return: output tensor
        """
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)
            x = block(x)
        ctx.y = x.detach()  # not using ctx.save_for_backward(x) saves us memory by beeing able to free ctx.y earlier in the backward pass
        ctx.reversible_blocks = reversible_blocks
        ctx.eagerly_discard_variables = eagerly_discard_variables
        return x

    @staticmethod
    def backward(ctx, dy):
        """
        Performs the backward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param dy: derivatives of the outputs
        :return: derivatives of the inputs
        """
        y = ctx.y
        if ctx.eagerly_discard_variables:
            del ctx.y
        for i in range(len(ctx.reversible_blocks) - 1, -1, -1):
            y, dy = ctx.reversible_blocks[i].backward_pass(
                y, dy, not ctx.eagerly_discard_variables
            )
        if ctx.eagerly_discard_variables:
            del ctx.reversible_blocks
        return dy, None, None


class ReversibleSequence(nn.Module):
    """
    Basic building element for (partially) reversible networks

    A reversible sequence is a sequence of arbitrarly many reversible blocks. The entire sequence is reversible.
    The activations are only saved at the end of the sequence. Backpropagation leverages the reversible nature of
    the reversible sequece to save memory.

    Arguments:
        reversible_blocks (nn.ModuleList): A ModuleList that exclusivly contains instances of ReversibleBlock
        which are to be used in the reversible sequence.
        eagerly_discard_variables (bool): If set to true backward() discards the variables requried for
                calculating the gradient and therefore saves memory. Disable if you call backward() multiple times.
    """

    def __init__(self, reversible_blocks, eagerly_discard_variables=True):
        super(ReversibleSequence, self).__init__()
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)

        self.reversible_blocks = reversible_blocks
        self.eagerly_discard_variables = eagerly_discard_variables

    def forward(self, x):
        """
        Forward pass of a reversible sequence
        :param x: Input tensor
        :return: Output tensor
        """
        x = _ReversibleModuleFunction.apply(
            x, self.reversible_blocks, self.eagerly_discard_variables
        )
        return x


"""
RevMHAEncoder from https://github.com/Pointerformer/Pointerformer/blob/main/models.py
"""


class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states: Tensor):
        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        hidden_states_t = hidden_states.transpose(0, 1)
        mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)[
            0
        ].transpose(0, 1)

        return mha_output


class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = (
            self.output_layer_norm(hidden_states.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        intermediate_output = self.feed_forward(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output


class RevMHAEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        input_dim: int,
        intermediate_dim: int,
        add_init_projection=True,
        problem="TSP",
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.num_hidden_layers = n_layers
        blocks = []
        for _ in range(n_layers):
            f_func = MHABlock(embedding_dim, n_heads)
            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = ReversibleSequence(nn.ModuleList(blocks))
        self.problem = problem

    def forward(self, data, mask=None):
        x = data.x
        x, _ = to_dense_batch(x, data.batch)  # B x N x H
        x = data_augment(x)

        if self.problem == "CVRP":
            demands = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                :, :, :, -1
            ]  # B x (N+1) x (N+1)

            demands = demands[
                :, :1, :
            ].squeeze()  # B x (N+1) # reduce from an edge to a node level
            if demands.dim() == 1:  # only a single elem in batch
                demands = demands.unsqueeze(0)
            demands = demands.unsqueeze(-1)

            x = torch.cat((x, demands), dim=2)

        elif self.problem == "OP":
            prizes = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                :, :, :, -1
            ]  # B x (N+1) x (N+1)

            prizes = prizes[
                :, :1, :
            ].squeeze()  # B x (N+1) # reduce from an edge to a node level
            if prizes.dim() == 1:  # only a single elem in batch
                prizes = prizes.unsqueeze(0)
            prizes = prizes.unsqueeze(-1)

            dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                :, :, :, 0
            ]  # B x (N+1) x (N+1)
            max_size = data.instance_feature
            if max_size.dim() == 0:
                max_size = max_size.unsqueeze(0)
            dists = dists / max_size[:, None, None]
            return_dists = dists[:, :, 0].unsqueeze(-1)

            x = torch.cat((x, prizes, return_dists), dim=2)

        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)

        x = torch.cat([x, x], dim=-1)
        out = self.sequence(x)
        out = torch.stack(out.chunk(2, dim=-1))[-1]

        return out


#### from https://github.com/Pointerformer/Pointerformer/blob/main/utils.py


def augment_xy_data_by_8_fold(xy_data, training=False):
    # xy_data.shape = [B, N, 2]
    # x,y shape = [B, N, 1]

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)

    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # data_augmented.shape = [B, N, 16]
    if training:
        data_augmented = torch.cat(
            (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=2
        )
        return data_augmented

    # data_augmented.shape = [8*B, N, 2]
    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return data_augmented


def data_augment(batch):
    batch = augment_xy_data_by_8_fold(batch, training=True)
    theta = []
    for i in range(8):
        theta.append(
            torch.atan(batch[:, :, i * 2 + 1] / batch[:, :, i * 2]).unsqueeze(-1)
        )
    theta.append(batch)
    batch = torch.cat(theta, dim=2)
    return batch
