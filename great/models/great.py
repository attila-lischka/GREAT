import math

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax, to_dense_batch


class GREATEncoder(nn.Module):
    """
    This class is a wrapper to built a GREAT-based encoder for a model trained with RL to solve TSP
    """

    def __init__(
        self,
        initial_dim,
        hidden_dim,
        num_layers,
        num_nodes,
        heads,
        final_node_layer=True,
        nodeless=False,
        asymmetric=False,
        dropout=0.1,
    ):
        super(GREATEncoder, self).__init__()
        assert (
            hidden_dim % heads == 0
        ), "hidden_dimension must be divisible by the number of heads such that the dimension of the concatenation is equal to hidden_dim again"

        self.nodeless = nodeless
        self.asymmetric = asymmetric
        self.dropout = nn.Dropout(p=dropout)

        self.embedder = Linear(initial_dim, hidden_dim)

        if self.nodeless:
            self.att_layers = [
                GREATLayerNodeless(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
                for _ in range(num_layers - 1)
            ]
            if self.asymmetric:
                self.att_layers.append(
                    GREATLayerAsymmetric(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True
                    )
                )
            else:
                self.att_layers.append(
                    GREATLayer(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True
                    )
                )
        else:
            if self.asymmetric:
                self.att_layers = [
                    GREATLayerAsymmetric(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True
                    )
                    for _ in range(num_layers)
                ]
            else:
                self.att_layers = [
                    GREATLayer(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True
                    )
                    for _ in range(num_layers)
                ]
        self.att_layers = torch.nn.ModuleList(self.att_layers)

        self.ff_layers = [
            FFLayer(hidden_dim, hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ]
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)

        ### Norms for layers
        self.att_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.ff_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.final_node_layer = final_node_layer
        if final_node_layer:
            # node specific components of last layer
            self.final_transconv_layer = TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                edge_dim=hidden_dim,
            )
            self.final_ff_layer_nodes = FFLayer(hidden_dim, hidden_dim * 2, hidden_dim)
            self.final_att_norm_nodes = nn.BatchNorm1d(hidden_dim)
            self.final_ff_norm_nodes = nn.BatchNorm1d(hidden_dim)

            # edge specific components of last layer
            self.final_att_norm_edges = nn.BatchNorm1d(hidden_dim)
            self.final_ff_norm_edges = nn.BatchNorm1d(hidden_dim)
            self.final_ff_layer_edges = FFLayer(hidden_dim, hidden_dim * 2, hidden_dim)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr")
        ### compute initial edge features from the graph data
        if edge_attr.ndimension() == 1:
            edge_attr = edge_attr.unsqueeze(1)  # E x 1
        edges = self.embedder(edge_attr)  # E x hidden_dim
        edge_index = getattr(data, "edge_index")  # 2 x E
        # x = getattr(data, "x")  # N x 1, just to get_number of nodes
        # if x.ndimension() == 1:
        #    x = x.unsqueeze(1)

        for i, layer in enumerate(self.att_layers[:-1]):
            edges_agg = layer(
                edge_attr=edges, edge_index=edge_index, num_nodes=data.num_nodes
            )  # E x H
            edges_agg = self.dropout(edges_agg)
            edges = edges_agg + edges
            edges = self.att_Norms[i](edges)
            # edges_agg = torch.relu(edges_agg)
            # edges = edges_agg + edges  # residual layer
            edge_ff = self.ff_layers[i](edges)
            edge_ff = self.dropout(edge_ff)
            edges = edge_ff + edges
            edges = self.ff_Norms[i](edges)

        layer = self.att_layers[
            -1
        ]  # get last layer which is special as we want to return the node values
        node_embeddings, edges_agg = layer(
            edge_attr=edges,
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            return_nodes=True,
        )
        node_embeddings = self.att_Norms[-1](node_embeddings)
        node_embeddings_ff = self.ff_layers[-1](node_embeddings)
        node_embeddings = node_embeddings_ff + node_embeddings
        node_embeddings = self.ff_Norms[-1](node_embeddings)

        if self.final_node_layer:
            # prepare edge features of last layer for final layer
            edges = edges_agg + edges
            edges = self.final_att_norm_edges(edges)
            edges_ff = self.final_ff_layer_edges(edges)
            edges = edges_ff + edges
            edges = self.final_ff_norm_edges(edges)

            # do one "transformer layer" of node updates and postprocess accordingly
            node_embeddings_att = self.final_transconv_layer(
                x=node_embeddings, edge_index=edge_index, edge_attr=edges
            )
            node_embeddings = node_embeddings + node_embeddings_att
            node_embeddings = self.final_att_norm_nodes(node_embeddings)
            node_embeddings_ff = self.final_ff_layer_nodes(node_embeddings)
            node_embeddings = node_embeddings + node_embeddings_ff
            node_embeddings = self.final_ff_norm_nodes(node_embeddings)

        return node_embeddings


class GREAT(nn.Module):
    """
    A model for edge level classification or regression tasks
    """

    def __init__(
        self,
        initial_dim,
        hidden_dim=32,
        num_layers=3,
        num_nodes=100,
        heads=4,
        num_classes=0,
        regression=False,
        concat=True,
        nodeless=False,
        instance_repr=False,
    ):
        super(GREAT, self).__init__()

        assert (num_classes > 0 or regression) and not (regression and num_classes > 0)
        assert (
            hidden_dim % heads == 0
        ), "hidden_dimension must be divisible by the number of heads such that the dimension of the concatenation is equal to hidden_dim again"

        self.embedder = Linear(initial_dim, hidden_dim)

        self.concat = concat
        self.nodeless = nodeless

        if self.concat:
            if self.nodeless:
                self.att_layers = [
                    GREATLayerNodeless(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=True
                    )
                    for _ in range(num_layers)
                ]
            else:
                self.att_layers = [
                    GREATLayer(
                        hidden_dim, hidden_dim // heads, heads=heads, concat=concat
                    )
                    for _ in range(num_layers)
                ]
        else:
            if self.nodeless:
                self.att_layers = [
                    GREATLayerNodeless(
                        hidden_dim, hidden_dim, heads=heads, concat=False
                    )
                    for _ in range(num_layers)
                ]
            else:
                self.att_layers = [
                    GREATLayer(hidden_dim, hidden_dim, heads=heads, concat=concat)
                    for _ in range(num_layers)
                ]
        self.att_layers = torch.nn.ModuleList(self.att_layers)

        self.ff_layers = [
            FFLayer(hidden_dim, hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ]
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)

        ### Norms for layers
        self.att_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.ff_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        ### MLP for postprocessing
        self.MLP1 = Linear(hidden_dim, hidden_dim // 2)

        if num_classes > 0:
            self.MLP2 = Linear(hidden_dim // 2, num_classes)
        if regression:
            self.MLP2 = Linear(hidden_dim // 2, 1)

        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.regression = regression
        self.instance_repr = instance_repr

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr")
        ### compute initial edge features from the graph data
        if edge_attr.ndimension() == 1:
            edge_attr = edge_attr.unsqueeze(1)  # E x 1
        edges = self.embedder(edge_attr)  # E x hidden_dim
        edge_index = getattr(data, "edge_index")  # 2 x E

        if self.instance_repr:
            for i, layer in enumerate(self.att_layers[:-1]):
                edges_agg = layer(
                    edge_attr=edges, edge_index=edge_index, num_nodes=data.num_nodes
                )  # E x H
                edges = edges_agg + edges
                edges = self.att_Norms[i](edges)
                edge_ff = self.ff_layers[i](edges)
                edges = edge_ff + edges
                edges = self.ff_Norms[i](edges)

            layer = self.att_layers[
                -1
            ]  # get last layer which is special as we want to return the node values
            node_embeddings, _ = layer(
                edge_attr=edges,
                edge_index=edge_index,
                num_nodes=data.num_nodes,
                return_nodes=True,
            )
            node_embeddings = self.att_Norms[-1](node_embeddings)
            node_embeddings_ff = self.ff_layers[-1](node_embeddings)
            node_embeddings = node_embeddings_ff + node_embeddings
            node_embeddings = self.ff_Norms[-1](node_embeddings)

            node_embeddings, _ = to_dense_batch(node_embeddings, data.batch)

            instance_embeddings = torch.mean(node_embeddings, dim=1)

            instance_embeddings = self.MLP1(instance_embeddings)  # B x H//2
            instance_embeddings = torch.relu(instance_embeddings)
            instance_embeddings = self.MLP2(
                instance_embeddings
            )  # B x num_classes (classification with 2 classes) or B x 1 (regression)

            return instance_embeddings

        else:
            for i, layer in enumerate(self.att_layers):
                edges_agg = layer(
                    edge_attr=edges, edge_index=edge_index, num_nodes=data.num_nodes
                )  # E x H
                edges = edges_agg + edges
                edges = self.att_Norms[i](edges)
                edge_ff = self.ff_layers[i](edges)
                edges = edge_ff + edges
                edges = self.ff_Norms[i](edges)

            edges = self.MLP1(edges)  # E x H//2
            edges = torch.relu(edges)
            edges = self.MLP2(
                edges
            )  # E x num_classes (classification with 2 classes) or E x 1 (regression)

            return edges

    def apply_criterion(
        self,
        outputs,
        data,
        criterion,
    ):
        if self.regression:
            # MSE or MAE loss
            if self.instance_repr:
                loss = criterion(outputs.squeeze(), data.instance_target)
            else:
                loss = criterion(outputs.squeeze(), data.edge_target)

        else:
            ### classification task, apply cross entropy to get probabilities
            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
            if self.instance_repr:
                loss = criterion(outputs.squeeze(), data.instance_target)
            else:
                loss = criterion(outputs.squeeze(), data.edge_target)
        return loss


class FFLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(FFLayer, self).__init__()
        self.linear1 = Linear(dim_in, dim_hid)
        self.linear2 = Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class GREATLayer(MessagePassing):
    """
    A node-based GREAT layer for symmetric graphs (i.e., edge feature for edge (i,j) is THE SAME as for edge (j,i))
    """

    def __init__(self, dim_in, dim_out, heads=4, concat=True):
        super().__init__(node_dim=0)
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        self.lin_values = Linear(dim_in, self.heads * dim_out)
        self.lin_keys = Linear(dim_in, self.heads * dim_out)
        self.lin_queries = Linear(dim_in, self.heads * dim_out)

        if concat:
            self.edge_o = Linear(self.heads * dim_out * 2, self.heads * dim_out)
            self.lin_o = Linear(self.heads * dim_out, self.heads * dim_out)
        else:
            self.edge_o = Linear(dim_out * 2, dim_out)
            self.lin_o = Linear(dim_out, dim_out)

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]

        # x = torch.zeros((num_nodes, self.heads, self.dim_out)).float()
        out = self.propagate(
            edge_index, edge_attr=edge_attr
        )  # N x self.heads x self.dim_out

        if self.concat:
            out = out.view(-1, self.heads * self.dim_out)  # N x self.dim_out
            out = self.lin_o(out)  # N x self.dim_out
        else:
            out = out.mean(dim=1)  # N x self.dim_out
            out = self.lin_o(out)  # N x self.dim_out

        edge_agg = torch.cat(
            (out[edge_index[0]], out[edge_index[1]]), dim=1
        )  # E x self.dim_out * 2

        edge_agg = self.edge_o(edge_agg)  # E x self.dim_out

        if return_nodes:
            return out, edge_agg
        else:
            return edge_agg

    def message(self, edge_attr, index):
        h = self.heads
        d_out = self.dim_out

        values = self.lin_values(edge_attr)  # E x dim_out * h
        values = values.view(-1, h, d_out)  # E x h x dim_out

        queries = self.lin_queries(edge_attr)  # E x dim_out * h
        queries = queries.view(-1, h, d_out)  # E x h x dim_out

        keys = self.lin_keys(edge_attr)  # E x dim_out * h
        keys = keys.view(-1, h, d_out)  # E x h x dim_out

        alpha = (queries * keys).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha = softmax(alpha, index)

        out = values * alpha.view(-1, self.heads, 1)

        return out


class GREATLayerAsymmetric(MessagePassing):
    """
    A node-based GREAT layer for asymmetric graphs (i.e., edge feature for edge (i,j) is NOT THE SAME as for edge (j,i))
    """

    def __init__(self, dim_in, dim_out, heads=4, concat=True):
        super().__init__(node_dim=0)
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        # Outgoing edges
        self.lin_values_out = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_out = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_out = Linear(dim_in, self.heads * dim_out)

        # Ingoing edges
        self.lin_values_in = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_in = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_in = Linear(dim_in, self.heads * dim_out)

        if concat:
            self.edge_o = Linear(self.heads * dim_out * 2, self.heads * dim_out)
            self.lin_o = Linear(self.heads * dim_out * 4, self.heads * dim_out)
        else:
            self.edge_o = Linear(dim_out * 2, dim_out)
            self.lin_o = Linear(dim_out * 4, dim_out)

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]

        # x = torch.zeros((num_nodes, self.heads, self.dim_out)).float()
        out = self.propagate(
            edge_index, edge_attr=edge_attr
        )  # N x self.heads x self.dim_out

        if self.concat:
            out = out.view(-1, self.heads * self.dim_out * 4)  # N x self.dim_out * 4
            out = self.lin_o(out)  # N x self.dim_out
        else:
            out = out.mean(dim=1)  # N x self.dim_out * 4
            out = self.lin_o(out)  # N x self.dim_out

        edge_agg = torch.cat(
            (out[edge_index[0]], out[edge_index[1]]), dim=1
        )  # E x self.dim_out * 2

        edge_agg = self.edge_o(edge_agg)  # E x self.dim_out

        if return_nodes:
            return out, edge_agg
        else:
            return edge_agg

    def message(self, edge_attr, edge_index):
        h = self.heads
        d_out = self.dim_out

        # ingoing edges
        values_in = self.lin_values_in(
            edge_attr
        )  # E x temp * h; temp = dim_out if concat = False, else dim_out//h
        values_in = values_in.view(-1, h, d_out)  # E x h x temp

        queries_in = self.lin_queries_in(edge_attr)  # E x temp * h
        queries_in = queries_in.view(-1, h, d_out)  # E x h x temp

        keys_in = self.lin_keys_in(edge_attr)  # E x temp * h
        keys_in = keys_in.view(-1, h, d_out)  # E x h x temp

        alpha_in = (queries_in * keys_in).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_in = softmax(alpha_in, edge_index[1])  # E x h

        out_in = values_in * alpha_in.view(-1, self.heads, 1)  # E x h x temp

        #  outgoing edges
        values_out = self.lin_values_out(edge_attr)  # E x temp * h
        values_out = values_out.view(-1, h, d_out)  # E x h x temp

        queries_out = self.lin_queries_out(edge_attr)  # E x temp * h
        queries_out = queries_out.view(-1, h, d_out)  # E x h x temp

        keys_out = self.lin_keys_out(edge_attr)  # E x temp * h
        keys_out = keys_out.view(-1, h, d_out)  # E x h x temp

        alpha_out = (queries_out * keys_out).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_out = softmax(alpha_out, edge_index[0])  # E x h

        out_out = values_out * alpha_out.view(-1, self.heads, 1)  # E x h x temp

        ### Note that this only works because we know that edges are arranged a certain way in our data!
        out_out_swapped = out_out.clone()  # Clone to preserve original tensor
        out_out_swapped[::2], out_out_swapped[1::2] = (
            out_out[1::2].clone(),
            out_out[::2].clone(),
        )

        out_in_swapped = out_in.clone()
        out_in_swapped[::2], out_in_swapped[1::2] = (
            out_in[1::2].clone(),
            out_in[::2].clone(),
        )

        out = torch.cat((out_in, out_in_swapped, out_out, out_out_swapped), dim=2)

        return out


class GREATLayerNodeless(nn.Module):
    """
    Nodefree GREAT Layer. Note that the way it is implemented it can only be used for complete graphs where the edges are in the edge list
    of the pytorch geometric data objects are ordered in a certain way. A more general implementation can be achieved by commenting out the
    corresponding code in the forward function. However, by this the architecture is much slower. It can probably be done much faster by using
    a smarter tensor based torch implementation.
    """

    def __init__(self, dim_in, dim_out, heads=4, concat=True):
        super(GREATLayerNodeless, self).__init__()
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        # Outgoing edges
        self.lin_values_out = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_out = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_out = Linear(dim_in, self.heads * dim_out)

        # Ingoing edges
        self.lin_values_in = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_in = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_in = Linear(dim_in, self.heads * dim_out)

        if concat:
            self.edge_o = Linear(self.heads * dim_out * 4, self.heads * dim_out)
        else:
            self.edge_o = Linear(dim_out * 4, dim_out)

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]

        h = self.heads
        d_out = self.dim_out

        # ingoing edges
        values_in = self.lin_values_in(
            edge_attr
        )  # E x temp * h; temp = dim_out if concat = False, else dim_out//h
        values_in = values_in.view(-1, h, d_out)  # E x h x temp

        queries_in = self.lin_queries_in(edge_attr)  # E x temp * h
        queries_in = queries_in.view(-1, h, d_out)  # E x h x temp

        keys_in = self.lin_keys_in(edge_attr)  # E x temp * h
        keys_in = keys_in.view(-1, h, d_out)  # E x h x temp

        alpha_in = (queries_in * keys_in).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_in = softmax(alpha_in, edge_index[1])  # E x h

        out_in = values_in * alpha_in.view(-1, self.heads, 1)  # E x h x temp

        #  outgoing edges
        values_out = self.lin_values_out(edge_attr)  # E x temp * h
        values_out = values_out.view(-1, h, d_out)  # E x h x temp

        queries_out = self.lin_queries_out(edge_attr)  # E x temp * h
        queries_out = queries_out.view(-1, h, d_out)  # E x h x temp

        keys_out = self.lin_keys_out(edge_attr)  # E x temp * h
        keys_out = keys_out.view(-1, h, d_out)  # E x h x temp

        alpha_out = (queries_out * keys_out).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_out = softmax(alpha_out, edge_index[0])  # E x h

        out_out = values_out * alpha_out.view(-1, self.heads, 1)  # E x h x temp

        if self.concat:
            out_in = out_in.view(-1, self.heads * self.dim_out)  # E x h x dim_out
            out_out = out_out.view(-1, self.heads * self.dim_out)  # E x h x dim_out
        else:
            out_in = out_in.mean(dim=1)  # E x h x dim_out
            out_out = out_out.mean(dim=1)  # E x h x dim_out

        # # the following code can be used on "general graphs. In our case, we know how the edge indices are aranged so we can compute the inv more efficiently"
        # # Initialize the 1D tensor 'inv' with the same size as the number of rows in 'tensor'
        # E = edge_index.size(1)
        # inv = torch.empty(E, dtype=torch.long)

        # # Create a dictionary to map each pair to its index
        # edge_list = edge_index.transpose(0, 1).cpu().tolist()
        # pair_to_index = {tuple(pair): idx for idx, pair in enumerate(edge_list)}

        # # Fill the 'inv' tensor
        # for idx, pair in enumerate(edge_list):
        #     inverse_pair = (pair[1], pair[0])  # Find the inverse of the current pair
        #     inv[idx] = pair_to_index[inverse_pair]

        # inv = inv.to(edge_attr.device)

        ### Note that this only works because we know that edges are arranged a certain way in our data! a more general (slower) approach is commented out above
        E = edge_index.size(1)
        inv = torch.arange(E)

        inv = inv.view(-1, 2)

        inv = inv[:, [1, 0]].contiguous().view(-1)

        ### We know have embeddings for out and ingoing edges, we know group them such that an edge
        ### (i,j) consists of in and outgoing edges to node i and node j
        edge_agg = torch.cat(
            (out_in, out_in[inv], out_out, out_out[inv]), dim=1
        )  # E x self.dim_out * 4

        edge_agg = self.edge_o(edge_agg)  # E x self.dim_out

        return edge_agg
