import math

import numpy as np
import torch
import torch.nn.functional as F

import great.decoding.utils as utils

"""Code by https://github.com/Pointerformer/Pointerformer """


class CVRPDecoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=8,
        tanh_clipping=10.0,
        multi_pointer=1,
        multi_pointer_level=1,
        add_more_query=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.multi_pointer = multi_pointer
        self.multi_pointer_level = multi_pointer_level
        self.add_more_query = add_more_query

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W_visited = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_load = torch.nn.Linear(1, embedding_dim)

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

        self.load_embedder = torch.nn.Linear(1, self.embedding_dim)

    def reset(self, dists, embeddings, P):
        B, N, H = embeddings.shape

        # self.coordinates = coordinates  # [:,:2]
        self.dists = dists
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, P, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.logit_k = embeddings.transpose(1, 2)
        if self.multi_pointer > 1:
            self.logit_k = utils.make_heads(
                self.Wk(embeddings), self.multi_pointer
            ).transpose(2, 3)  # [B, n_heads, key_dim, N]

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = self.Wq_first(encoded_q1)

    def forward(self, last_node, group_ninf_mask, load):
        B, N, H = self.embeddings.shape
        P = group_ninf_mask.size(1)

        # Get last node embedding
        last_node_index = last_node.view(B, P, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = self.Wq_last(last_node_embedding)

        group_ninf_mask = group_ninf_mask.detach()

        mask_visited = group_ninf_mask.clone()
        mask_visited[mask_visited == -np.inf] = 1.0
        q_visited = self.W_visited(torch.bmm(mask_visited, self.embeddings) / N)
        # D = self.coordinates.size(-1)
        # last_node_coordinate = self.coordinates.gather(
        #    dim=1, index=last_node.unsqueeze(-1).expand(B, G, D)
        # )
        # distances = torch.cdist(
        #    last_node_coordinate, self.coordinates
        # )  ### and B, G, D and  B x N x D --> B x G x N

        batch_indices = torch.arange(B).unsqueeze(1)
        distances = self.dists[batch_indices, last_node]  # B x G x N

        if self.add_more_query:
            final_q = q_last + self.q_first + self.q_graph + q_visited
        else:
            final_q = q_last + self.q_first + self.q_graph

        load_exp = load.unsqueeze(2).clone()
        load_emb = self.load_embedder(load_exp)
        final_q = final_q + load_emb

        if self.multi_pointer > 1:
            final_q = utils.make_heads(
                self.wq(final_q), self.n_heads
            )  # (B,n_head,G,H)  (B,n_head,H,N)
            score = (torch.matmul(final_q, self.logit_k) / math.sqrt(H)) - (
                distances / math.sqrt(2)
            ).unsqueeze(1)  # (B,n_head,G,N)
            if self.multi_pointer_level == 1:
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8

                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs
        else:
            score = torch.matmul(final_q, self.logit_k) / math.sqrt(
                H
            ) - distances / math.sqrt(2)
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        score_masked = score_clipped + mask_prob
        probs = F.softmax(score_masked, dim=2)

        return probs
