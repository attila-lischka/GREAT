import random

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch_geometric.utils import to_dense_adj, to_dense_batch

from great.decoding.tsp_decoder import DecoderForLarge
from great.envs.tsp_env import MultiTrajectoryTSP

from .great import GREATEncoder
from .matnet import MatNet_Encoder
from .pointerformer import RevMHAEncoder, augment_xy_data_by_8_fold


class GREATRL_TSP(nn.Module):
    """
    This class is a wrapper for a GREAT model used in a RL setting to solve the TSP.
    The forward and get_tour function are adapted from https://github.com/Pointerformer/Pointerformer
    """

    def __init__(
        self,
        initial_dim,
        hidden_dim,
        num_layers,
        num_nodes,
        heads,
        group_size=20,
        final_node_layer=True,
        nodeless=False,
        asymmetric=False,
        dropout=0.1,
        matnet=False,
        pointerformer=False,
    ):
        super(GREATRL_TSP, self).__init__()
        assert (
            hidden_dim % heads == 0
        ), "hidden_dimension must be divisible by the number of heads such that the dimension of the concatenation is equal to hidden_dim again"

        if matnet:
            encoder_params = {
                "encoder_layer_num": num_layers,
                "embedding_dim": hidden_dim,
                "head_num": heads,
                "qkv_dim": hidden_dim // heads,
                "ms_hidden_dim": hidden_dim // heads,
                "ms_layer1_init": (1 / 2) ** (1 / 2),
                "ms_layer2_init": (1 / 16) ** (1 / 2),
                "ff_hidden_dim": hidden_dim * 2,
                "one_hot_seed_cnt": num_nodes,
                "problem": "TSP",
            }
            self.encoder = MatNet_Encoder(**encoder_params)
        elif pointerformer:
            self.encoder = RevMHAEncoder(
                n_layers=num_layers,
                n_heads=heads,
                embedding_dim=hidden_dim,
                input_dim=24,  # (2D EUC + theta) * 8
                intermediate_dim=hidden_dim * 4,
                add_init_projection=True,
                problem="TSP",
            )
        else:
            self.encoder = GREATEncoder(
                initial_dim=initial_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_nodes=num_nodes,
                heads=heads,
                final_node_layer=final_node_layer,
                nodeless=nodeless,
                asymmetric=asymmetric,
                dropout=dropout,
            )
        self.decoder = DecoderForLarge(
            embedding_dim=hidden_dim,
            n_heads=8,
            tanh_clipping=50,
            multi_pointer=8,
            multi_pointer_level=1,
            add_more_query=True,
        )
        self.group_size = group_size
        self.final_node_layer = final_node_layer

    def forward(self, data, return_length=False, augmentation_factor=1):
        factor = 1.0
        if augmentation_factor > 1 and isinstance(
            self.encoder, GREATEncoder
        ):  # Matnet and pointerformer learn augmenation "automatically"
            step_size = 0.5 / (augmentation_factor // 2)

            possible_factors = [1]
            possible_factors.extend(
                [0.5 + x * step_size for x in range(augmentation_factor // 2)]
            )
            possible_factors.extend(
                [1.5 - x * step_size for x in range(augmentation_factor // 2)]
            )  ## 0.5 ... 1 ... 1.5
            factor = random.choice(possible_factors)

            if data.edge_attr.dim() == 2:
                data.edge_attr[:, 0] = data.edge_attr[:, 0] * factor
            else:  # only 1D
                data.edge_attr = data.edge_attr * factor

        embeddings = self.encoder(data)

        if embeddings.dim() == 2:
            embeddings, _ = to_dense_batch(embeddings, data.batch)
        if data.edge_attr.dim() == 2:  # augmented edge attributes
            dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                :, :, :, 0
            ]
        else:  # only 1D attributes
            dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)

        B, N, H = embeddings.shape
        G = self.group_size

        batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        group_idx_range = torch.arange(G)[None, :].expand(B, G)

        env = MultiTrajectoryTSP(dists.cpu())
        s, r, d = env.reset(group_size=G)

        # self.decoder.reset(batch.x.view(B, N, new_node_dim), embeddings, G)
        self.decoder.reset(dists, embeddings, G)

        entropy_list = []
        log_prob = torch.zeros(B, G, device=embeddings.device)
        while not d:
            if s.current_node is None:
                first_action = torch.randperm(N)[None, :G].expand(B, G)
                s, r, d = env.step(first_action)
                continue
            else:
                last_node = s.current_node

            action_probs = self.decoder(
                last_node.to(embeddings.device),
                s.ninf_mask.to(embeddings.device),
                s.selected_count,
            )
            m = Categorical(action_probs.reshape(B * G, -1))
            entropy_list.append(m.entropy().mean().item())
            action = m.sample().view(B, G)
            chosen_action_prob = (
                action_probs[batch_idx_range, group_idx_range, action].reshape(B, G)
                + 1e-8
            )
            log_prob += chosen_action_prob.log()
            s, r, d = env.step(action.cpu())

        r_trans = r.to(embeddings.device)

        advantage = (
            (r_trans - r_trans.mean(dim=1, keepdim=True))
            / (r_trans.std(dim=1, unbiased=False, keepdim=True) + 1e-8)
            if G != 1
            else r_trans
        )

        loss = (-advantage * log_prob).mean()

        if return_length:
            length_max = -r.max(dim=1)[0].mean().clone().detach().item()
            length_max = length_max / factor  # renormalize distance by factor
            return loss, length_max

        return loss

    def get_tour(self, data, augmentation_factor=1):
        embedding_list = []
        dists_list = []

        if augmentation_factor > 1:
            step_size = 0.5 / (augmentation_factor // 2)
        possible_factors = [1]
        possible_factors.extend(
            [0.5 + x * step_size for x in range(augmentation_factor // 2)]
        )
        possible_factors.extend(
            [1.5 - x * step_size for x in range(augmentation_factor // 2)]
        )  ## 0.9 ... 1 ... 1.1
        augmentation_factor = len(possible_factors)  # might be increased by 1
        if data.edge_attr.dim() == 2:
            orig_dist = data.edge_attr[:, 0].detach().clone()
        else:
            orig_dist = data.edge_attr.detach().clone()

        if isinstance(self.encoder, RevMHAEncoder):
            if augmentation_factor > 8:
                augmentation_factor = (
                    8  # Pointerformer supports at most x8 augmentation
                )

            coords = data.x.unsqueeze(0)
            coords_augmented = augment_xy_data_by_8_fold(coords)

            for i in range(augmentation_factor):
                data.x = coords_augmented[i, :, :]
                embeddings = self.encoder(data)  # B*(N+1) x H

                if data.edge_attr.dim() == 2:  # augmented edge attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                        :, :, :, 0
                    ]
                else:  # only 1D attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)

                embedding_list.append(embeddings)
                dists_list.append(dists)
        elif isinstance(self.encoder, MatNet_Encoder):
            for _ in possible_factors:
                embeddings = self.encoder(
                    data
                )  # no adjustment necessary, random initial node encodings get different embeddings each time

                if embeddings.dim() == 2:
                    embeddings, _ = to_dense_batch(embeddings, data.batch)
                if data.edge_attr.dim() == 2:  # augmented edge attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                        :, :, :, 0
                    ]
                else:  # only 1D attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
                embedding_list.append(embeddings)
                dists_list.append(dists)
        else:
            for f in possible_factors:
                if data.edge_attr.dim() == 2:
                    data.edge_attr[:, 0] = orig_dist * f
                else:  # only 1D
                    data.edge_attr = orig_dist * f

                embeddings = self.encoder(data)

                if embeddings.dim() == 2:
                    embeddings, _ = to_dense_batch(embeddings, data.batch)
                if data.edge_attr.dim() == 2:  # augmented edge attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                        :, :, :, 0
                    ]
                else:  # only 1D attributes
                    dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
                embedding_list.append(embeddings)
                dists_list.append(dists)

        embeddings = torch.cat(embedding_list, dim=0)
        dists = torch.cat(dists_list, dim=0)

        B, N, H = embeddings.shape
        G = self.group_size

        # batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        # group_idx_range = torch.arange(G)[None, :].expand(B, G)

        env = MultiTrajectoryTSP(dists.cpu())
        s, r, d = env.reset(group_size=G)

        # self.decoder.reset(batch.x.view(B, N, new_node_dim), embeddings, G)
        self.decoder.reset(dists, embeddings, G)

        first_action = torch.randperm(N)[None, :G].expand(B, G)
        pi = first_action[..., None].to(embeddings.device)
        s, r, d = env.step(first_action)

        while not d:
            action_probs = self.decoder(
                s.current_node.to(embeddings.device),
                s.ninf_mask.to(embeddings.device),
                s.selected_count,
            )
            action = action_probs.argmax(dim=2)
            pi = torch.cat([pi, action[..., None]], dim=-1)
            s, r, d = env.step(action.cpu())

        B = round(B / augmentation_factor)
        reward = r.reshape(augmentation_factor, B, G)
        pi = pi.reshape(augmentation_factor, B, G, N)

        # We experimented with additive data augmentation but then moved to multiplicative instead
        # subtraction_values = torch.tensor(
        #    [x * (0.1 / augmentation_factor) for x in range(0, augmentation_factor)]
        # ).view(augmentation_factor, 1, 1)
        # subtraction_values = subtraction_values * N
        # subtraction_values.to(reward.device)

        # reward = (
        #    reward + subtraction_values
        # )  # revert the additional cost that was imposed by increasing the distances of edges

        if isinstance(
            self.encoder, GREATEncoder
        ):  # Pointerformer/Matner does not need rescaling
            coefficients = torch.tensor(possible_factors).view(-1, 1, 1)
            reward = reward / coefficients

        max_reward_aug_ntraj, idx_dim_2 = reward.max(dim=2)
        max_reward_aug_ntraj, idx_dim_0 = max_reward_aug_ntraj.max(dim=0)

        idx_dim_0 = idx_dim_0.to(pi.device)
        idx_dim_2 = idx_dim_2.to(pi.device)

        idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
        idx_dim_2 = idx_dim_2.reshape(augmentation_factor, B, 1, 1).gather(0, idx_dim_0)
        best_pi_aug_ntraj = pi.gather(0, idx_dim_0.repeat(1, 1, G, N))
        best_pi_aug_ntraj = best_pi_aug_ntraj.gather(
            2, idx_dim_2.repeat(1, 1, 1, N)
        ).squeeze()

        return -max_reward_aug_ntraj.mean().clone().detach().item(), best_pi_aug_ntraj
