import random

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from great.decoding.op_decoder import OPDecoder
from great.envs.op_env import OPEnv

from .great import GREATEncoder
from .matnet import MatNet_Encoder
from .pointerformer import RevMHAEncoder, augment_xy_data_by_8_fold


class GREATRL_OP(nn.Module):
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
        xasy=False,
        pointerformer=False,
    ):
        super().__init__()
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
                "one_hot_seed_cnt": num_nodes + 1,
                "problem": "OP",
            }
            assert (
                num_nodes + 1 + 1 <= hidden_dim
            )  # we need one free dim to encode demand (#nodes + depot + free dim --> therefore + 2)
            self.encoder = MatNet_Encoder(**encoder_params)
        elif pointerformer:
            self.encoder = RevMHAEncoder(
                n_layers=num_layers,
                n_heads=heads,
                embedding_dim=hidden_dim,
                input_dim=26,  # (2D EUC + theta) * 8 + prize + return dist
                intermediate_dim=hidden_dim * 4,
                add_init_projection=True,
                problem="OP",
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
        self.decoder = OPDecoder(
            embedding_dim=hidden_dim,
            n_heads=8,
            tanh_clipping=50,
            multi_pointer=8,
            multi_pointer_level=1,
            add_more_query=True,
        )
        self.xasy = xasy
        self.group_size = group_size
        self.final_node_layer = final_node_layer

    def forward(self, data, return_length=False, augmentation_factor=1):
        assert (
            data.edge_attr.dim() == 2
        ), "edge features need to be 2D (distance and prize)"

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

            data.edge_attr[:, 0] = data.edge_attr[:, 0] * factor

        embeddings = self.encoder(data)  # B*(N+1) x H

        if embeddings.dim() == 2:
            embeddings, _ = to_dense_batch(embeddings, data.batch)  # B x (N+1) x H
        dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
            :, :, :, 0
        ]  # B x (N+1) x (N+1)
        prizes = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
            :, :, :, -1
        ]  # B x (N+1) x (N+1)

        prizes = prizes[
            :, :1, :
        ].squeeze()  # B x (N+1) # reduce from an edge to a node level

        self.decoder.reset(dists, embeddings, embeddings.size(1))  # decoder reset

        env = OPEnv(
            dists / factor,
            prizes,
            data.instance_feature,
            self.group_size,
            self.xasy,
        )

        _, _, _ = env.reset()

        prob_list = torch.zeros(size=(embeddings.size(0), env.pomo_size, 0)).to(
            embeddings.device
        )
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()

        while not done:
            selected, prob = self.step(state, embeddings, training=True)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss

        log_prob = prob_list.log().sum(dim=2)
        # shape = (batch, group)

        r_trans = reward.to(embeddings.device)

        advantage = (
            (r_trans - r_trans.mean(dim=1, keepdim=True))
            / (r_trans.std(dim=1, unbiased=False, keepdim=True) + 1e-8)
            if env.pomo_size != 1
            else r_trans
        )

        loss = (-advantage * log_prob).mean()

        if return_length:
            length_max = reward.max(dim=1)[0].mean().clone().detach().item()
            return loss, length_max

        return loss

    def get_tour(self, data, augmentation_factor=1):
        embedding_list, dists_list, prizes_list, env_dists, max_sizes_list = (
            [],
            [],
            [],
            [],
            [],
        )
        assert (
            data.edge_attr.dim() == 2
        ), "edge features need to be 2D (distance and prize)"

        if augmentation_factor > 1:
            step_size = 0.5 / (augmentation_factor // 2)

        possible_factors = [1]
        possible_factors.extend(
            [0.5 + x * step_size for x in range(augmentation_factor // 2)]
        )
        possible_factors.extend(
            [1.5 - x * step_size for x in range(augmentation_factor // 2)]
        )  ## 0.5 ... 1 ... 1.5

        augmentation_factor = len(possible_factors)

        orig_dist = data.edge_attr[:, 0].detach().clone()

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

                dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, 0
                ]  # B x (N+1) x (N+1)
                prizes = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, -1
                ]  # B x (N+1) x (N+1)

                prizes = prizes[
                    :, :1, :
                ].squeeze()  # B x (N+1) # reduce from an edge to a node level
                if prizes.dim() == 1:  # only a single elem in batch
                    prizes = prizes.unsqueeze(0)

                embedding_list.append(embeddings)
                dists_list.append(dists)
                env_dists.append(dists)
                prizes_list.append(prizes)
                max_size = data.instance_feature
                if max_size.dim() == 0:
                    max_size = max_size.unsqueeze(0)
                max_sizes_list.append(max_size)
        elif isinstance(self.encoder, MatNet_Encoder):
            for _ in possible_factors:
                embeddings = self.encoder(data)  # B*(N+1) x H

                if embeddings.dim() == 2:
                    embeddings, _ = to_dense_batch(
                        embeddings, data.batch
                    )  # B x (N+1) x H
                dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, 0
                ]  # B x (N+1) x (N+1)
                prizes = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, -1
                ]  # B x (N+1) x (N+1)

                prizes = prizes[
                    :, :1, :
                ].squeeze()  # B x (N+1) # reduce from an edge to a node level
                if prizes.dim() == 1:  # only a single elem in batch
                    prizes = prizes.unsqueeze(0)

                embedding_list.append(embeddings)
                dists_list.append(dists)
                env_dists.append(dists)
                prizes_list.append(prizes)
                max_size = data.instance_feature
                if max_size.dim() == 0:
                    max_size = max_size.unsqueeze(0)
                max_sizes_list.append(max_size)
        else:
            for f in possible_factors:
                data.edge_attr[:, 0] = orig_dist * f

                embeddings = self.encoder(data)  # B*(N+1) x H

                if embeddings.dim() == 2:
                    embeddings, _ = to_dense_batch(
                        embeddings, data.batch
                    )  # B x (N+1) x H
                dists = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, 0
                ]  # B x (N+1) x (N+1)
                prizes = to_dense_adj(data.edge_index, data.batch, data.edge_attr)[
                    :, :, :, -1
                ]  # B x (N+1) x (N+1)

                prizes = prizes[
                    :, :1, :
                ].squeeze()  # B x (N+1) # reduce from an edge to a node level
                if prizes.dim() == 1:  # only a single elem in batch
                    prizes = prizes.unsqueeze(0)

                embedding_list.append(embeddings)
                dists_list.append(dists)
                env_dists.append(dists / f)
                prizes_list.append(prizes)
                max_size = data.instance_feature
                if max_size.dim() == 0:
                    max_size = max_size.unsqueeze(0)
                max_sizes_list.append(max_size)

        embeddings = torch.cat(embedding_list, dim=0)
        dists = torch.cat(dists_list, dim=0)
        prizes = torch.cat(prizes_list, dim=0)
        env_dists = torch.cat(env_dists, dim=0)
        max_sizes = torch.cat(max_sizes_list, dim=0)

        self.decoder.reset(dists, embeddings, embeddings.size(1))  # decoder reset

        env = OPEnv(
            env_dists,
            prizes,
            max_sizes,
            self.group_size,
            self.xasy,
        )

        _, _, _ = env.reset()

        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()

        pi = None
        while not done:
            selected, _ = self.step(state, embeddings, training=False)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            if pi is None:
                start_nodes = torch.zeros_like(selected, device=embeddings.device)
                pi = torch.cat([start_nodes[..., None], selected[..., None]], dim=-1)
            else:
                pi = torch.cat([pi, selected[..., None]], dim=-1)

        B = embeddings.size(0)
        G = self.group_size
        B = round(B / augmentation_factor)
        reward = reward.reshape(augmentation_factor, B, G)
        max_tour_length = pi.shape[-1]
        pi = pi.reshape(augmentation_factor, B, G, max_tour_length)  # fix needed

        max_reward_aug_ntraj, idx_dim_2 = reward.max(dim=2)
        max_reward_aug_ntraj, idx_dim_0 = max_reward_aug_ntraj.max(dim=0)

        idx_dim_0 = idx_dim_0.to(pi.device)
        idx_dim_2 = idx_dim_2.to(pi.device)

        idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
        idx_dim_2 = idx_dim_2.reshape(augmentation_factor, B, 1, 1).gather(0, idx_dim_0)
        best_pi_aug_ntraj = pi.gather(0, idx_dim_0.repeat(1, 1, G, max_tour_length))
        best_pi_aug_ntraj = best_pi_aug_ntraj.gather(
            2, idx_dim_2.repeat(1, 1, 1, max_tour_length)
        ).squeeze()

        return max_reward_aug_ntraj.mean().clone().detach().item(), best_pi_aug_ntraj

    def step(self, state, embeddings, training):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(
                embeddings.device
            )
            prob = torch.ones(size=(batch_size, pomo_size)).to(embeddings.device)

            encoded_first_node = _get_encoding(embeddings, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = (
                torch.arange(start=1, end=pomo_size + 1)[None, :]
                .expand(batch_size, pomo_size)
                .to(embeddings.device)
            )
            prob = torch.ones(size=(batch_size, pomo_size)).to(embeddings.device)

        else:
            probs = self.decoder(
                state.current_node, state.ninf_mask, state.travelled_distance
            )
            # shape: (batch, pomo, job)

            if training:
                while (
                    True
                ):  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = (
                            probs.reshape(batch_size * pomo_size, -1)
                            .multinomial(1)
                            .squeeze(dim=1)
                            .reshape(batch_size, pomo_size)
                        )
                        # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(
                        batch_size, pomo_size
                    )
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


### UTILS ###


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(
        batch_size, pomo_size, embedding_dim
    )
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes
