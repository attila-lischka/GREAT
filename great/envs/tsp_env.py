import torch

""" Code based on https://github.com/Pointerformer/Pointerformer/blob/main/env.py """


class TSP_GroupState:
    def __init__(self, group_size, dists):
        # dists.shape = [B, N, N]
        self.batch_size = dists.size(0)
        self.group_size = group_size
        self.device = dists.device

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(
            dists.size(0), group_size, 0, device=dists.device
        ).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(
            dists.size(0), group_size, dists.size(1), device=dists.device
        )

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2
        )
        self.ninf_mask.scatter_(
            dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf
        )


class MultiTrajectoryTSP:
    def __init__(self, dists, integer=False):
        self.integer = integer
        self.dists = dists
        # self.x = x
        self.batch_size = self.B = dists.size(0)
        self.graph_size = self.N = dists.size(1)
        # self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.group_state = None

    def reset(self, group_size):
        self.group_size = group_size
        self.group_state = TSP_GroupState(group_size=group_size, dists=self.dists)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.selected_count == self.graph_size
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        # ordered_seq.shape = [B, G, N, C]

        # G = self.group_size
        ordered_seq = self.group_state.selected_node_list  # B x G x N
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)  # B x G x N

        batch_indices = (
            torch.arange(self.B).unsqueeze(1).unsqueeze(2).expand_as(ordered_seq)
        )

        segment_lengths = self.dists[batch_indices, ordered_seq, rolled_seq]
        # segment_lengths.size = [B, G, N]
        if self.integer:
            group_travel_distances = segment_lengths.round().sum(2)
        else:
            group_travel_distances = segment_lengths.sum(2)
        return group_travel_distances
