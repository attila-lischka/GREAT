from dataclasses import dataclass

import torch

# Code based on https://github.com/yd-kwon/POMO/blob/master/NEW_py_ver/CVRP/POMO/CVRPEnv.py and adapted for OP


@dataclass
class Reset_State:
    dists: torch.Tensor = None  # (B, problem, problem)
    prizes: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    travelled_distance: torch.Tensor = None
    # shape: (batch, pomo)


class OPEnv:
    def __init__(self, distances, prizes, max_size, pomo_size, xasy=False):
        # Const @Load_Problem
        ####################################
        self.batch_size = distances.size(0)
        self.problem_size = distances.size(1)
        self.pomo_size = pomo_size
        self.max_size = max_size
        # IDX.shape: (batch, pomo)
        self.dists = (
            distances / max_size[:, None, None]
        )  # all distances reflect how much they cost relative to the maximum allowed travel distance. this way we always know we need to stop when the travelled distance approaches 1
        # shape: (batch, node, node)
        self.prizes = prizes
        # shape: (B, N)
        self.xasy = xasy  # flag that indicates whether triangle inequality holds

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.travelled_distance = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        ### Skip augmentation during training ###

        self.BATCH_IDX = (
            torch.arange(self.batch_size)[:, None]
            .expand(self.batch_size, self.pomo_size)
            .to(self.dists.device)
        )
        self.POMO_IDX = (
            torch.arange(self.pomo_size)[None, :]
            .expand(self.batch_size, self.pomo_size)
            .to(self.dists.device)
        )

        self.reset_state.dists = self.dists
        self.reset_state.prizes = self.prizes

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.firststep = True
        self.collected_prize = torch.zeros(
            (self.batch_size, self.pomo_size), dtype=torch.float
        ).to(self.dists.device)
        self.return_cost = self.dists[:, :, 0]
        self.return_cost = self.dists + self.return_cost.unsqueeze(1)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long
        ).to(self.dists.device)
        # shape: (batch, pomo, 0~problem)

        self.at_the_depot = torch.ones(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool
        ).to(self.dists.device)
        # shape: (batch, pomo)
        self.travelled_distance = torch.ones(size=(self.batch_size, self.pomo_size)).to(
            self.dists.device
        )
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.dists.device)
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size)
        ).to(self.dists.device)
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool
        ).to(self.dists.device)
        # shape: (batch, pomo)

        reward = None
        done = False
        self.firststep = True

        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.travelled_distance = self.travelled_distance

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        B = self.batch_size
        G = self.pomo_size
        # selected.shape: (batch, pomo)
        if self.firststep:
            travelled_distance_new = torch.zeros(
                size=(self.batch_size, self.pomo_size)
            ).to(self.dists.device)
        else:
            from_indices = self.current_node
            to_indices = selected
            # Create batch indices
            batch_indices = (
                torch.arange(self.batch_size).unsqueeze(-1).expand(-1, self.pomo_size)
            )  # shape (B, G)

            # Now gather
            travelled_distance_new = self.dists[
                batch_indices, from_indices, to_indices
            ]  # shape (B, G)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2
        )
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.at_the_depot = selected == 0

        new_prizes = torch.gather(self.prizes, dim=1, index=selected)
        self.collected_prize += new_prizes

        # shape: (batch, pomo)
        self.travelled_distance -= travelled_distance_new

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float("-inf")
        # shape: (batch, pomo, problem)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = (
            0  # depot is considered unvisited, unless you are AT the depot
        )

        return_expanded = self.return_cost.unsqueeze(1).expand(
            -1, self.pomo_size, -1, -1
        )  # (B, G, N, N) # copy for pomo

        # Build the indices
        b_idx = torch.arange(B).view(B, 1).expand(B, G)  # (B, G)
        g_idx = torch.arange(G).view(1, G).expand(B, G)  # (B, G)

        # Now index
        return_expanded = return_expanded[b_idx, g_idx, selected, :]  # (B, G, N)

        distance_too_far = (
            self.travelled_distance.unsqueeze(-1) - return_expanded < 0
        )  # some error margin
        depot_too_far = distance_too_far[
            :, :, 0
        ]  # (B,G) depot is too far away to reach again

        self.ninf_mask = self.visited_ninf_flag.clone()

        # shape: (batch, pomo, problem)
        if not self.xasy:
            self.ninf_mask[distance_too_far] = float(
                "-inf"
            )  # for xasy we cannot mask like this since there might be an indirecty shortest path that is still short enough
        # shape: (batch, pomo, problem)

        if self.firststep:
            self.firststep = False
            newly_finished = torch.zeros(
                size=(self.batch_size, self.pomo_size), dtype=torch.bool
            ).to(self.dists.device)
        else:
            newly_finished = (
                self.at_the_depot
            )  # if we are back at the depot, we are done
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # if finished (== back at depot) mark all other cities even though they might still be reachable from the depot again (since otherwise we would allow multiple tours)
        self.ninf_mask[self.finished] = float("-inf")

        # due to POMO rollouts, it might happen that we move to a very far away node in the first step from which we cannot return anymore without disrespecting the max_length.
        # e.g., in EUC20, if we have the depot at (0,0) and move to a node at (1,1), a single distance is sqrt(2). This twice (go there and return directly) is > 2 which is the max_length of OP20.
        # To not make the model crash, we need to set such cases as to "finished" and make sure they return directly to the depot and don't leave the depot anymore
        invalids = depot_too_far  # we note that for XASY this is not true, since there could be a path that does not directly lead to the depot that is still short enought since triangle inequality does not hold there.

        # do not mask depot for finished episode or invalid ones.
        if not self.xasy:
            self.ninf_mask[:, :, 0][self.finished + invalids] = 0
        else:
            self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.travelled_distance = self.travelled_distance

        # returning values
        done = self.finished.all()
        if done:
            reward = self._get_reward()
        else:
            reward = None

        return self.step_state, reward, done

    def _get_reward(self):
        travelled_too_far = self.travelled_distance < 0
        self.collected_prize[travelled_too_far] = (
            0  # invalid moves dont lead to a reward
        )
        # especially for XASY it can easily happen that we go from node to node because it is very cheap but then there is no short enough connection to the depot any more, so the overall tour is too expensive!
        return self.collected_prize
