import torch
import torch.nn as nn
import tqdm

from torch.distributions.categorical import Categorical
from gflownet_simulator_tools import Tools, RewardCalc


class FlowModel(nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6, num_hid), nn.LeakyReLU(), nn.Linear(num_hid, 6))

    def forward(self, x):
        F = self.mlp(x).exp() * (1 - x)
        return F

    def face_parents(self, state):
        parent_states = []  # states that are parents of state
        parent_actions = []  # actions that lead from those parents to state
        for face_part in state:
            # there is a parent without that part
            parent_states.append([i for i in state if i != face_part])
            # The action to get there is the corresponding index of that face part
            parent_actions.append(self.sorted_keys.index(face_part))
        return parent_states, parent_actions

def main():
    # Initialize objects
    gfn_tools = Tools()
    rcalc = RewardCalc()
    F_sa = FlowModel(512)  # 512 hidden layers

    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)
    losses = []
    sampled_faces = []
    minibatch_loss = 0
    update_freq = 4

    # provides a visual progress bar that shows the progress of the loop.
    for episode in tqdm.tqdm(range(50000), ncols=40):
        state = []  # episode state initialize
        edge_flow_prediction = F_sa(gfn_tools.face_to_tensor(state))  # Prediction phase of F(s, a) with NN.
        for t in range(3):
            # The policy is just normalizing, and gives us the probability of each action
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            action = Categorical(probs=policy).sample()     # probabilistic sample of action on policy distribution
            new_state = state + [gfn_tools.sorted_keys[action]]

            # Computing the loss
            parent_states, parent_actions = F_sa.face_parents(new_state)    # enumerate the parents
            # And compute the edge flows F(s, a) of each parent
            px = torch.stack([gfn_tools.face_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = F_sa(px)[torch.arange(len(parent_states)), pa]
            # Now we need to compute the reward and F(s, a) of the current state,
            # which is currently `new_state`
            if t == 2:
                # If we've built a complete face, we're done, so the reward is > 0
                # (unless the face is invalid)
                reward = F_sa.face_reward(new_state)
                # and since there are no children to this state F(s,a) = 0 \forall a
                edge_flow_prediction = torch.zeros(6)
            else:
                # Otherwise we keep going, and compute F(s, a)
                reward = 0
                edge_flow_prediction = F_sa(gfn_tools.face_to_tensor(new_state))

            # The loss as per the equation above
            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            minibatch_loss += flow_mismatch
            state = new_state
        sampled_faces.append(state)     # after 4 steps append face
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            minibatch_loss = 0


if __name__ == "__main__":
    main()


