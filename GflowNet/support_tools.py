import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from gflownet_simulator_tools import Tools, RewardCalc


def calc_face_categorical_output_arr(gflownet_nn_tools, num_rounds):
    gfn_tools = Tools()
    rcalc = RewardCalc()
    sampled_faces = []
    loss_per_face = []
    total_loss = 0
    num_valid_faces = 0  # Initialize a counter for valid faces
    num_smily_faces = 0  # Initialize a counter for smily faces

    for _ in range(num_rounds):
        state = []  # state initialize
        # Prediction phase of F(s, a) with NN.
        edge_flow_prediction = gflownet_nn_tools(gfn_tools.face_to_tensor(state))

        for t in range(3):
            # The policy is calculated by normalizing, and gives us the probability of each action
            policy = edge_flow_prediction / edge_flow_prediction.sum()
            policy = policy / policy.sum()
            action = Categorical(probs=policy).sample()  # probabilistic sample of action on policy distribution
            new_state = state + [gfn_tools.sorted_keys[action]]

            # Computing the loss
            parent_states, parent_actions = gflownet_nn_tools.face_parents(new_state)  # enumerate the parents
            # And compute the edge flows F(s, a) of each parent
            px = torch.stack([gfn_tools.face_to_tensor(p) for p in parent_states])
            pa = torch.tensor(parent_actions).long()
            parent_edge_flow_preds = gflownet_nn_tools(px)[torch.arange(len(parent_states)), pa]

            if t == 2:
                # complete face
                reward = rcalc.face_reward(new_state)
                edge_flow_prediction = torch.zeros(6)
                # Check if the sampled face is valid and if it's a smily face
                if rcalc.is_valid_face(new_state):
                    num_valid_faces += 1
                    if rcalc.is_smily_face(new_state):
                        num_smily_faces += 1
            else:
                # incomplete face
                reward = 0
                edge_flow_prediction = gflownet_nn_tools(gfn_tools.face_to_tensor(new_state))

            # The loss function as per the equation above
            flow_mismatch = (parent_edge_flow_preds.sum() - edge_flow_prediction.sum() - reward).pow(2)
            total_loss += flow_mismatch
            state = new_state
        loss_per_face.append(total_loss)
        sampled_faces.append(state)  # after 4 steps append face
    return [num_valid_faces, num_smily_faces, loss_per_face, sampled_faces]


def visualize_save_results(loss_values_arr, num_agents, num_glb_itr, num_local_itr):
    if num_agents == 1 and num_glb_itr == 1:
        # Handle the case of one agent and one global iteration differently
        plt.plot(loss_values_arr)
        plt.xlabel("Local Iteration Number")
        plt.ylabel("Loss")
        plt.title(f"Agent {0}, Global Iteration {0}")
        plt.show()

    # this check is for maintaining displayability in the plot
    elif num_agents <= 5 and num_glb_itr <= 5:
        # Create subplots for multiple agents and global iterations
        fig, axs = plt.subplots(num_agents, num_glb_itr,
                                figsize=(20, 10))
        for agent in range(num_agents):
            for global_itr in range(num_glb_itr):
                axs[agent, global_itr].plot(loss_values_arr[global_itr][agent])
                axs[agent, global_itr].set_xlabel("Iteration Number")
                axs[agent, global_itr].set_ylabel("Loss")
                axs[agent, global_itr].set_title(f"Agent {agent}, Global Iteration {global_itr}")

        plt.tight_layout()
        plt.show()