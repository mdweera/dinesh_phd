from gflownet_training import FlowModel
import torch

import scipy.io
import json
import numpy as np
from torch.distributions.categorical import Categorical
from gflownet_simulator_tools import Tools, RewardCalc
from support_tools import calc_face_categorical_output_arr, visualize_save_results





def generate_file_name(num_local_itr, num_glb_itr, num_agents):
    if num_agents == 1:
        file_name = f'results/GFlowNet_without_FL_itr_{num_local_itr}.mat'
    else:
        file_name = f'results/GFlowNet_with_FL_local{num_local_itr}_global{num_glb_itr}_agents{num_agents}.mat'
    return file_name


class GflowNetFL(Tools):
    def __init__(self, num_devices, number_global_itr, num_hidlayers, num_local_itr):
        self.number_of_agents = num_devices
        self.num_local_itr = num_local_itr
        self.number_global_itr = number_global_itr
        self.global_model = FlowModel(num_hidlayers)
        self.global_model_parameters = self.global_model.mlp.state_dict()
        # Local models are properties of the instance
        self.local_models = [FlowModel(num_hidlayers) for _ in range(self.number_of_agents)]  # Initialize local models

    def train_local_models(self):
        losses_all_agents = []
        tmp_results_for_centralized = []
        for agent in range(self.number_of_agents):
            print(f"Local Training agent no: {agent}")
            # self.local_models per agent is updated after training
            self.local_models[agent], \
                losses_current_agent, \
                tmp_results_for_centralized = self.local_models[agent].train(
                gflowNet_nn_tools=self.local_models[agent],
                itr=self.num_local_itr,
                number_of_agents=self.number_of_agents)
            losses_all_agents.append(losses_current_agent)
        return losses_all_agents, tmp_results_for_centralized

    def aggregate_global_model(self):
        for param_name in self.global_model_parameters.keys():
            param_sum = sum(self.local_models[i].mlp.state_dict()[param_name] for i in range(self.number_of_agents))
            # print(f"Parameter: {param_name}, Sum: {param_sum}")
            self.global_model_parameters[param_name] = param_sum / self.number_of_agents
        self.global_model.mlp.load_state_dict(self.global_model_parameters)
        return self.global_model


if __name__ == "__main__":
    federated_learner = GflowNetFL(num_devices=1, number_global_itr=1, num_hidlayers=512, num_local_itr=3000)
    results = []
    losses_global_itr = []
    result_sample_itr = [2, 3,4,5,6,7,8,9]  # global iterations to sample results
    for global_itr in range(federated_learner.number_global_itr):  # Corrected typo here
        print(f"Global Iteration: {global_itr}")
        losses_all_agents, tmp_results_for_centralized = federated_learner.train_local_models()
        losses_global_itr.append(losses_all_agents)
        global_model = federated_learner.aggregate_global_model()

        if global_itr in result_sample_itr:  # Check if this global iteration is in the list to sample results
            cat_result_arr = calc_face_categorical_output_arr(gflownet_nn_tools=federated_learner.global_model,
                                                              num_rounds=1000)
            print(f"num_valid_faces: {cat_result_arr[0]}")
            print(f"num_smily_faces: {cat_result_arr[1]}")

            results.append([cat_result_arr[0], cat_result_arr[1]])

        # Transmit global model to agents
        for agent in range(federated_learner.number_of_agents):
            print(f"Transmit Global model to agent no: {agent}")
            federated_learner.local_models[agent].mlp.load_state_dict(global_model.mlp.state_dict())

    if federated_learner.number_of_agents == 1:
        data_to_save = {
            'losses_global_itr': losses_global_itr[0][0],
            'categorical_results': tmp_results_for_centralized
        }
    else:
        visualize_save_results(losses_global_itr, federated_learner.number_of_agents,
                               federated_learner.number_global_itr,
                               federated_learner.num_local_itr)
        data_to_save = {
            'losses_global_itr': losses_global_itr,
            'categorical_results': results
        }
    file_name = generate_file_name(federated_learner.num_local_itr, federated_learner.number_global_itr,
                                   federated_learner.number_of_agents)

    # Save the data to a .mat file
    scipy.io.savemat(file_name, data_to_save)
    print(f"Saved : {file_name}")

