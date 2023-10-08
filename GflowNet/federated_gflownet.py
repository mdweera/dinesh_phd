from gflownet_training import FlowModel
from gflownet_simulator_tools import Tools
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

class GflowNetFL(Tools):
    def __init__(self, num_devices, number_global_itr, num_hidlayers, num_local_itr):
        self.number_of_agents = num_devices
        self.num_local_itr=num_local_itr
        self.number_global_itr = number_global_itr
        self.global_model = FlowModel(num_hidlayers)
        self.global_model_parameters = self.global_model.mlp.state_dict()
        self.local_models = [FlowModel(num_hidlayers) for _ in range(self.number_of_agents)]  # Initialize local models

    def train_local_models(self):
        losses = []
        for i in range(self.number_of_agents):
            print(f"Local Training agent no: {gobal_itr}")
            self.local_models[i], losses_tmp = self.local_models[i].train(self.local_models[i], self.num_local_itr)
            losses.append(losses_tmp)
        return losses

    def aggregate_global_model(self):
        for param_name in self.global_model_parameters.keys():
            param_sum = sum(self.local_models[i].mlp.state_dict()[param_name] for i in range(self.number_of_agents))
            # print(f"Parameter: {param_name}, Sum: {param_sum}")
            self.global_model_parameters[param_name] = param_sum / self.number_of_agents
        self.global_model.mlp.load_state_dict(self.global_model_parameters)
        return self.global_model


if __name__ == "__main__":
    federated_learner = GflowNetFL(num_devices=3, number_global_itr=3, num_hidlayers=512, num_local_itr=1000)
    all_losses = []
    for gobal_itr in range(federated_learner.number_global_itr):
        print(f"Global Iteration: {gobal_itr}")
        losses = federated_learner.train_local_models()
        all_losses.append(losses)
        global_model = federated_learner.aggregate_global_model()
        for agent in range(federated_learner.number_of_agents):
            print(f"Transmit Global model to agent no: {agent}")
            federated_learner.local_models[agent].mlp.load_state_dict(global_model.mlp.state_dict())

    if federated_learner.number_of_agents == 1 and federated_learner.number_global_itr == 1:
        # Handle the case of one agent and one global iteration differently
        plt.plot(all_losses[0][0])
        plt.xlabel("Local Iteration Number")
        plt.ylabel("Loss")
        plt.title(f"Agent {0}, Global Iteration {0}")
        plt.show()
        data = {'all_losses': all_losses[0][0]}
        file_name = f'results/GFlowNet_without_FL_itr_{federated_learner.num_local_itr}.mat'
        scipy.io.savemat(file_name, data)
    elif federated_learner.number_of_agents <= 5 and federated_learner.number_global_itr <= 5:
        # Create subplots for multiple agents and global iterations
        fig, axs = plt.subplots(federated_learner.number_of_agents, federated_learner.number_global_itr, figsize=(20, 10))
        for agent in range(federated_learner.number_of_agents):
            for global_itr in range(federated_learner.number_global_itr):
                axs[agent, global_itr].plot(all_losses[global_itr][agent])
                axs[agent, global_itr].set_xlabel("Iteration Number")
                axs[agent, global_itr].set_ylabel("Loss")
                axs[agent, global_itr].set_title(f"Agent {agent}, Global Iteration {global_itr}")

        plt.tight_layout()
        plt.show()
        data = {'all_losses': all_losses}
        file_name = f'results/GFlowNet_with_FL_local{federated_learner.num_local_itr}_global{federated_learner.number_global_itr}_agents{federated_learner.number_of_agents}.mat'
        scipy.io.savemat(file_name, data)
    else:
        data = {'all_losses': all_losses}
        file_name = f'results/GFlowNet_with_FL_local{federated_learner.num_local_itr}_global{federated_learner.number_global_itr}_agents{federated_learner.number_of_agents}.mat'
        scipy.io.savemat(file_name, data)
