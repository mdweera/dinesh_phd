from gflownet_training import FlowModel


class GflowNetFL:
    def __init__(self, num_devices, num_epochs, num_hidlayers):
        self.no_of_devices = num_devices
        self.num_epochs = num_epochs
        self.global_model = FlowModel(num_hidlayers)
        self.global_model_parameters = self.global_model.mlp.state_dict()
        self.local_models = [FlowModel(num_hidlayers) for _ in range(self.no_of_devices)]  # Initialize local models

    def train_local_models(self):
        for epoch in range(self.num_epochs):
            for i in range(self.no_of_devices):
                self.local_models[i] = self.local_models[i].train(self.local_models[i], 10000)
        return self.local_models

    def aggregate_global_model(self):
        for param_name in self.global_model_parameters.keys():
            param_sum = sum(self.local_models[i].state_dict()[param_name] for i in range(self.no_of_devices))
            self.global_model_parameters[param_name] = param_sum / self.no_of_devices

        self.global_model.load_state_dict(self.global_model_parameters)
        return self.global_model


if __name__ == "__main__":
    federated_learner = GflowNetFL(10, 1, 512)
    for gobal_itr in range(10):
        print(f"Global Iteration: {gobal_itr}")
        federated_learner.train_local_models()
        global_model = federated_learner.aggregate_global_model()
        for i in range(federated_learner.no_of_devices):
            federated_learner.local_models[i].mlp.load_state_dict(global_model.state_dict())
