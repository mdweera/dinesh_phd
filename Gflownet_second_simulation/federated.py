import argparse
import utils_data


parser = argparse.ArgumentParser()

# Add arguments for federated learning
parser.add_argument("--num_agents", type=int, default=2, help="Number of agents for federated learning")
parser.add_argument("--local_epochs", type=int, default=5, help="Number of local epochs per agent")
parser.add_argument("--global_rounds", type=int, default=10, help="Number of global aggregation rounds")

args = parser.parse_args()

# Access arguments
num_agents = args.num_agents
local_epochs = args.local_epochs
global_rounds = args.global_rounds

# Split data for federated learning
agent_data_loaders = utils_data.split_dataset_for_fl(train_loader, num_agents)

# Initialize global model
global_model = copy.deepcopy(model)

# Federated learning loop
if __name__ == "__main__":
    for global_round in range(global_rounds):
        print("Global Round:", global_round + 1)

        # Local training
        for agent_id, agent_loader in enumerate(agent_data_loaders):
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr)

            print("Agent:", agent_id + 1)
            for local_epoch in range(local_epochs):
                print("Local Epoch:", local_epoch + 1)
                local_model.train()
                for local_batch in agent_loader:
                    local_batch = preprocess(local_batch[0].to(device))

                    # Compute local loss
                    local_optimizer.zero_grad()
                    local_loss = local_model(local_batch).mean()
                    local_loss.backward()
                    local_optimizer.step()

            # Aggregate local models to update global model
            with torch.no_grad():
                for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                    global_param.data += local_param.data / num_agents

        # Evaluation after global aggregation
        if (global_round + 1) % args.eval_every == 0:
            model.eval()
            print("GFN TEST")
            gfn.model.eval()
            gfn_test_ll = gfn.evaluate(test_loader, preprocess, args.mc_num)
            print("GFN Test log-likelihood ({}) with {} samples: {}".format(global_round + 1, args.mc_num,
                                                                            gfn_test_ll.item()))

            model.cpu()
            d = {}
            d['model'] = global_model.state_dict()
            d['optimizer'] = optimizer.state_dict()
            gfn_ckpt = {"model": gfn.model.state_dict(), "optimizer": gfn.optimizer.state_dict(), }
            gfn_ckpt["logZ"] = gfn.logZ.detach().cpu()
            torch.save(d, "{}/ckpt.pt".format(args.save_dir))
            torch.save(gfn_ckpt, "{}/gfn_ckpt.pt".format(args.save_dir))
            model.to(device)

    print("Federated Learning finished!")
