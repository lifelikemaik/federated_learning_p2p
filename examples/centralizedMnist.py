import wandb
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time

def centralized_train():
    node = Node(
            CNN(),
            #MLP(),
            MnistFederatedDM(sub_id=0, number_sub=1, iid=False),
            simulation=True,
        )
        
    wandb.init(project="p2pfl", sync_tensorboard=True)
    node.start()
    wandb.watch(node.learner.model, log="all", log_freq=50)
    node.set_start_learning(rounds=10, epochs=4)



if __name__ == "__main__":
    centralized_train()
