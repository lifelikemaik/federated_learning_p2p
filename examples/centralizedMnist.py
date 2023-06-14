import wandb
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time
from datetime import datetime
import os

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
    node.set_start_learning(rounds=5, epochs=1)



if __name__ == "__main__":
    wandb.login(key="a2d90cdeb8de7e5e4f8baf1702119bcfee78d1ee")
    wandb.init(
        project="p2pflSTANDALONE",
        sync_tensorboard=True,
        name="Standalone "
        + str(datetime.now().strftime("%H:%M")),
    )
    centralized_train()