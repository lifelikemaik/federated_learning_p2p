#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import wandb
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time


def mnist_execution(n, start, simulation, conntect_to=None, iid=True):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(
            # CNN(),
            MLP(),
            MnistFederatedDM(sub_id=i, number_sub=n, iid=iid),
            simulation=simulation,
        )
        config = {
            "model": "MLP",
            "namenode": "node_"+str(i)
        }
        wandb.init(project="p2pfl", config=config, sync_tensorboard=True)
        node.start()
        wandb.watch(node.learner.model, log="all", log_freq=50)
        nodes.append(node)

    # Connect other network
    if conntect_to is not None:
        nodes[0].connect_to(conntect_to[0], conntect_to[1])

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect_to(nodes[i].host, nodes[i].port, full=True)
        time.sleep(1)

    time.sleep(5)
    print("Starting...")

    for n in nodes:
        print(len(n.get_neighbors()))
        print(len(n.get_network_nodes()))

    # Start Learning
    if start:
        nodes[0].set_start_learning(rounds=10, epochs=2)
    else:
        time.sleep(20)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()


if __name__ == "__main__":
    for _ in range(50):
        mnist_execution(6, True, True)
        break
