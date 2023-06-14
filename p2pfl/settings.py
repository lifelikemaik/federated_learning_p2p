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

"""
Module to define constants for the p2pfl system.
"""

import logging
from p2pfl.encrypter import AESCipher

###################
# Global Settings #
###################


class Settings:
    """
    Class to define global settings for the p2pfl system.
    """

    # TCP Block Size
    BLOCK_SIZE = 2048
    """
    Max ammount of bytes to read from the buffer at a time. If ```BLOCK_SIZE`` is not divisible by the block size used for symetric encryption it will be rounded to the next closest value.

    Try to strike a balance between hyper-segmentation and excessively large block size.
    """
    rest = BLOCK_SIZE % AESCipher.get_block_size()
    if rest != 0:
        new_value = BLOCK_SIZE + AESCipher.get_block_size() - rest
        logging.info(
            "Changing buffer size to %d. %d is incompatible with the AES block size.",
            BLOCK_SIZE,
            new_value,
        )
        BLOCK_SIZE = new_value

    # TIMEOUT'S
    NODE_TIMEOUT = 2000
    """
    Timeout (seconds) for a node to be considered dead.
    """
    VOTE_TIMEOUT = 6000
    """
    Timeout (seconds) for a node to wait for a vote.
    """
    AGGREGATION_TIMEOUT = 6000
    """
    Timeout (seconds) for a node to wait for other models. Timeout starts when the first model is added.
    """

    # HEARTBEATER
    HEARTBEAT_PERIOD = 4
    """
    Period (seconds) for the node to send a heartbeat.
    """
    HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 4
    """
    Times by period to refresh the neighbors list.
    """
    WAIT_HEARTBEATS_CONVERGENCE = 10
    """
    Time (seconds) to wait for the heartbeats to converge.
    """

    # TRAIN SET
    TRAIN_SET_SIZE = 10
    """
    Size of the training set.
    """
    TRAIN_SET_CONNECT_TIMEOUT = 500
    """
    Timeout (seconds) to wait for a node to connect to the training set.
    """

    # GOSSIP Messages
    AMOUNT_LAST_MESSAGES_SAVED = 100
    """
    Amount of last messages saved. Used to avoid gossiping cycles.
    """
    GOSSIP_MESSAGES_FREC = 100
    """
    Frequency of gossiping messages (times by second).
    """
    GOSSIP_MESSAGES_PER_ROUND = 100
    """
    Amount of gossip messages per round.
    """

    # GOSSIP Models
    GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 20
    """
    Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping.
    """
    GOSSIP_MODELS_FREC = 1
    """
    Frequency of gossiping models (times by second).
    """
    GOSSIP_MODELS_PER_ROUND = 2
    """
    Amount of gossip models per round.
    """
