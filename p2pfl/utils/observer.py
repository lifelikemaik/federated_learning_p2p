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
Module that implements the observer pattern.
"""

##################################
#    Generic Observable class    #
##################################


class Events:
    """
    Class that represents the events that can be observed.
    """

    SEND_BEAT_EVENT = "SEND_BEAT_EVENT"
    """
    Used to notify that beats must be sent.
    """
    END_CONNECTION_EVENT = "END_CONNECTION_EVENT"
    """
    Used to notify that a connection has been closed. (arg: NodeConnection)
    """
    AGGREGATION_FINISHED_EVENT = "AGGREGATION_FINISHED_EVENT"
    """
    Used to notify that the aggregation was done. (arg: model or None)
    """
    CONN_TO_EVENT = "CONN_TO_EVENT"
    """
    Used to notify when a node must connect to another. (arg: (host,port))
    """
    START_LEARNING_EVENT = "START_LEARNING_EVENT"
    """
    Used to notify when the learning process starts. (arg: (rounds,epochs))
    """
    STOP_LEARNING_EVENT = "STOP_LEARNING_EVENT"
    """
    Used to notify when the learning process stops.
    """
    PARAMS_RECEIVED_EVENT = "PARAMS_RECEIVED_EVENT"
    """
    Used to notify when the parameters are received. (arg: params (encoded))
    """
    METRICS_RECEIVED_EVENT = "METRICS_RECEIVED_EVENT"
    """
    Used to notify when the metrics are received. (arg: (node, round, loss, metric))
    """
    TRAIN_SET_VOTE_RECEIVED_EVENT = "TRAIN_SET_VOTE_RECEIVED_EVENT"
    """
    Used to notify when a vote is received. (arg: (node,votes))
    """
    NODE_CONNECTED_EVENT = "NODE_CONNECTED_EVENT"
    """
    Used to notify when a node is connected. (arg: (n, force))
    """
    PROCESSED_MESSAGES_EVENT = "PROCESSED_MESSAGES_EVENT"
    """
    Used to notify when a node processes messages. (arg: (node, messages))
    """
    GOSSIP_BROADCAST_EVENT = "GOSSIP_BROADCAST_EVENT"
    """
    Used to notify when a node must send gossiped messages. (arg: (msg,nodes))
    """
    BEAT_RECEIVED_EVENT = "BEAT_RECEIVED_EVENT"
    """
    Used to notify when a node receives a beat. (arg: node)
    """


##################################
#    Generic Observable class    #
##################################


class Observable:
    """
    Class that implements the **Observable** at the observer pattern.
    """

    def __init__(self):
        self.__observers = []

    def add_observer(self, observer):
        """
        Adds an observer to the list of observers.

        Args:
            observer: The observer to add.
        """
        self.__observers.append(observer)

    def get_observers(self):
        """
        Returns the list of observers.

        Returns:
            The list of observers.
        """
        return self.__observers

    def notify(self, event, obj):
        """
        Notifies an event to all the observers.

        Args:
            event: The event to notify.
            obj: The object to pass to the observer. For each event, the object is different (check it at the ``Event`` class).
        """
        [o.update(event, obj) for o in self.__observers]


##################################
#    Generic Observer class    #
##################################


class Observer:
    """
    Class for the **Observer** at the observer pattern.

    Args:
        event: The event that is notified.
        obj: The object that is passed by the observable.
    """

    def update(self, event, obj):
        pass
