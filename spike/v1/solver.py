from dataclasses import dataclass, field
from spike.v1.network import Network
import numpy as np


@dataclass
class SimpleSolverMembraneParams:
    clusters: list[np.ndarray]

    @staticmethod
    def from_network(network: Network):
        clusters = []

        for cluster in network.clusters:
            matrix = SimpleSolverMembraneParams.matrix_from_neurons(cluster.neurons)
            clusters.append(matrix)

        return SimpleSolverMembraneParams(clusters)
    
    @staticmethod
    def matrix_from_neurons(neurons: list[Neuron]):
        mat = np.zeros([len(neurons), 5], dtype=float)
        for (n, neuron) in enumerate(neurons):
            col = mat[n]
            
            membrane = neuron.membrane
            refractory = membrane.refractory_model
            col[0] = membrane.resting_potential
            col[1] = membrane.firing_potential
            col[2] = membrane.reset_potential
            col[3] = refractory.amplitude
            col[4] = refractory.kernel.tau_constant
        
        return mat

@dataclass
class SimpleSolverMembraneState:
    clusters: list[np.ndarray]
    
    @staticmethod
    def from_network(network: Network):
        clusters = []

        for cluster in network.clusters:
            matrix = SimpleSolverMembraneState.matrix_from_neurons(cluster.neurons)
            clusters.append(matrix)

        return SimpleSolverMembraneState(clusters)
    
    @staticmethod
    def matrix_from_neurons(neurons: list[Neuron]):
        mat = np.zeros([len(neurons), 4], dtype=float)
        for (n, neuron) in enumerate(neurons):
            col = mat[n]
            
            membrane = neuron.membrane
            refractory = membrane.refractory_model
            col[0] = membrane.potential_curr
            col[1] = membrane.potential_prev
            col[2] = membrane.spike
            col[3] = refractory.potential
        
        return mat


class SimpleSolver:
    def __init__(self, network: Network):
        self.network = network
        self.reset()

    def reset(self):
        self.prev_spikes = [False] * self.network.N
        self.curr_spikes = [False] * self.network.N

    def membrane_params(self):
        return SimpleSolverMembraneParams.from_network(self.network)
    
    def membrane_state(self):
        return SimpleSolverMembraneState.from_network(self.network)

    def tick(self, recv_potentials: list[float]):
        assert len(recv_potentials) == len(self.network.clusters[0].neurons)

        # Creates place to store the neuron input vectors.
        neuron_inputs: list[list[mV]] = [
            [] for _ in range(self.network.N)
        ]

        # Manually insert the activation values into the network.
        for (idx, neuron) in enumerate(self.network.clusters[0].neurons):
            neuron_inputs[neuron.id].append(recv_potentials[idx])

        # Go through every connection and actuate the synapse, storing
        # the result in the neuron_inputs vector.
        for connection in self.network.connections:
            pre_spiked = self.prev_spikes[connection.src_neuron]
            potential = connection.synapse.tick(pre_spiked)
            neuron_inputs[connection.dst_neuron].append(potential)
        
        # Dense neuron pass.
        for i, neuron in enumerate(self.network.neurons):
            self.curr_spikes[i] = neuron.tick(neuron_inputs[i])
        
        # Swap spike states.
        self.prev_spikes = self.curr_spikes
        self.curr_spikes = [False] * self.network.N
