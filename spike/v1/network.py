from dataclasses import dataclass, field
from itertools import product
from random import uniform
from spike.v1.neuron import NeuronModel, Neuron
from spike.v1.synapse import SynapseModel, Synapse


@dataclass
class Connection:
    src_neuron: int
    dst_neuron: int
    synapse: Synapse = field(default_factory=Synapse)

@dataclass
class Cluster:
    neurons: list[Neuron]
    connections: list[Connection] = field(default_factory=list)

    def connect(
        self, 
        rhs: 'Cluster', 
        connectivity_ratio: float,
        model_synapse: SynapseModel,
    ):
        """
        Connect `self` cluster to a cluster on the rhs.  The connections will
        reside in `self` cluster.
        """
        for (src, dst) in product(self.neurons, rhs.neurons):
            assert isinstance(src, Neuron), src
            assert isinstance(dst, Neuron), dst
            if src == dst:
                continue

            if uniform(0, 1.0) > connectivity_ratio:
                continue

            connection = Connection(
                src_neuron=src.id,
                dst_neuron=dst.id,
                synapse=model_synapse.random_synapse()
            )
            self.connections.append(connection)

@dataclass
class ClusterBuilder:
    model_synapse: SynapseModel
    model_neuron: NeuronModel
    neuron_count: int
    connectivity_ratio: float

    def build(self) -> Cluster:
        # First, construct the neurons.
        neurons = []
        
        for _ in range(self.neuron_count):
            # Create the neuron and add it to the list.
            neuron = self.model_neuron.random_neuron()
            neurons.append(neuron)

        # Build the cluster and connect it to itself.
        cluster = Cluster(neurons=neurons)
        cluster.connect(
            cluster,
            connectivity_ratio=self.connectivity_ratio,
            model_synapse=self.model_synapse
        )
        
        return cluster


class FinalizedCluster:
    def __init__(self, *clusters):
        self.neurons: list[Neuron] = []
        self.connections: list[Neuron] = []
        self.clusters = list(clusters)

        for cluster in self.clusters:
            self.neurons.extend(cluster.neurons)
            self.connections.extend(cluster.connections)
        
        # Now, we want to sort the neurons in increasing order.
        self.neurons.sort(key=lambda x: x.id)

        # We then want to ensure that it's contiguously increasing.
        for (n0, n1) in zip(self.neurons, self.neurons[1:]):
            assert n1.id - n0.id == 1

        # If it is contiguous, let's re-write all of the neurons to have their 
        # ID at 0.  It helps speed up lookup times.
        min_id = self.neurons[0].id
        for neuron in self.neurons:
            neuron.id -= min_id

        for connection in self.connections:
            connection.src_neuron -= min_id
            connection.dst_neuron -= min_id
    
    def plot_graph(self):
        import matplotlib
        import matplotlib.pyplot as plt
        import networkx as nx

        matplotlib.use("QtAgg")

        G = nx.DiGraph()
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        for (color, cluster) in zip(colors, self.clusters):
            for neuron in cluster.neurons:
                G.add_node(neuron.id, color=color)

        for connection in self.connections:
            G.add_edge(connection.src_neuron, connection.dst_neuron)

        node_colors = [
            G.nodes[n].get("color", "gray")
            for n in G.nodes
        ]

        pos = nx.spring_layout(G, method="energy")
        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.25,
            width=0.1,
        )
        
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=10,      # try 5–20
            linewidths=0,
        )

        plt.show()

