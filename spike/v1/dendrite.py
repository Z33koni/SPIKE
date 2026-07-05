import math
from dataclasses import dataclass, field
from spike.v1.primitive import mV 
from spike.v1.synapse import Synapse


@dataclass
class Dendrite:
    synapses: list[Synapse] = field(default_factory=list)

    def tick(self, pre_synaptic_spikes: list[bool]) -> mV:
        assert len(pre_synaptic_spikes) == len(self.synapses)

        potential = mV(0.0)
    
        synapse_spike_pairs = zip(self.synapses, pre_synaptic_spikes)
        for (synapse, pre_synaptic_spike) in synapse_spike_pairs:
            potential += synapse.tick(pre_synaptic_spike)
            
        return potential
