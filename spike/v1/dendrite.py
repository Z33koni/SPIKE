import math
from dataclasses import dataclass, field
from spike.v1.primitive import mV 
from spike.v1.synapse import Synapse


@dataclass
class Dendrite:
    def tick(self, synaptic_potentials: list[mV]) -> mV:
        assert isinstance(synaptic_potentials[0], float)
        return mV(sum(synaptic_potentials))

