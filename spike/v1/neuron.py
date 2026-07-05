from dataclasses import dataclass, field
from spike.v1.primitive import mV, ms
from spike.v1.membrane import Membrane


@dataclass
class Neuron:
    id: int
    membrane: Membrane = field(default_factory=Membrane)
    
    def tick(self, synaptic_potential: mV) -> bool:
        return self.membrane.tick(synaptic_potential)
