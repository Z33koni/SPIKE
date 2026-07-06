from copy import deepcopy
from dataclasses import dataclass, field
from itertools import count
from random import uniform, randint

from spike.v1.primitive import mV, ms
from spike.v1.axon import Axon
from spike.v1.dendrite import Dendrite
from spike.v1.membrane import (
    Membrane, 
    RefractoryKernel, 
    RefractoryModel,
)


@dataclass
class Neuron:
    __counter: count = count()

    id: int = field(default_factory=lambda: Neuron.next_id())
    dendrite: Dendrite = field(default_factory=Dendrite)
    membrane: Membrane = field(default_factory=Membrane)
    axon: Axon = field(default_factory=Axon)
    
    @staticmethod
    def next_id():
        return next(Neuron.__counter)
    
    def tick(self, synaptic_potentials: list[mV]) -> bool:
        synaptic_potential = self.dendrite.tick(synaptic_potentials)
        membrane_fired = self.membrane.tick(synaptic_potential)

        return self.axon.tick(membrane_fired)


@dataclass
class NeuronModel:
    membrane_resting_potential: tuple[float, float]
    membrane_firing_potential: tuple[float, float]
    membrane_reset_potential: tuple[float, float]
    membrane_refractory_amplitude : tuple[float, float]
    membrane_refractory_tau: tuple[int, int]

    def clone(self):
        return deepcopy(self)

    def random_membrane(self) -> Membrane:
        return Membrane(
            resting_potential=self.random_membrane_resting_potential(),
            firing_potential=self.random_membrane_firing_potential(),
            reset_potential=self.random_membrane_reset_potential(),
            refractory_model=RefractoryModel(
                amplitude=self.random_membrane_refractory_amplitude(),
                kernel=RefractoryKernel(
                    tau=self.random_membrane_refractory_tau(),
                )
            )
        )

    def random_membrane_resting_potential(self) -> mV:
        return uniform(*self.membrane_resting_potential)
    
    def random_membrane_firing_potential(self) -> mV:
        return uniform(*self.membrane_firing_potential)

    def random_membrane_reset_potential(self) -> mV:
        return uniform(*self.membrane_reset_potential)
    
    def random_membrane_refractory_amplitude(self) -> float:
        return uniform(*self.membrane_refractory_amplitude)
    
    def random_membrane_refractory_tau(self) -> ms:
        return randint(*self.membrane_refractory_tau)

    def random_neuron(self) -> Neuron:
        return Neuron(membrane=self.random_membrane())


DefaultRecvNeuronModel = NeuronModel(
    membrane_resting_potential=(-70.0, -70.0),
    membrane_firing_potential=(-55.0, -55.0),
    membrane_reset_potential=(-70.0, -70.0),

    membrane_refractory_amplitude=(0.0, 0.0),
    membrane_refractory_tau=(1, 1),
)

DefaultDataNeuronModel = NeuronModel(
    # Membrane: near-biological LIF-ish values.
    membrane_resting_potential=(-72.0, -68.0),
    membrane_firing_potential=(-58.0, -52.0),
    membrane_reset_potential=(-74.0, -68.0),

    # Refractory: enough suppression to prevent constant chatter.
    membrane_refractory_amplitude=(8.0, 20.0),
    membrane_refractory_tau=(4, 32),
)

DefaultSendNeuronModel = NeuronModel(
    membrane_resting_potential=(-70.0, -70.0),
    membrane_firing_potential=(-55.0, -55.0),
    membrane_reset_potential=(-70.0, -70.0),

    membrane_refractory_amplitude=(5.0, 5.0),
    membrane_refractory_tau=(4, 4),
)

