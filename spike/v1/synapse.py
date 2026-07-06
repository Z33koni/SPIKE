from copy import deepcopy
from math import exp
from collections import deque
from dataclasses import dataclass, field
from random import uniform, randint
from spike.v1.primitive import mV, ms


@dataclass
class SynapticKernel:
    """
    """
    tau: ms = ms(10.0)
    tau_constant: float|None = 0.0

    def __post_init__(self):
        self.tau_constant = exp(-(1.0 / self.tau))


@dataclass
class Synapse:
    amplitude: mV = mV(5.0)
    delay: ms = ms(0)
    kernel: SynapticKernel  = field(default_factory=SynapticKernel)
    potential: mV = mV(0.0)

    def __post_init__(self):
        self.__delays = deque([False] * self.delay, maxlen=self.delay)

    def tick(self, pre_synapse_fired: bool) -> mV:
        self.potential *= self.kernel.tau_constant

        if self.delay == 0:
            now_spike = pre_synapse_fired
        else:
            now_spike = self.__delays.popleft()
            self.__delays.append(pre_synapse_fired)

        if now_spike:
            self.potential += self.amplitude

        return self.potential


@dataclass
class SynapseModel:
    synaptic_amplitude: tuple[mV, mV]
    synaptic_delay: tuple[ms, ms]
    synaptic_tau: tuple[ms, ms]

    def clone(self):
        return deepcopy(self)

    def random_synaptic_amplitude(self) -> mV:
        return uniform(*self.synaptic_amplitude)
    
    def random_synaptic_delay(self) -> ms:
        return randint(*self.synaptic_delay)

    def random_synaptic_tau(self) -> ms:
        return randint(*self.synaptic_tau)

    def random_synapse(self) -> Synapse:
        return Synapse(
            amplitude=self.random_synaptic_amplitude(),
            delay=self.random_synaptic_delay(),
            kernel=SynapticKernel(tau=self.random_synaptic_tau()),
        )

DefaultRecvSynapseModel = SynapseModel(
    synaptic_amplitude=(1.0, 5.0),
    synaptic_delay=(0, 48),
    synaptic_tau=(1, 32),
)

DefaultDataSynapseModel = SynapseModel(
    synaptic_amplitude=(.5, 8.0),
    synaptic_delay=(0, 20),
    synaptic_tau=(4, 32),
)

DefaultSendSynapseModel = SynapseModel(
    synaptic_amplitude=(2.0, 10.0),
    synaptic_delay=(0, 4),
    synaptic_tau=(4, 16),
)
