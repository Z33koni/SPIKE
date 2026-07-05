import math
from collections import deque
from dataclasses import dataclass, field
from spike.v1.primitive import mV, ms


@dataclass
class SynapticKernel:
    """
    """
    tau: ms = ms(10.0)
    tau_constant: float|None = 0.0

    def __post_init__(self):
        self.tau_constant = math.exp(-(1.0 / self.tau))


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
