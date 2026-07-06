from dataclasses import dataclass
from collections import deque

from spike.v1.primitive import ms


@dataclass
class Axon:
    """
    Output side of a neuron.

    Holds only present tick output plus a fixed conduction delay buffer.
    """

    delay: ms = ms(0)
    fired: bool = False
    output: bool = False

    def __post_init__(self):
        assert self.delay >= 0.0
        self.__delay_buffer = deque([False] * self.delay)

    def tick(self, membrane_fired: bool) -> bool:
        """
        Advance axon one tick.

        membrane_fired:
            Whether the membrane emitted a spike this tick.

        returns:
            Whether the spike exits the axon this tick.
        """

        self.fired = membrane_fired

        if self.delay == ms(0):
            self.output = membrane_fired
        else:
            self.output = self.__delay_buffer.popleft()
            self.__delay_buffer.append(membrane_fired)

        return self.output
