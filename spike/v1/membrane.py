import math
from dataclasses import dataclass, field
from spike.v1.primitive import mV, ms


@dataclass
class RefractoryKernel:
    """
    """
    tau: ms = ms(20.0)
    tau_constant: float|None = 0.0

    def __post_init__(self):
        self.tau_constant = math.exp(-(1.0 / self.tau))

@dataclass
class RefractoryModel:
    amplitude: mV = mV(15.0)
    kernel: RefractoryKernel = field(default_factory=RefractoryKernel)
    potential: mV = mV(0.0)

    def tick(self):
        self.potential *= self.kernel.tau_constant
        return self.potential
    
    def fire(self):
        self.potential += self.amplitude

@dataclass
class Membrane:
    # Resting potential.
    resting_potential: mV = mV(-70.0)

    # Triggered firing potential.
    firing_potential: mV = mV(-55.0)

    # After firingm the potential returns 
    reset_potential: mV = mV(-70.0)
    
    # Potential tracking
    potential_curr: mV | None = None 
    potential_prev: mV | None = None 
    
    # The number of ms in the refractory period.
    refractory_model: RefractoryModel = field(default_factory=RefractoryModel)
    
    # Did this membrane fire in the last tick
    spike: bool = field(default_factory=bool)

    def __post_init__(self):
        if self.potential_curr is None:
            self.potential_curr = self.resting_potential

        if self.potential_prev is None:
            self.potential_prev = self.resting_potential
    
    # 
    def tick(self, synaptic_potential: mV) -> bool:
        # Move over the previous potential.
        self.potential_prev = self.potential_curr
        
        # Compute the refractory.
        refractory = self.refractory_model.tick()

        # Check if we should spike.
        self.potential_curr = (
                  self.resting_potential 
                + synaptic_potential 
                - refractory
        )

        # Determine if we should spike.
        self.spike: bool = (
            self.potential_prev <  self.firing_potential and 
            self.potential_curr >= self.firing_potential
        )

        if self.spike:
            self.potential_curr = self.reset_potential
            self.refractory_model.fire()

        return self.spike
