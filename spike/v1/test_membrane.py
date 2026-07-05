# test_membrane.py
import pytest
from spike.v1.primitive import mV, ms 
from spike.v1.membrane import Membrane


def test_membrane_initializes_to_resting_potential():
    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
    )

    assert membrane.potential_curr == mV(-70.0)
    assert membrane.potential_prev == mV(-70.0)


def test_membrane_does_not_spike_below_threshold():
    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
    )

    spike = membrane.tick(synaptic_potential=mV(10.0))

    # -70 + 10 = -60, still below -55
    assert spike is False
    assert membrane.potential_curr == mV(-60.0)


def test_membrane_spikes_on_upward_threshold_crossing():
    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
    )

    spike = membrane.tick(synaptic_potential=mV(20.0))

    # -70 + 20 = -50, crosses -55 from below
    assert spike is True


def test_membrane_resets_after_spike():
    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-75.0),
    )

    spike = membrane.tick(synaptic_potential=mV(20.0))

    assert spike is True
    assert membrane.potential_curr == mV(-75.0)


def test_membrane_does_not_spike_without_upward_crossing():
    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
    )

    membrane.potential_curr = mV(-50.0)
    membrane.potential_prev = mV(-50.0)

    spike = membrane.tick(synaptic_potential=mV(20.0))

    # Still above threshold, but no upward crossing.
    assert spike is False


class FakeRefractoryModel:
    def __init__(self, potential=mV(0.0)):
        self.potential = potential
        self.tick_calls = []

    def tick(self, spike: bool):
        self.tick_calls.append(spike)
        return self.potential


def test_membrane_uses_refractory_potential():
    refractory = FakeRefractoryModel(potential=mV(-5.0))

    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
        refractory_model=refractory,
    )

    spike = membrane.tick(synaptic_potential=mV(20.0))

    # -70 + 20 - 5 = -55
    assert spike is True


def test_membrane_ticks_refractory_model_once_with_spike_value():
    refractory = FakeRefractoryModel(potential=mV(0.0))

    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
        refractory_model=refractory,
    )

    spike = membrane.tick(synaptic_potential=mV(20.0))

    assert spike is True
    assert refractory.tick_calls == [True]


def test_membrane_ticks_refractory_model_once_without_spike():
    refractory = FakeRefractoryModel(potential=mV(0.0))

    membrane = Membrane(
        resting_potential=mV(-70.0),
        firing_potential=mV(-55.0),
        reset_potential=mV(-70.0),
        refractory_model=refractory,
    )

    spike = membrane.tick(synaptic_potential=mV(5.0))

    assert spike is False
    assert refractory.tick_calls == [False]
