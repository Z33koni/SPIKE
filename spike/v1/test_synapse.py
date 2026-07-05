# test_synapse.py
import pytest
from spike.v1.primitive import mV, ms 
from spike.v1.synapse import Synapse


class FakeKernel:
    def __init__(self, tau_constant: float):
        self.tau_constant = tau_constant


def test_synapse_initializes_with_zero_potential():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=1.0),
    )

    assert syn.potential == mV(0.0)


def test_zero_delay_spike_immediately_adds_amplitude():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=1.0),
    )

    value = syn.tick(pre_synapse_fired=True)

    assert value == mV(5.0)
    assert syn.potential == mV(5.0)


def test_zero_delay_no_spike_does_not_add_amplitude():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=1.0),
    )

    value = syn.tick(pre_synapse_fired=False)

    assert value == mV(0.0)
    assert syn.potential == mV(0.0)


def test_synapse_decays_existing_potential_each_tick():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=0.5),
        potential=mV(8.0),
    )

    value = syn.tick(pre_synapse_fired=False)

    assert value == mV(4.0)
    assert syn.potential == mV(4.0)


def test_spike_adds_after_decay():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=0.5),
        potential=mV(8.0),
    )

    value = syn.tick(pre_synapse_fired=True)

    # decay first: 8 * 0.5 = 4
    # then inject: 4 + 5 = 9
    assert value == mV(9.0)
    assert syn.potential == mV(9.0)


def test_delay_one_delivers_spike_one_tick_later():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(1),
        kernel=FakeKernel(tau_constant=1.0),
    )

    v0 = syn.tick(pre_synapse_fired=True)
    v1 = syn.tick(pre_synapse_fired=False)

    assert v0 == mV(0.0)
    assert v1 == mV(5.0)


def test_delay_two_delivers_spike_two_ticks_later():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(2),
        kernel=FakeKernel(tau_constant=1.0),
    )

    v0 = syn.tick(pre_synapse_fired=True)
    v1 = syn.tick(pre_synapse_fired=False)
    v2 = syn.tick(pre_synapse_fired=False)

    assert v0 == mV(0.0)
    assert v1 == mV(0.0)
    assert v2 == mV(5.0)


def test_delayed_spike_then_decays_after_arrival():
    syn = Synapse(
        amplitude=mV(8.0),
        delay=ms(1),
        kernel=FakeKernel(tau_constant=0.5),
    )

    v0 = syn.tick(pre_synapse_fired=True)
    v1 = syn.tick(pre_synapse_fired=False)
    v2 = syn.tick(pre_synapse_fired=False)

    assert v0 == mV(0.0)
    assert v1 == mV(8.0)
    assert v2 == mV(4.0)


def test_multiple_spikes_accumulate():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=1.0),
    )

    v0 = syn.tick(pre_synapse_fired=True)
    v1 = syn.tick(pre_synapse_fired=True)
    v2 = syn.tick(pre_synapse_fired=False)

    assert v0 == mV(5.0)
    assert v1 == mV(10.0)
    assert v2 == mV(10.0)


def test_multiple_spikes_accumulate_with_decay():
    syn = Synapse(
        amplitude=mV(5.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=0.5),
    )

    v0 = syn.tick(pre_synapse_fired=True)
    v1 = syn.tick(pre_synapse_fired=True)
    v2 = syn.tick(pre_synapse_fired=False)

    assert v0 == mV(5.0)

    # tick 1: decay 5 -> 2.5, then add 5 -> 7.5
    assert v1 == mV(7.5)

    # tick 2: decay 7.5 -> 3.75
    assert v2 == mV(3.75)


def test_inhibitory_synapse_uses_negative_amplitude():
    syn = Synapse(
        amplitude=mV(-4.0),
        delay=ms(0),
        kernel=FakeKernel(tau_constant=1.0),
    )

    value = syn.tick(pre_synapse_fired=True)

    assert value == mV(-4.0)
    assert syn.potential == mV(-4.0)


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        Synapse(
            amplitude=mV(5.0),
            delay=ms(-1),
            kernel=FakeKernel(tau_constant=1.0),
        )
