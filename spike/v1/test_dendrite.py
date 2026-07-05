# test_dendrite.py
import pytest

from spike.v1.primitive import mV
from spike.v1.dendrite import Dendrite


class FakeSynapse:
    def __init__(self, output: mV):
        self.output = output
        self.calls = []

    def tick(self, pre_synaptic_spike: bool) -> mV:
        self.calls.append(pre_synaptic_spike)
        return self.output


def test_empty_dendrite_returns_zero_potential():
    dendrite = Dendrite()

    potential = dendrite.tick([])

    assert potential == mV(0.0)


def test_single_synapse_output_is_returned():
    synapse = FakeSynapse(output=mV(5.0))
    dendrite = Dendrite(synapses=[synapse])

    potential = dendrite.tick([True])

    assert potential == mV(5.0)
    assert synapse.calls == [True]


def test_multiple_synapse_outputs_are_summed():
    s1 = FakeSynapse(output=mV(5.0))
    s2 = FakeSynapse(output=mV(-2.0))
    s3 = FakeSynapse(output=mV(3.5))

    dendrite = Dendrite(synapses=[s1, s2, s3])

    potential = dendrite.tick([True, False, True])

    assert potential == mV(6.5)

    assert s1.calls == [True]
    assert s2.calls == [False]
    assert s3.calls == [True]


def test_dendrite_passes_spikes_to_matching_synapses_by_index():
    s1 = FakeSynapse(output=mV(1.0))
    s2 = FakeSynapse(output=mV(2.0))
    s3 = FakeSynapse(output=mV(3.0))

    dendrite = Dendrite(synapses=[s1, s2, s3])

    dendrite.tick([False, True, False])

    assert s1.calls == [False]
    assert s2.calls == [True]
    assert s3.calls == [False]


def test_dendrite_raises_when_too_few_spikes_are_provided():
    s1 = FakeSynapse(output=mV(1.0))
    s2 = FakeSynapse(output=mV(2.0))

    dendrite = Dendrite(synapses=[s1, s2])

    with pytest.raises(AssertionError):
        dendrite.tick([True])


def test_dendrite_raises_when_too_many_spikes_are_provided():
    s1 = FakeSynapse(output=mV(1.0))

    dendrite = Dendrite(synapses=[s1])

    with pytest.raises(AssertionError):
        dendrite.tick([True, False])


def test_dendrite_accumulates_stateful_synapse_outputs_across_ticks():
    class StatefulSynapse:
        def __init__(self):
            self.potential = mV(0.0)

        def tick(self, pre_synaptic_spike: bool) -> mV:
            if pre_synaptic_spike:
                self.potential += mV(1.0)
            return self.potential

    s1 = StatefulSynapse()
    s2 = StatefulSynapse()

    dendrite = Dendrite(synapses=[s1, s2])

    assert dendrite.tick([True, False]) == mV(1.0)
    assert dendrite.tick([False, True]) == mV(2.0)
    assert dendrite.tick([False, False]) == mV(2.0)
