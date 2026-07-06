# test_axon.py

import pytest

from spike.v1.primitive import ms
from spike.v1.axon import Axon


def test_axon_zero_delay_outputs_immediately():
    axon = Axon(delay=ms(0))

    assert axon.tick(True) is True
    assert axon.fired is True
    assert axon.output is True


def test_axon_zero_delay_no_spike_outputs_false():
    axon = Axon(delay=ms(0))

    assert axon.tick(False) is False
    assert axon.fired is False
    assert axon.output is False


def test_axon_delay_one_outputs_next_tick():
    axon = Axon(delay=ms(1))

    assert axon.tick(True) is False
    assert axon.tick(False) is True
    assert axon.tick(False) is False


def test_axon_delay_two_outputs_two_ticks_later():
    axon = Axon(delay=ms(2))

    assert axon.tick(True) is False
    assert axon.tick(False) is False
    assert axon.tick(False) is True
    assert axon.tick(False) is False


def test_axon_rejects_negative_delay():
    with pytest.raises(AssertionError):
        Axon(delay=ms(-1))
