import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display


def animate_synapse_neuron(
    neuron,
    synapse,
    *,
    gain: float,
    frequency_hz: float,
    duration_s: float = 2.0,
    dt_s: float = 0.001,
    plot_every: int = 10,
    interval_s: float = 0.01,
) -> None:
    if gain < 0.0:
        raise ValueError("gain must be non-negative")

    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")

    if plot_every < 1:
        raise ValueError("plot_every must be at least 1")

    tick_count = int(duration_s / dt_s)

    times: list[float] = []
    inputs: list[float] = []
    membrane_potentials: list[float] = []
    synapse_potentials: list[float] = []

    neuron_spike_times: list[float] = []
    synapse_event_times: list[float] = []
    synapse_event_values: list[float] = []

    membrane = neuron.membrane

    firing_potential = float(membrane.firing_potential)
    resting_potential = float(membrane.resting_potential)
    reset_potential = float(membrane.reset_potential)

    synapse_amplitude = float(synapse.amplitude)

    background_color = "#050b0d"
    trace_yellow = "#ffd54f"
    cyan = "#00e5ff"
    cyan_dim = "#0097a7"
    grid_color = "#1b5e5f"
    text_color = "#b2ebf2"
    threshold_color = "#ffb74d"
    resting_color = "#80cbc4"
    reset_color = "#ce93d8"
    synapse_color = "#ff8a65"

    figure, (
        input_axis,
        membrane_axis,
        synapse_axis,
    ) = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        facecolor=background_color,
        constrained_layout=True,
        gridspec_kw={
            "height_ratios": [1, 2, 1.5],
        },
    )

    for axis in (
        input_axis,
        membrane_axis,
        synapse_axis,
    ):
        axis.set_facecolor(background_color)

        axis.tick_params(
            axis="both",
            colors=text_color,
            labelsize=9,
        )

        for spine in axis.spines.values():
            spine.set_color(cyan_dim)
            spine.set_linewidth(0.8)

        axis.grid(
            True,
            color=grid_color,
            linewidth=0.6,
            alpha=0.65,
        )

        axis.minorticks_on()

        axis.grid(
            True,
            which="minor",
            color=grid_color,
            linewidth=0.3,
            alpha=0.25,
        )

    input_line, = input_axis.plot(
        [],
        [],
        color=cyan,
        linewidth=1.2,
    )

    membrane_line, = membrane_axis.plot(
        [],
        [],
        color=cyan,
        linewidth=1.2,
        label="Membrane potential",
    )

    neuron_spike_points, = membrane_axis.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=2.9,
        markerfacecolor=trace_yellow,
        markeredgecolor=trace_yellow,
        label="Neuron spike",
    )

    synapse_line, = synapse_axis.plot(
        [],
        [],
        color=synapse_color,
        linewidth=1.2,
        label="Synaptic potential",
    )

    synapse_event_points, = synapse_axis.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=3.0,
        markerfacecolor=trace_yellow,
        markeredgecolor=trace_yellow,
        label="Synapse event",
    )

    input_axis.set_title(
        f"Neuron and synapse response — {frequency_hz:g} Hz",
        color=text_color,
        fontsize=13,
    )

    input_axis.set_ylabel(
        "Input",
        color=text_color,
    )

    input_axis.set_xlim(0.0, duration_s)
    input_axis.set_ylim(
        -0.05 * max(gain, 1.0),
        1.05 * max(gain, 1.0),
    )

    membrane_axis.axhline(
        firing_potential,
        color=threshold_color,
        linestyle="--",
        linewidth=0.9,
        label="Firing threshold",
    )

    membrane_axis.axhline(
        resting_potential,
        color=resting_color,
        linestyle=":",
        linewidth=0.8,
        label="Resting potential",
    )

    membrane_axis.axhline(
        reset_potential,
        color=reset_color,
        linestyle="-.",
        linewidth=0.8,
        label="Reset potential",
    )

    membrane_axis.set_ylabel(
        "Membrane (mV)",
        color=text_color,
    )

    synapse_axis.axhline(
        0.0,
        color=text_color,
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )

    synapse_axis.set_xlabel(
        "Time (seconds)",
        color=text_color,
    )

    synapse_axis.set_ylabel(
        "Synapse (mV)",
        color=text_color,
    )

    for axis in (membrane_axis, synapse_axis):
        legend = axis.legend(
            loc="lower center",
            facecolor=background_color,
            edgecolor=cyan_dim,
            labelcolor=text_color,
            fontsize=8,
        )
        legend.get_frame().set_alpha(0.85)

    time_text = membrane_axis.text(
        0.01,
        0.96,
        "",
        transform=membrane_axis.transAxes,
        color=cyan,
        fontsize=9,
        verticalalignment="top",
    )

    display_handle = display(
        figure,
        display_id=True,
    )

    previous_synapse_potential = float(
        getattr(synapse, "potential", 0.0)
    )

    try:
        for tick in range(tick_count):
            t = tick * dt_s

            input_value = gain * 0.5 * (
                1.0
                + np.cos(
                    2.0 * np.pi * frequency_hz * t
                )
            )

            neuron_fired = neuron.tick([input_value])

            synapse_potential = float(
                synapse.tick(neuron_fired)
            )

            membrane_potential = float(
                neuron.membrane.potential_curr
            )

            times.append(t)
            inputs.append(input_value)
            membrane_potentials.append(membrane_potential)
            synapse_potentials.append(synapse_potential)

            if neuron_fired:
                neuron_spike_times.append(t)

            synapse_delta = (
                synapse_potential
                - previous_synapse_potential
            )

            # Detect the arrival of a delayed excitatory or
            # inhibitory synaptic event.
            synapse_event = (
                synapse_delta > 0.0
                if synapse_amplitude >= 0.0
                else synapse_delta < 0.0
            )

            if synapse_event:
                synapse_event_times.append(t)
                synapse_event_values.append(
                    synapse_potential
                )

            previous_synapse_potential = (
                synapse_potential
            )

            if (
                tick % plot_every != 0
                and tick != tick_count - 1
            ):
                continue

            input_line.set_data(
                times,
                inputs,
            )

            membrane_line.set_data(
                times,
                membrane_potentials,
            )

            neuron_spike_points.set_data(
                neuron_spike_times,
                [firing_potential]
                * len(neuron_spike_times),
            )

            synapse_line.set_data(
                times,
                synapse_potentials,
            )

            synapse_event_points.set_data(
                synapse_event_times,
                synapse_event_values,
            )

            membrane_minimum = min(
                membrane_potentials,
                default=resting_potential,
            )

            membrane_maximum = max(
                membrane_potentials,
                default=firing_potential,
            )

            membrane_minimum = min(
                membrane_minimum,
                reset_potential,
                resting_potential,
            )

            membrane_maximum = max(
                membrane_maximum,
                firing_potential,
                resting_potential,
            )

            membrane_padding = max(
                0.1
                * (
                    membrane_maximum
                    - membrane_minimum
                ),
                1.0,
            )

            membrane_axis.set_ylim(
                membrane_minimum
                - membrane_padding,
                membrane_maximum
                + membrane_padding,
            )

            synapse_minimum = min(
                synapse_potentials,
                default=0.0,
            )

            synapse_maximum = max(
                synapse_potentials,
                default=0.0,
            )

            synapse_minimum = min(
                synapse_minimum,
                0.0,
            )

            synapse_maximum = max(
                synapse_maximum,
                0.0,
            )

            synapse_padding = max(
                0.1
                * (
                    synapse_maximum
                    - synapse_minimum
                ),
                0.5,
            )

            synapse_axis.set_ylim(
                synapse_minimum - synapse_padding,
                synapse_maximum + synapse_padding,
            )

            time_text.set_text(
                f"t = {t:0.3f} s"
            )

            figure.canvas.draw()
            display_handle.update(figure)

            if interval_s > 0.0:
                time.sleep(interval_s)

    finally:
        plt.close(figure)
