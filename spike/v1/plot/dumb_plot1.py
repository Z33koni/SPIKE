import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display


def animate_neuron(
    neuron,
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
    potentials: list[float] = []
    spike_times: list[float] = []

    membrane = neuron.membrane

    firing_potential = float(membrane.firing_potential)
    resting_potential = float(membrane.resting_potential)
    reset_potential = float(membrane.reset_potential)

    background_color = "#050b0d"
    trace_yellow = "#ffd54f"
    cyan = "#00e5ff"
    cyan_dim = "#0097a7"
    grid_color = "#1b5e5f"
    text_color = "#b2ebf2"
    threshold_color = "#ffb74d"
    resting_color = "#80cbc4"
    reset_color = "#ce93d8"

    figure, (input_axis, membrane_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        facecolor=background_color,
        constrained_layout=True,
    )

    for axis in (input_axis, membrane_axis):
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
        antialiased=True,
    )

    membrane_line, = membrane_axis.plot(
        [],
        [],
        color=cyan,
        linewidth=1.2,
        antialiased=True,
        label="Membrane potential",
    )

    spike_points, = membrane_axis.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=2.9,
        markerfacecolor=trace_yellow,
        markeredgecolor=trace_yellow,
        label="Spike",
    )

    input_axis.set_title(
        f"Neuron response — {frequency_hz:g} Hz",
        color=text_color,
        fontsize=13,
    )

    input_axis.set_ylabel(
        "Input potential",
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
        alpha=0.9,
        label="Firing threshold",
    )

    membrane_axis.axhline(
        resting_potential,
        color=resting_color,
        linestyle=":",
        linewidth=0.8,
        alpha=0.8,
        label="Resting potential",
    )

    membrane_axis.axhline(
        reset_potential,
        color=reset_color,
        linestyle="-.",
        linewidth=0.8,
        alpha=0.8,
        label="Reset potential",
    )

    membrane_axis.set_xlabel(
        "Time (seconds)",
        color=text_color,
    )

    membrane_axis.set_ylabel(
        "Membrane potential (mV)",
        color=text_color,
    )

    membrane_axis.set_xlim(0.0, duration_s)

    legend = membrane_axis.legend(
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

    try:
        for tick in range(tick_count):
            t = tick * dt_s

            # Positive oscillation from 0 to gain.
            input_value = gain * 0.5 * (
                1.0 + np.cos(
                    2.0 * np.pi * frequency_hz * t
                )
            )

            fired = neuron.tick([input_value])

            potential = float(
                neuron.membrane.potential_curr
            )

            times.append(t)
            inputs.append(input_value)
            potentials.append(potential)

            if fired:
                spike_times.append(t)

            if tick % plot_every != 0 and tick != tick_count - 1:
                continue

            input_line.set_data(times, inputs)
            membrane_line.set_data(times, potentials)

            spike_points.set_data(
                spike_times,
                [firing_potential] * len(spike_times),
            )

            minimum = min(
                potentials,
                default=resting_potential,
            )

            maximum = max(
                potentials,
                default=firing_potential,
            )

            minimum = min(
                minimum,
                reset_potential,
                resting_potential,
            )

            maximum = max(
                maximum,
                firing_potential,
                resting_potential,
            )

            padding = max(
                0.1 * (maximum - minimum),
                1.0,
            )

            membrane_axis.set_ylim(
                minimum - padding,
                maximum + padding,
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
