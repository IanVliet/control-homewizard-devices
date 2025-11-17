from control_homewizard_devices.e_paper_display import power_array_to_points
import numpy as np
import pytest


def test_no_nan_array_to_points():
    x_pixels = np.array([0, 100, 200, 300, 400])
    power_array = np.array([0, 1, 1, 1, 0])
    plot_height = 300
    points = power_array_to_points(power_array, 0, 1, plot_height, x_pixels)
    desired_points = [
        [
            (0.0, plot_height - 1),
            (100.0, 0.0),
            (200.0, 0.0),
            (300.0, 0.0),
            (400.0, plot_height - 1),
        ]
    ]
    assert points[0] == pytest.approx(desired_points[0])


def test_array_with_nan_to_points():
    x_pixels = np.array([0, 100, 200, 300, 400])
    power_array = np.array([0, 1, np.nan, 1, 0])
    plot_height = 300
    points = power_array_to_points(power_array, 0, 1, plot_height, x_pixels)
    desired_points = [
        [(0.0, plot_height - 1), (100.0, 0.0)],
        [(300.0, 0.0), (400.0, plot_height - 1)],
    ]
    assert points[0] == pytest.approx(desired_points[0])
    assert points[1] == pytest.approx(desired_points[1])
