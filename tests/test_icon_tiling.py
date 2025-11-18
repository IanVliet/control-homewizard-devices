from control_homewizard_devices.e_paper_display import (
    calculate_icon_positions,
    get_segments_upper_and_lower,
)
import numpy as np


def test_calculate_icon_positions_8x24_steps_of_4():
    height = 8
    width = 24
    x_pixels = np.linspace(0, width, num=7, dtype=int)
    upper_pixels = np.zeros(len(x_pixels), dtype=int)
    upper_points = list(zip(x_pixels, upper_pixels, strict=True))
    lower_pixels = np.zeros(len(x_pixels), dtype=int) + height
    lower_points = list(zip(x_pixels, lower_pixels, strict=True))
    expected_positions = [(0, 0, 8), (8, 0, 8), (16, 0, 8)]

    positions = calculate_icon_positions(
        [upper_points], [lower_points], init_icon_size=8
    )
    assert positions == expected_positions


def test_calculate_icon_positions_8x24_steps_of_6():
    height = 8
    width = 24
    x_pixels = np.linspace(0, width, num=5, dtype=int)
    upper_pixels = np.zeros(len(x_pixels), dtype=int)
    upper_points = list(zip(x_pixels, upper_pixels, strict=True))
    lower_pixels = np.zeros(len(x_pixels), dtype=int) + height
    lower_points = list(zip(x_pixels, lower_pixels, strict=True))
    expected_positions = [(0, 0, 8), (12, 0, 8)]

    positions = calculate_icon_positions(
        [upper_points], [lower_points], init_icon_size=8
    )
    assert positions == expected_positions


def test_calculate_icon_positions_stepped_16x24():
    height = 16
    width = 24
    x_pixels = np.linspace(0, width, num=7, dtype=int)

    upper_pixels = np.array(
        [8 if idx < 2 else 0 for idx in range(len(x_pixels))], dtype=int
    )
    upper_points = list(zip(x_pixels, upper_pixels, strict=True))
    lower_pixels = np.zeros(len(x_pixels), dtype=int) + height
    lower_points = list(zip(x_pixels, lower_pixels, strict=True))
    expected_positions = [(8, 0, 16), (0, 8, 8)]

    positions = calculate_icon_positions(
        [upper_points], [lower_points], init_icon_size=16
    )
    assert positions == expected_positions


def test_calculate_icon_positions_triangle():
    height = 24
    width = 24
    x_pixels = np.linspace(0, width, num=7, dtype=int)
    upper_pixels = np.array(
        [
            24,
            16,
            8,
            0,
            0,
            8,
            24,
        ],
        dtype=int,
    )
    upper_points = list(zip(x_pixels, upper_pixels, strict=True))
    lower_pixels = np.zeros(len(x_pixels), dtype=int) + height
    lower_points = list(zip(x_pixels, lower_pixels, strict=True))
    expected_positions = [(8, 8, 16)]

    positions = calculate_icon_positions(
        [upper_points], [lower_points], init_icon_size=16
    )
    assert positions == expected_positions


def test_get_shared_x_line_segments():
    initial_upper = [[(0, 0), (100, 0)], [(300, 0), (400, 0)]]
    initial_lower = [[(0, 300), (100, 300), (200, 300), (300, 300), (400, 300)]]

    upper, lower = get_segments_upper_and_lower(initial_upper, initial_lower)
    expected_upper = initial_upper
    expected_lower = [[(0, 300), (100, 300)], [(300, 300), (400, 300)]]

    assert upper == expected_upper
    assert lower == expected_lower


def test_calculate_icon_positions_missing_data():
    height = 8
    upper_points = [[(0, 0), (4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (24, 0)]]
    lower_points = [
        [(0, height), (4, height), (8, height)],
        [(16, height), (20, height), (24, height)],
    ]
    expected_positions = [(0, 0, 8), (16, 0, 8)]

    positions = calculate_icon_positions(upper_points, lower_points, init_icon_size=8)
    assert positions == expected_positions
