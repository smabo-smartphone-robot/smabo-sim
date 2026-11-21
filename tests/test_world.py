import math

import numpy as np
import pytest

from smabo_sim.world import World


def room(w=4.0, h=2.0):
    return World.from_dict({"shapes": [{"type": "room", "x": 0, "y": 0, "w": w, "h": h}]})


def test_raycast_hits_walls_at_exact_distance():
    r = room().raycast(0.0, 0.0, np.array([0.0, math.pi / 2, math.pi, -math.pi / 2]), 12.0)
    assert r == pytest.approx([2.0, 1.0, 2.0, 1.0])


def test_raycast_beyond_range_is_inf():
    r = room(w=40).raycast(0.0, 0.0, np.array([0.0]), 12.0)
    assert np.isinf(r[0])


def test_raycast_takes_nearest_of_overlapping_shapes():
    w = World.from_dict({"shapes": [
        {"type": "room", "x": 0, "y": 0, "w": 10, "h": 10},
        {"type": "box", "x": 1.0, "y": 0.0, "w": 0.2, "h": 0.2},
    ]})
    assert w.raycast(0, 0, np.array([0.0]), 12.0)[0] == pytest.approx(0.9)


def test_clearance():
    assert room().clearance(1.5, 0.0) == pytest.approx(0.5)


def test_unknown_shape_rejected():
    with pytest.raises(ValueError):
        World.from_dict({"shapes": [{"type": "blob"}]})


def test_bundled_world_loads_and_start_is_free():
    w = World.load()
    assert len(w.segments) > 4
    assert w.clearance(w.start.x, w.start.y) > 0.3


def test_export_map_marks_walls_and_origin(tmp_path):
    w = room()
    pgm, yml = w.export_map(str(tmp_path / "m"), resolution=0.05)
    data = open(pgm, "rb").read()
    header, _, pixels = data.split(b"\n", 3)[0], None, data.split(b"\n", 4)[4]
    assert header == b"P5"
    img = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, int(round((4 + 1.0) / 0.05)))
    assert (img == 0).any() and (img == 254).any()
    # centre of the room is free
    assert img[img.shape[0] // 2, img.shape[1] // 2] == 254
    text = open(yml).read()
    assert "origin: [-2.500, -1.500, 0.0]" in text
    assert "image: m.pgm" in text
