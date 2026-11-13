"""Data pipeline tests: the Argoverse 2 parser on a hand-built scenario, and the on-disk store."""

import json

import numpy as np
import pandas as pd
import pytest

from scenecast.data.av2 import load_map, load_scenario
from scenecast.data.scene import AGENT_TYPES, OTHER_TYPE, SAMPLE_SPEC, T_FUT, T_OBS, T_TOTAL, build_sample
from scenecast.data.store import SceneDataset, SplitWriter
from scenecast.data.synthetic import make_scene

SID = "test-scenario"


def _track(track_id, object_type, x0, y0, vx, vy, steps):
    t = np.asarray(steps)
    return pd.DataFrame(
        {
            "track_id": track_id,
            "object_type": object_type,
            "timestep": t,
            "position_x": x0 + vx * t * 0.1,
            "position_y": y0 + vy * t * 0.1,
            "heading": float(np.arctan2(vy, vx)),
            "velocity_x": vx,
            "velocity_y": vy,
            "focal_track_id": "focal",
        }
    )


@pytest.fixture
def scenario_dir(tmp_path):
    folder = tmp_path / SID
    folder.mkdir()
    tracks = pd.concat(
        [
            _track("other", "pedestrian", 105.0, 69.5, 0.0, 1.0, range(20, T_TOTAL)),  # appears late
            _track("focal", "vehicle", 100.0, 50.0, 0.0, 5.0, range(T_TOTAL)),  # drives north at 5 m/s
            _track("cone", "construction", 98.0, 60.0, 0.0, 0.0, range(T_TOTAL)),
            _track("gone", "vehicle", 90.0, 50.0, 3.0, 0.0, range(0, 30)),  # leaves before prediction time
            _track("odd", "something_new", 110.0, 50.0, 0.0, 0.0, range(T_TOTAL)),
        ]
    )
    tracks.to_parquet(folder / f"scenario_{SID}.parquet")
    lane = lambda xs, ys: [{"x": float(x), "y": float(y), "z": 0.0} for x, y in zip(xs, ys)]
    ys = np.linspace(40, 120, 9)
    map_json = {
        "lane_segments": {
            "1": {"centerline": lane(np.full(9, 100.0), ys), "is_intersection": False},
            # no centerline stored: must be recovered as the mean of the two boundaries
            "2": {
                "left_lane_boundary": lane(np.full(9, 102.0), ys),
                "right_lane_boundary": lane(np.full(9, 106.0), ys),
                "is_intersection": True,
            },
        }
    }
    (folder / f"log_map_archive_{SID}.json").write_text(json.dumps(map_json))
    return folder


def test_av2_parser_reads_tracks(scenario_dir):
    scene = load_scenario(scenario_dir)
    assert scene.positions.shape == (5, T_TOTAL, 2)
    assert scene.focal == 1  # second track in the file, found by id rather than by position
    assert scene.types.tolist() == [
        AGENT_TYPES["pedestrian"],
        AGENT_TYPES["vehicle"],
        AGENT_TYPES["construction"],
        AGENT_TYPES["vehicle"],
        OTHER_TYPE,
    ]
    assert scene.valid[1].all()
    assert not scene.valid[0, :20].any() and scene.valid[0, 20:].all()
    assert np.allclose(scene.positions[1, 10], [100.0, 55.0])


def test_av2_map_falls_back_to_boundaries(scenario_dir):
    lanes, attr = load_map(scenario_dir / f"log_map_archive_{SID}.json")
    assert attr == [0, 1]
    assert np.allclose(lanes[1][:, 0], 104.0)


def test_av2_scenario_becomes_focal_centric_sample(scenario_dir):
    sample = build_sample(load_scenario(scenario_dir))
    # the focal car drives north in the world, so in its own frame it drives along +x at 5 m/s
    assert np.allclose(sample["future"][-1], [5.0 * T_FUT * 0.1, 0.0], atol=1e-3)
    assert np.allclose(sample["agents"][0, -1, 4:6], [5.0, 0.0], atol=1e-4)
    # the track that left before the prediction time is not a neighbour
    assert sample["agent_mask"].any(axis=1).sum() == 4
    # the nearest neighbour is the pedestrian, 5 m to the focal car's right and 0.1 m behind
    assert sample["agent_type"][1] == AGENT_TYPES["pedestrian"]
    assert np.allclose(sample["agents"][1, -1, :2], [-0.1, -5.0], atol=1e-3)
    assert not sample["agent_mask"][1, :20].any()
    assert sample["lane_mask"].sum() == 2


def test_unusable_scenes_are_skipped():
    scene = make_scene(np.random.default_rng(0))
    scene.valid[scene.focal, T_OBS - 1] = False  # focal not observed at prediction time
    assert build_sample(scene) is None
    scene = make_scene(np.random.default_rng(0))
    scene.valid[scene.focal, T_OBS:] = False  # nothing to supervise
    assert build_sample(scene) is None


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    samples = [build_sample(make_scene(rng)) for _ in range(5)]
    writer = SplitWriter(tmp_path / "split", capacity=8)  # capacity above the final count
    for i, s in enumerate(samples):
        writer.add(s, f"id-{i}")
    writer.close()

    data = SceneDataset(tmp_path / "split")
    assert len(data) == 5 and data.ids[4] == "id-4"
    assert not list((tmp_path / "split").glob("*.tmp.npy"))
    for k in SAMPLE_SPEC:
        assert np.array_equal(data[3][k].numpy(), samples[3][k]), k
    assert len(SceneDataset(tmp_path / "split", limit=2)) == 2
