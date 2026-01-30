from speedestimation.output.trails import TrackTrails
from speedestimation.utils.types import TrackState


def _ts(track_id: int, x: float, y: float) -> TrackState:
    return TrackState(
        track_id=track_id,
        camera_id="cam",
        class_id=0,
        class_name="car",
        bbox_xyxy=(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
        score=0.9,
    )


def test_trails_update_and_prune() -> None:
    trails = TrackTrails(max_len=3)
    out1 = trails.update([_ts(1, 10, 10), _ts(2, 20, 20)])
    assert set(out1.keys()) == {1, 2}
    assert len(out1[1]) == 1

    out2 = trails.update([_ts(1, 11, 11)])
    assert set(out2.keys()) == {1}
    assert len(out2[1]) == 2


def test_trails_max_len() -> None:
    trails = TrackTrails(max_len=2)
    trails.update([_ts(1, 0, 0)])
    trails.update([_ts(1, 1, 1)])
    out = trails.update([_ts(1, 2, 2)])
    assert len(out[1]) == 2
    assert out[1][0] == (1.0, 11.0)
    assert out[1][1] == (2.0, 12.0)

