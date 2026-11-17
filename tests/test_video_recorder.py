"""Headless coverage of ``core.visuals.video_recorder.VideoRecorder``.

Despite living under ``visuals/``, the recorder is not display-bound: it takes
plain numpy RGB frames and writes them to an mp4 with ``cv2.VideoWriter``
(``cv2.destroyAllWindows()`` is a no-op with no open windows). The only external
process is the optional ffmpeg compression, which we monkeypatch so the tests
don't depend on ffmpeg being installed in CI -- letting us drive both the
success and the ``CalledProcessError`` branches deterministically.
"""
import subprocess

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from loco_mujoco.core.visuals import video_recorder as vr_mod
from loco_mujoco.core.visuals.video_recorder import VideoRecorder


def _frames(n=3, h=16, w=24):
    # deterministic non-uniform uint8 RGB frames
    return [np.full((h, w, 3), i * 10 % 256, dtype=np.uint8) for i in range(n)]


def test_record_and_stop_no_compress(tmp_path):
    rec = VideoRecorder(path=str(tmp_path), tag="run", video_name="clip",
                        fps=10, compress=False)
    for f in _frames():
        rec(f)  # first call lazily creates the writer
    out = rec.stop()
    assert out is not None
    assert out.endswith("clip.mp4")
    assert (tmp_path / "run" / "clip.mp4").exists()
    # counter advanced so the next clip gets a suffixed name
    assert rec._counter == 1


def test_default_tag_uses_timestamp(tmp_path):
    # tag=None -> the datetime-stamped directory branch
    rec = VideoRecorder(path=str(tmp_path), tag=None, video_name="clip", compress=False)
    assert rec._path.parent == tmp_path
    assert rec._path.name != ""  # a timestamp dir name was generated


def test_counter_suffix_on_second_clip(tmp_path):
    rec = VideoRecorder(path=str(tmp_path), tag="run", video_name="clip", compress=False)
    for f in _frames(2):
        rec(f)
    rec.stop()
    # second recording with the same recorder -> _counter > 0 naming branch
    for f in _frames(2):
        rec(f)
    out = rec.stop()
    assert out.endswith("clip-1.mp4")
    assert (tmp_path / "run" / "clip-1.mp4").exists()


def test_compress_success(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        # emulate ffmpeg: write the requested output (last cmd arg) then succeed
        open(cmd[-1], "wb").close()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(vr_mod.subprocess, "run", fake_run)
    rec = VideoRecorder(path=str(tmp_path), tag="run", video_name="clip",
                        fps=10, compress=True)
    for f in _frames(2):
        rec(f)
    out = rec.stop()
    # os.replace moved the compressed tmp file onto the original path
    assert (tmp_path / "run" / "clip.mp4").exists()
    assert out.endswith("clip.mp4")


def test_compress_failure_is_swallowed(tmp_path, monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(vr_mod.subprocess, "run", boom)
    rec = VideoRecorder(path=str(tmp_path), tag="run", video_name="clip",
                        fps=10, compress=True)
    for f in _frames(2):
        rec(f)
    # failure is caught and logged, not raised; the uncompressed file remains
    out = rec.stop()
    assert (tmp_path / "run" / "clip.mp4").exists()
    assert out.endswith("clip.mp4")
