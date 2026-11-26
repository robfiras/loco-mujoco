"""The interactive viewer needs a display and is not run in CI; its camera helper does not."""
import mujoco
import numpy as np
import pytest

from loco_mujoco.core.visuals.viewer import move_camera


@pytest.mark.parametrize("action", (mujoco.mjtMouse.mjMOUSE_ZOOM, mujoco.mjtMouse.mjMOUSE_ROTATE_V))
def test_move_camera_works_with_the_installed_mujoco(action):
    """mjv_moveCamera lost its scene argument in MuJoCo 4; the helper has to work before and after."""
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=10)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    before = (camera.distance, camera.azimuth, camera.elevation)

    move_camera(model, action, 0.0, 0.1, scene, camera)

    assert (camera.distance, camera.azimuth, camera.elevation) != before


def _scene_with_geoms(model, n, maxgeom):
    scene = mujoco.MjvScene(model, maxgeom)
    for i in range(n):
        mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.1),
                            np.zeros(3), np.eye(3).reshape(-1), np.ones(4, dtype=np.float32))
        scene.ngeom += 1
    return scene


def test_add_user_scene_geoms_uses_the_full_capacity_and_reports_overflow():
    """A scene can be filled completely; one geom more raises a readable error.

    The error message used to fail itself with an AttributeError (``ngeom.maxgeom``), and
    the check was off by one."""
    from types import SimpleNamespace
    from loco_mujoco.core.visuals.viewer import MujocoViewer

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")

    viewer = SimpleNamespace(_scene=mujoco.MjvScene(model, 3), _user_scene=_scene_with_geoms(model, 3, 5))
    MujocoViewer._add_user_scene_geoms(viewer)
    assert viewer._scene.ngeom == 3

    viewer = SimpleNamespace(_scene=mujoco.MjvScene(model, 3), _user_scene=_scene_with_geoms(model, 4, 5))
    with pytest.raises(RuntimeError, match="Ran out of geoms. maxgeom: 3"):
        MujocoViewer._add_user_scene_geoms(viewer)
