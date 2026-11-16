"""The interactive viewer needs a display and is not run in CI; its camera helper does not."""
import mujoco
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
