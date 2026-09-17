# Blender rendering scripts

These scripts are intentionally separate from LocoMuJoCo's tested core. The core package only
exports animated USD stages; Blender-specific appearance, rendering, compositing, and mosaics live
here because they require an external Blender installation and GPU/display-specific behavior.

Install the optional Python dependencies:

```bash
pip install -e ".[blender]"
```

Export and render one environment:

```bash
python scripts/blender/render_locomujoco.py \
  --env UnitreeH1 --task walk --render --engine CYCLES \
  --blender-executable /Applications/Blender.app/Contents/MacOS/Blender
```

Render the 12 humanoids from a LAFAN1 dance clip and create a native-resolution mosaic:

```bash
python scripts/blender/render_locomujoco.py \
  --all-humanoids --dataset-source lafan1 --task dance2_subject4 \
  --steps 751 --render-frame 150 --overlay-interval 0.5 \
  --overlay-opacity 0.3 --overlay-whiten 0.55 \
  --render --engine CYCLES --format png --mosaic \
  --blender-executable /Applications/Blender.app/Contents/MacOS/Blender
```

`render_usd.py` is executed inside Blender by `render_locomujoco.py`; it is not intended to be
called directly. The runner and compositor have full unit-test coverage. A real Blender/Cycles
integration test runs automatically when Blender is installed and otherwise skips, so CI hosts
without Blender can still run the portable USD suite.
