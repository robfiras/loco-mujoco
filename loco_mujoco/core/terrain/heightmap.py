import hashlib
import os
from types import ModuleType
from typing import Any, List, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core.terrain.base import hfield_surface_height
from loco_mujoco.core.terrain.static import StaticTerrain


class HeightmapTerrain(StaticTerrain):
    """
    A static terrain loaded from a height map file, shared by all envs and aligned with the world frame.

    Meant for motion datasets whose clips are already placed on their part of the map (e.g. every GRAIL staircase in
    its own plot, with the clip shifted onto it). The terrain reports ``trajectories_in_world_frame``, so episodes
    start where their clip is.

    The file is an ``.npz`` with
        - ``h``: (ny, nx) heights in metres; cell (i, j) is centred at ``origin + (j * cell, i * cell)``,
        - ``cell``: cell size in metres,
        - ``origin``: (2,) world xy of cell (0, 0) (optional, default 0).

    The map is split into square tiles, each its own heightfield geom (neighbouring tiles share their border samples,
    so the surface is continuous). Tiling matters for MuJoCo Warp: its heightfield collision runs in float32 in the
    heightfield's frame, so on one large map a geom tens of metres from the centre collides with prisms at
    coordinates where float32 resolves only ~1e-5 m, and the contacts come out wrong and blow the simulation up.
    Each tile also gets a tight bounding box (its own height range), so geoms well above a tile skip it.

    The tiles are written once to MuJoCo's binary heightfield format in ``cache_dir``.
    """

    CONTACT_BIT = 1 << 3

    POOL_PARK_DEPTH = 10.0  # unused pool slots sit this far below the ground plane

    def __init__(self, env: Any, file: str, cache_dir: str = "~/.cache/loco_mujoco/hfields", base_depth: float = 1.0,
                 contact_geoms: Union[List[str], None] = None, ground_clearance: Union[float, None] = 0.002,
                 tile_size: Union[float, None] = 8.0, contact: str = "hfield", box_quant: float = 0.01,
                 pool_file: Union[str, None] = None, **kwargs: Any):
        """
        Args:
            env: The environment instance.
            file: Path to the height map ``.npz``.
            cache_dir: Where the binary heightfield tiles are written.
            base_depth: Depth of each tile's base box below its lowest point, in metres.
            contact_geoms: If given, only these geoms (e.g. the foot capsules) collide with the terrain, through a
                collision bit of their own (``CONTACT_BIT``). A fine heightfield gives every touching geom dozens of
                prism contacts, so letting the whole body collide with it is expensive.
            ground_clearance: On reset, lift the robot so that its terrain-colliding geoms start at least this far
                above the terrain (metres). Reference motions and reset noise put feet into steps and risers.
                None disables the correction.
            tile_size: Edge length of the square heightfield tiles in metres (rounded to whole cells). None puts the
                whole map in one heightfield.
            contact: "hfield": the heightfield tiles are the collision surface. "boxes": the map is covered with
                axis-aligned boxes (equal heights, quantised to ``box_quant``, merged into rectangles) on a ground
                plane, and the tiles stay in the model without collisions, for height queries only. A heightfield
                turns every step riser into a slope one cell wide, and feet on that slope slide off the tread edge.
                "pool": the boxes of "boxes", but each env only gets the boxes near its own clip. The model holds K box
                slots (K = the most boxes any clip needs), and every step writes the boxes of the env's current
                trajectory into them (MuJoCo Warp keeps geom position, size and bounds per world). With "boxes" every
                world's broadphase tests the contact geoms against every box of the map; this keeps it to K boxes.
                Within ``pool_file``'s reach of a clip's root path the contacts are those of "boxes".
            box_quant: Height quantisation of the boxes in metres.
            pool_file: For ``contact="pool"``: ``.npz`` with ``idx`` (n_trajectories, K), indices into ``boxes()`` of
                each trajectory's boxes (-1 pads), and ``n_boxes``, the box count they index (checked against
                ``boxes()``). Rows follow the trajectory order of the dataset the env is trained on. ``write_pool_file``
                builds it.
        """
        super().__init__(env, **kwargs)
        assert contact in ("hfield", "boxes", "pool"), f"contact must be 'hfield', 'boxes' or 'pool', got {contact}"
        assert (contact == "pool") == (pool_file is not None), "contact='pool' needs pool_file, and only it uses one"
        self.contact_mode, self.box_quant = contact, float(box_quant)
        self.box_heights = contact in ("boxes", "pool")  # heights are those of the boxes: nearest sample, quantised
        self.pool_file = None if pool_file is None else os.path.expanduser(pool_file)
        self._pool_dev = None         # device copies of the per-trajectory slot centres and half sizes
        self.contact_geoms = None if contact_geoms is None else list(contact_geoms)
        self.ground_clearance = ground_clearance
        self._ground = None   # (geom ids, local sample points, radii, root z qpos index), built on first use
        d = np.load(os.path.expanduser(file))
        self.heights = np.asarray(d["h"], dtype=np.float32)
        self.cell = float(d["cell"])
        self.origin = np.asarray(d["origin"], dtype=np.float64) if "origin" in d.files else np.zeros(2)
        self.nrow, self.ncol = self.heights.shape
        self.hmin, self.hmax = float(self.heights.min()), float(self.heights.max())
        self.base_depth = base_depth
        self._heights_dev = None      # fallback device copy, only made if the MJX model has no hfield_data
        self._tile_adr = None         # (nty, ntx) hfield_data address of every tile, set from the compiled model

        # tiles of nt x nt cells ((nt + 1)^2 samples), the map padded at its far edges to whole tiles
        span = max(self.nrow, self.ncol) - 1
        self.nt = span if tile_size is None else max(1, min(int(round(tile_size / self.cell)), span))
        self.nty, self.ntx = -(-(self.nrow - 1) // self.nt), -(-(self.ncol - 1) // self.nt)
        padded = np.pad(self.heights, ((0, self.nty * self.nt + 1 - self.nrow), (0, self.ntx * self.nt + 1 - self.ncol)),
                        mode="edge")
        self.tile_min = np.zeros((self.nty, self.ntx), np.float32)
        self.tile_range = np.zeros((self.nty, self.ntx), np.float32)

        key = hashlib.sha1(self.heights.tobytes() + np.float64(self.cell).tobytes() + np.int64(self.nt).tobytes())
        self.tile_dir = os.path.join(os.path.expanduser(cache_dir), f"hfield_{key.hexdigest()[:16]}")
        write = not os.path.exists(os.path.join(self.tile_dir, "done"))
        if write:
            os.makedirs(self.tile_dir, exist_ok=True)
        n = self.nt + 1
        for a in range(self.nty):
            for b in range(self.ntx):
                t = padded[a * self.nt:a * self.nt + n, b * self.nt:b * self.nt + n]
                lo, hi = float(t.min()), float(t.max())
                self.tile_min[a, b], self.tile_range[a, b] = lo, max(hi - lo, 1e-3)
                if write:
                    with open(self._tile_file(a, b), "wb") as f:
                        np.array([n, n], dtype=np.int32).tofile(f)
                        ((t - lo) / self.tile_range[a, b]).astype(np.float32).tofile(f)
        if write:
            open(os.path.join(self.tile_dir, "done"), "w").close()

        if self.contact_mode == "pool":
            p = np.load(self.pool_file)
            idx, boxes = np.asarray(p["idx"]), self.boxes()
            assert int(p["n_boxes"]) == len(boxes), (
                f"{self.pool_file} indexes {int(p['n_boxes'])} boxes, this map has {len(boxes)}: it was built for "
                f"another map or box_quant")
            # slot centres and half sizes per trajectory; unused slots parked as small boxes below the ground plane
            ctr = np.zeros(idx.shape + (3,), np.float32)
            half = np.full(idx.shape + (3,), 0.01, np.float32)
            ctr[..., 2] = self.hmin - self.POOL_PARK_DEPTH
            b = boxes[np.maximum(idx, 0)]
            used = idx >= 0
            lo, hi = b[..., 0::2], b[..., 1::2]     # (x0, y0, z0), (x1, y1, z1)
            ctr[used] = ((lo + hi) / 2)[used]
            half[used] = ((hi - lo) / 2)[used]
            self.pool_ctr, self.pool_half = ctr, half
            self.pool_slots = idx.shape[1]
            self._pool_geoms = None   # first slot geom id, set from the compiled model
            self._pool_bvh = None     # MuJoCo: the world body's bounding volume nodes the slots touch
            self._pool_checked = False

    def _tile_file(self, a: int, b: int) -> str:
        return os.path.join(self.tile_dir, f"tile_{a}_{b}.bin")

    def _tile_name(self, a: int, b: int) -> str:
        # with box contact the ground plane is the "floor"
        return "floor" if a == b == 0 and self.contact_mode == "hfield" else f"floor_{a}_{b}"

    def boxes(self) -> np.ndarray:
        """(n, 6) boxes [x0, x1, y0, y1, z0, z1] covering the map above its lowest height, cached next to the tiles.

        Cells of one quantised height form flat regions (treads); a region that fills its bounding rectangle is one
        box, any other is split greedily into rectangles. Cell (i, j) covers +-cell/2 around its sample point."""
        path = os.path.join(self.tile_dir, f"boxes_q{self.box_quant:g}.npy")
        if os.path.exists(path):
            return np.load(path)
        from scipy import ndimage
        q, c, ground = self.box_quant, self.cell, self.hmin
        levels = np.round((self.heights - ground) / q).astype(np.int64)
        out = []
        for v in np.unique(levels):
            if v <= 0:
                continue
            lab, n = ndimage.label(levels == v)
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                m = lab[sl] == k
                rects = [(0, m.shape[0], 0, m.shape[1])] if m.all() else _cover(m)
                for i0, i1, j0, j1 in rects:
                    i0, i1, j0, j1 = i0 + sl[0].start, i1 + sl[0].start, j0 + sl[1].start, j1 + sl[1].start
                    out.append([self.origin[0] + (j0 - 0.5) * c, self.origin[0] + (j1 - 0.5) * c,
                                self.origin[1] + (i0 - 0.5) * c, self.origin[1] + (i1 - 0.5) * c, ground, ground + v * q])
        boxes = np.asarray(out, dtype=np.float64).reshape(-1, 6)
        np.save(path, boxes)
        return boxes

    def write_pool_file(self, path: str, traj: Any, reach: float = 1.0) -> np.ndarray:
        """
        Build the box table of ``contact="pool"`` for a dataset and save it to ``path`` (``.npz``).

        Every trajectory gets the boxes of ``boxes()`` within ``reach`` metres (in xy) of its root path, the xy of
        the trajectory's first joint (the root free joint). As long as the robot stays within ``reach`` of its
        reference path, the pool gives it the contacts of the full box map.

        Args:
            path: Output ``.npz`` (``idx``, ``n_boxes``, ``reach``).
            traj: The dataset (a ``Trajectory``), in the order the env will use.
            reach: Distance from the root path in metres.

        Returns:
            np.ndarray: ``idx``, (n_trajectories, K) indices into ``boxes()``, -1 padded.
        """
        B = self.boxes()
        qpos, sp = np.asarray(traj.data.qpos), np.asarray(traj.data.split_points)
        rows = []
        for k in range(_n_trajectories(sp)):
            p = qpos[sp[k]:sp[k + 1], :2]
            lo, hi = p.min(0) - reach, p.max(0) + reach
            cand = np.where((B[:, 1] >= lo[0]) & (B[:, 0] <= hi[0]) & (B[:, 3] >= lo[1]) & (B[:, 2] <= hi[1]))[0]
            near = np.zeros(len(cand), bool)
            for s in range(0, len(p), 256):  # distance of every path point to every candidate rectangle
                q = p[s:s + 256, None, :]
                dx = np.maximum(np.maximum(B[cand, 0] - q[..., 0], q[..., 0] - B[cand, 1]), 0.0)
                dy = np.maximum(np.maximum(B[cand, 2] - q[..., 1], q[..., 1] - B[cand, 3]), 0.0)
                near |= (np.hypot(dx, dy) <= reach).any(0)
            rows.append(cand[near])
        idx = np.full((len(rows), max([len(r) for r in rows] + [1])), -1, np.int32)
        for k, r in enumerate(rows):
            idx[k, :len(r)] = r
        np.savez(os.path.expanduser(path), idx=idx, n_boxes=len(B), reach=reach)
        return idx

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        # replace the floor, keeping its contact settings
        contact = {}
        for g in spec.geoms:
            if g.name == "floor":
                contact = dict(contype=g.contype, conaffinity=g.conaffinity, condim=g.condim, priority=g.priority,
                               friction=list(g.friction), group=g.group)
                spec.delete(g)
                break
        if self.contact_geoms is not None:
            # a collision bit of its own: only the floor tiles and the listed geoms carry it, so only they collide
            # with each other. Unlike explicit contact pairs this keeps geom friction (and its randomization).
            names = {g.name: g for g in spec.geoms}
            missing = [n for n in self.contact_geoms if n not in names]
            assert not missing, f"contact_geoms not in the model: {missing}"
            taken = [g.name for g in spec.geoms if (g.contype | g.conaffinity) & self.CONTACT_BIT]
            assert not taken, f"collision bit {self.CONTACT_BIT} already used by {taken}"
            for n in self.contact_geoms:
                names[n].contype |= self.CONTACT_BIT
                names[n].conaffinity |= self.CONTACT_BIT
            contact.update(contype=self.CONTACT_BIT, conaffinity=self.CONTACT_BIT)

        half = self.nt * self.cell / 2
        boxes = self.contact_mode in ("boxes", "pool")
        tile_contact = dict(contact, contype=0, conaffinity=0) if boxes else contact
        self.hfield_names = []
        for a in range(self.nty):
            for b in range(self.ntx):
                hname = f"heightmap_terrain_{a}_{b}"
                spec.add_hfield(name=hname, size=[half, half, float(self.tile_range[a, b]), self.base_depth],
                                file=self._tile_file(a, b))
                pos = [self.origin[0] + b * self.nt * self.cell + half, self.origin[1] + a * self.nt * self.cell + half,
                       float(self.tile_min[a, b])]
                spec.worldbody.add_geom(name=self._tile_name(a, b), type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                        hfieldname=hname, pos=pos, material="MatPlane", rgba=(0.8, 0.9, 0.8, 1),
                                        **tile_contact)
                self.hfield_names.append(hname)
        if boxes:
            spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                    pos=[0, 0, self.hmin], material="MatPlane", rgba=(0.8, 0.9, 0.8, 1), **contact)
        if self.contact_mode == "boxes":
            for k, (x0, x1, y0, y1, z0, z1) in enumerate(self.boxes()):
                spec.worldbody.add_geom(name=f"terrain_box_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2],
                                        pos=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                                        rgba=(0.7, 0.75, 0.8, 1), **contact)
        elif self.contact_mode == "pool":
            # the slots start as trajectory 0's boxes; update() writes each env's own
            for k in range(self.pool_slots):
                spec.worldbody.add_geom(name=f"terrain_pool_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        size=self.pool_half[0, k].tolist(), pos=self.pool_ctr[0, k].tolist(),
                                        rgba=(0.7, 0.75, 0.8, 1), **contact)
        hf = [f.name for f in spec.hfields]
        self.hfield_ids = np.array([hf.index(n) for n in self.hfield_names]).reshape(self.nty, self.ntx)
        self.hfield_id = int(self.hfield_ids[0, 0])
        return spec

    def update(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType,
               traj=None) -> Tuple[Any, Any, Any]:
        """With ``contact="pool"``: put the boxes of each env's current trajectory into the box slots, in the model
        (position, size, bounds; for MuJoCo also the world body's bounding volume tree) and, for MJX, in the data
        (MuJoCo Warp computes the poses of world-body geoms only in make_data, and the JAX kinematics refresh after
        a step writes the slots' model-default poses back).

        The trajectory is read from ``carry.traj_state`` every step, so the boxes follow the clip even if it changes
        within an episode. Between a reset and the first step the slots still hold the previous trajectory's boxes
        (the model is not per-env state at reset); heights (``get_height_at``, grounding, height scans) come from the
        full map and are not affected."""
        if self.contact_mode != "pool":
            return model, data, carry
        traj_no = getattr(getattr(carry, "traj_state", None), "traj_no", None)
        if traj_no is None:
            raise ValueError("HeightmapTerrain(contact='pool') selects the boxes by the current trajectory, but the "
                             "env has no trajectory state. Load a trajectory or use contact='boxes'.")
        if not self._pool_checked and traj is not None and not isinstance(traj.data.split_points, jax.core.Tracer):
            n_traj = _n_trajectories(np.asarray(traj.data.split_points))
            if n_traj != len(self.pool_ctr):
                raise ValueError(f"{self.pool_file} has boxes for {len(self.pool_ctr)} trajectories, the dataset has "
                                 f"{n_traj}")
            self._pool_checked = True
        if self._pool_geoms is None:
            m = env._model
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"terrain_pool_{k}") for k in range(self.pool_slots)]
            assert ids == list(range(ids[0], ids[0] + self.pool_slots)), "the pool slot geoms are not contiguous"
            self._pool_geoms = ids[0]
        g = slice(self._pool_geoms, self._pool_geoms + self.pool_slots)
        if backend == np:
            ctr, half = self.pool_ctr[traj_no], self.pool_half[traj_no]
            model.geom_pos[g], model.geom_size[g] = ctr, half
            model.geom_rbound[g] = np.linalg.norm(half, axis=-1)
            model.geom_aabb[g] = np.concatenate((np.zeros_like(half), half), -1)
            # MuJoCo's midphase culls with the world body's bounding volume tree, which the compiler built for the
            # slots' first boxes: the slots' leaves get their new boxes, the nodes above them enclose every box
            if self._pool_bvh is None:
                self._pool_bvh = self._pool_bvh_nodes(model)
            leaves, inner, inner_aabb = self._pool_bvh
            model.bvh_aabb[inner] = inner_aabb
            model.bvh_aabb[leaves] = np.concatenate((ctr, half), -1)
            return model, data, carry
        if self._pool_dev is None:
            with jax.ensure_compile_time_eval():
                self._pool_dev = (jnp.asarray(self.pool_ctr), jnp.asarray(self.pool_half))
        ctr, half = self._pool_dev[0][traj_no], self._pool_dev[1][traj_no]
        aabb = jnp.concatenate((jnp.zeros_like(half), half), -1).reshape(model.geom_aabb[g].shape)
        model = model.replace(geom_pos=model.geom_pos.at[g].set(ctr), geom_size=model.geom_size.at[g].set(half),
                              geom_rbound=model.geom_rbound.at[g].set(jnp.linalg.norm(half, axis=-1)),
                              geom_aabb=model.geom_aabb.at[g].set(aabb))
        data = data.replace(geom_xpos=data.geom_xpos.at[g].set(ctr))
        return model, data, carry

    def _pool_bvh_nodes(self, model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The world body's bounding volume nodes of the pool slots: the slots' leaves (in slot order), the inner
        nodes above them, and boxes for those inner nodes that enclose every box a slot can hold."""
        adr, num = int(model.body_bvhadr[0]), int(model.body_bvhnum[0])
        slots = range(self._pool_geoms, self._pool_geoms + self.pool_slots)
        leaf = {int(model.bvh_nodeid[adr + n]): adr + n for n in range(num) if model.bvh_nodeid[adr + n] >= 0}
        assert all(k in leaf for k in slots), "a pool slot geom has no leaf in the world body's bounding volume tree"
        leaves = np.array([leaf[k] for k in slots])
        parent = {adr + int(c): adr + n for n in range(num) for c in model.bvh_child[adr + n] if c >= 0}
        inner = set()
        for n in leaves:
            while n in parent:
                n = parent[n]
                inner.add(n)
        inner = np.array(sorted(inner), int)
        lo = np.minimum((self.pool_ctr - self.pool_half).reshape(-1, 3).min(0),
                        model.bvh_aabb[inner, :3] - model.bvh_aabb[inner, 3:])
        hi = np.maximum((self.pool_ctr + self.pool_half).reshape(-1, 3).max(0),
                        model.bvh_aabb[inner, :3] + model.bvh_aabb[inner, 3:])
        return leaves, inner, np.concatenate(((lo + hi) / 2, (hi - lo) / 2), -1)

    @property
    def trajectories_in_world_frame(self) -> bool:
        return True

    @property
    def grounds_robot(self) -> bool:
        return self.ground_clearance is not None

    def _ground_setup(self, env: Any):
        """Geoms that collide with the terrain, as local sample points (8 per geom) with a radius each."""
        if self._ground is not None:
            return self._ground
        m = env._model
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.contact_geoms is not None:
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.contact_geoms]
        else:
            ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and
                   ((m.geom_contype[g] & m.geom_conaffinity[floor]) or (m.geom_contype[floor] & m.geom_conaffinity[g]))]
        pts, rad = np.zeros((len(ids), 8, 3)), np.zeros((len(ids), 8))
        for k, g in enumerate(ids):
            s = m.geom_size[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                rad[k] = s[0]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                pts[k, :, 2] = np.linspace(-s[1], s[1], 8)
                rad[k] = s[0]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                pts[k] = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
            else:  # conservative: bounding sphere
                rad[k] = m.geom_rbound[g]
        from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
        root_z = int(np.array(mj_jntname2qposid(env.root_free_joint_xml_name, m))[2])
        self._ground = (np.array(ids), pts, rad, root_z)
        return self._ground

    def ground_robot(self, env: Any, data: Any, backend: ModuleType) -> Any:
        ids, pts, rad, root_z = self._ground_setup(env)
        if backend == np:
            mujoco.mj_kinematics(env._model, data)
            gpos, gmat = data.geom_xpos[ids], data.geom_xmat[ids].reshape(-1, 3, 3)
        else:
            from mujoco import mjx
            if getattr(env, "_use_mjwarp", False):
                kin = mjx.kinematics(env._kin_sys, env._kin_data.replace(qpos=data.qpos))
            else:
                kin = mjx.kinematics(env.sys, data)
            gpos, gmat = kin.geom_xpos[ids], kin.geom_xmat[ids].reshape(-1, 3, 3)
        p = gpos[:, None, :] + backend.einsum("gij,gkj->gki", gmat, backend.asarray(pts))
        gap = p[..., 2] - backend.asarray(rad) - self.get_height_at(p[..., :2], backend, env=env)
        lift = backend.maximum(self.ground_clearance - gap.min(), 0.0)
        if backend == np:
            data.qpos[root_z] += lift
            mujoco.mj_kinematics(env._model, data)
            return data
        return data.replace(qpos=data.qpos.at[root_z].add(lift))

    def get_height_at(self, xy: Union[np.ndarray, jnp.ndarray], backend: ModuleType, env: Any = None,
                      carry: Any = None) -> Union[np.ndarray, jnp.ndarray]:
        """
        Height of the collision surface at world points ``xy`` (..., 2): the heightfield surface, or with box
        contact the top of the box (or ground plane) under the point.

        With the JAX backend and an ``env`` whose MJX model holds ``hfield_data`` as a device array, the heights are
        read from the tiles in that array, the same buffer the simulator collides with. A second copy of a large map
        would be embedded in every jitted program as a constant (and XLA sinks constants into loop bodies, copying
        them again), which costs gigabytes for a few-hundred-MB map.
        """
        fx = (xy[..., 0] - self.origin[0]) / self.cell
        fy = (xy[..., 1] - self.origin[1]) / self.cell
        if backend != jnp:
            return self._on_boxes(hfield_surface_height(self.heights, fx, fy, self.nrow, self.ncol, np,
                                                        nearest=self.box_heights), np)
        hfield_data = getattr(getattr(env, "sys", None), "hfield_data", None)
        if isinstance(hfield_data, jax.Array):
            if self._tile_adr is None:
                self._tile_adr = np.asarray(env._model.hfield_adr)[self.hfield_ids].astype(np.int32)
            fx = jnp.clip(fx, 0.0, self.ncol - 1.0)
            fy = jnp.clip(fy, 0.0, self.nrow - 1.0)
            bx = jnp.clip(jnp.floor(fx / self.nt).astype(int), 0, self.ntx - 1)
            by = jnp.clip(jnp.floor(fy / self.nt).astype(int), 0, self.nty - 1)
            n = self.nt + 1
            h = hfield_surface_height(hfield_data, fx - bx * self.nt, fy - by * self.nt, n, n, jnp,
                                      adr=jnp.asarray(self._tile_adr)[by, bx], nearest=self.box_heights)
            # MuJoCo keeps each tile's elevations normalized to [0, 1]
            return self._on_boxes(jnp.asarray(self.tile_min)[by, bx] + jnp.asarray(self.tile_range)[by, bx] * h, jnp)
        if self._heights_dev is None:
            # concrete even when first called inside a jit trace, so caching it cannot leak a tracer
            with jax.ensure_compile_time_eval():
                self._heights_dev = jnp.asarray(self.heights.reshape(-1))
        return self._on_boxes(hfield_surface_height(self._heights_dev, fx, fy, self.nrow, self.ncol, jnp,
                                                    nearest=self.box_heights), jnp)

    def _on_boxes(self, h, backend):
        """With box contact: the height of the box (or ground plane) surface, i.e. the nearest sample (callers pass
        it) quantised like the boxes. Bilinear heights would put points near a tread edge up to half a riser off."""
        if not self.box_heights:
            return h
        return self.hmin + backend.round((h - self.hmin) / self.box_quant) * self.box_quant


def _cover(mask: np.ndarray) -> List[tuple]:
    """Rectangles (i0, i1, j0, j1) covering a boolean mask: greedy runs along rows, grown down while they fit."""
    m, out = mask.copy(), []
    for i in range(m.shape[0]):
        j = 0
        while j < m.shape[1]:
            if not m[i, j]:
                j += 1
                continue
            w = 1
            while j + w < m.shape[1] and m[i, j + w]:
                w += 1
            h = 1
            while i + h < m.shape[0] and m[i + h, j:j + w].all():
                h += 1
            m[i:i + h, j:j + w] = False
            out.append((i, i + h, j, j + w))
            j += w
    return out


def _n_trajectories(split_points: np.ndarray) -> int:
    """Number of trajectories in a dataset whose split points may be padded with -1 (TrajectoryHandler)."""
    pad = np.flatnonzero(split_points == -1)
    return int(pad[0] if len(pad) else len(split_points)) - 1
