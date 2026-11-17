"""Blender-side renderer for the Stage 2/3 retargeting figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--target-axes", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material(name, color, roughness=0.72, alpha=1.0, emission=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Alpha"].default_value = alpha
    if emission:
        emission_color = principled.inputs.get("Emission Color") or principled.inputs.get("Emission")
        if emission_color is not None:
            emission_color.default_value = (*color, 1.0)
        emission_strength = principled.inputs.get("Emission Strength")
        if emission_strength is not None:
            emission_strength.default_value = emission
    if alpha < 1.0:
        try:
            material.surface_render_method = "DITHERED"
        except (AttributeError, TypeError):
            pass
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
    return material


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)


def _setup_scene(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = args.exposure
    scene.render.film_transparent = False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("RetargetingWorld")
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.92, 0.94, 0.97, 1.0)
    background.inputs["Strength"].default_value = 0.5

    for name, energy, location, size in (
        ("Key", 600.0, (4.0, -5.0, 5.0), 5.0),
        ("Fill", 380.0, (-4.0, -2.0, 3.0), 4.0),
        ("Rim", 450.0, (2.0, 4.0, 4.0), 3.0),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        _look_at(light, (0.0, 0.0, 0.85))

    camera_data = bpy.data.cameras.new("RetargetingCamera")
    camera = bpy.data.objects.new("RetargetingCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (4.2, -6.5, 2.05)
    _look_at(camera, (0.0, 0.0, 0.85))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.35
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, -0.006))
    floor = bpy.context.object
    floor.name = "RetargetingFloor"
    floor.data.materials.append(_material("FloorMaterial", (0.72, 0.75, 0.79), roughness=0.92))
    return floor


def _style_robot(objects):
    for obj in objects:
        if obj.type != "MESH":
            continue
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Metallic"].default_value = 0.0
                    node.inputs["Roughness"].default_value = 0.72


def _import_robot(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.usd_import(filepath=str(Path(path).resolve()))
    objects = list(set(bpy.context.scene.objects) - before)
    for obj in list(objects):
        if obj.type in {"LIGHT", "CAMERA"} or (obj.type == "MESH" and "floor" in obj.name.lower()):
            objects.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
    _style_robot(objects)
    return objects


def _shift_groups_to_floor(groups):
    current_meshes = [obj for obj in groups[1] if obj.type == "MESH"]
    min_z = min(
        (obj.matrix_world @ Vector(corner)).z
        for obj in current_meshes
        for corner in obj.bound_box
    )
    for objects in groups:
        object_set = set(objects)
        for obj in objects:
            if obj.parent not in object_set:
                obj.location.z -= min_z


def _mesh_object(name, vertices, faces, material):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _sphere(name, position, radius, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def _segment(name, start, end, radius, material):
    start, end = Vector(start), Vector(end)
    delta = end - start
    if delta.length < 1e-7:
        return None
    bpy.ops.mesh.primitive_cylinder_add(vertices=14, radius=radius, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (start + end) / 2
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def _set_visible(objects, visible):
    for obj in objects:
        if obj.type == "MESH":
            obj.hide_render = not visible


def _render(path, transparent):
    scene = bpy.context.scene
    scene.render.film_transparent = transparent
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _render_layered(output, name, groups, floor):
    for group in groups:
        _set_visible(group, False)
    _set_visible(groups[1], True)
    floor.hide_render = False
    _render(output / f"_{name}_base.png", transparent=False)

    floor.hide_render = True
    for suffix, group in (("past", groups[0]), ("future", groups[2])):
        for candidate in groups:
            _set_visible(candidate, False)
        _set_visible(group, True)
        _render(output / f"_{name}_{suffix}.png", transparent=True)
    for candidate in groups:
        _set_visible(candidate, False)
    _set_visible(groups[1], True)
    _render(output / f"_{name}_current.png", transparent=True)


def _robot_panel(args, output, name, paths):
    _clear_scene()
    floor = _setup_scene(args)
    groups = [_import_robot(path) for path in paths]
    _shift_groups_to_floor(groups)
    _render_layered(output, name, groups, floor)


def _smpl_panel(args, output, name, vertices, faces):
    _clear_scene()
    floor = _setup_scene(args)
    blue = _material("SMPLBlue", (0.08, 0.42, 0.82), roughness=0.78)
    groups = [[_mesh_object(f"{name}_{index}", state, faces, blue)]
              for index, state in enumerate(vertices)]
    _render_layered(output, name, groups, floor)


def _skeleton_objects(name, joints, parents, material):
    objects = []
    for index, position in enumerate(joints):
        radius = 0.055 if index == 15 else 0.026
        objects.append(_sphere(f"{name}_joint_{index}", position, radius, material))
        parent = int(parents[index])
        if index > 0 and 0 <= parent < len(joints):
            bone = _segment(
                f"{name}_bone_{parent}_{index}", joints[parent], position,
                0.018, material,
            )
            if bone is not None:
                objects.append(bone)
    return objects


def _skeleton_panel(args, output, name, joint_sequence, parents):
    _clear_scene()
    floor = _setup_scene(args)
    blue = _material("SMPLSkeletonBlue", (0.04, 0.36, 0.86), roughness=0.72)
    groups = [
        _skeleton_objects(f"{name}_{index}", joints, parents, blue)
        for index, joints in enumerate(joint_sequence)
    ]
    _render_layered(output, name, groups, floor)


def _target_panel(args, output, data):
    _clear_scene()
    floor = _setup_scene(args)
    smpl_material = _material("SMPLSkeleton", (0.04, 0.36, 0.86), emission=0.05)
    target_material = _material("TargetFrame", (1.0, 0.34, 0.035), emission=0.1)
    error_material = _material("Offset", (0.82, 0.20, 0.12))
    axis_materials = (
        _material("AxisX", (0.88, 0.12, 0.12), emission=0.2),
        _material("AxisY", (0.12, 0.72, 0.22), emission=0.2),
        _material("AxisZ", (0.10, 0.32, 0.95), emission=0.2),
    )
    groups = [
        _skeleton_objects(
            f"TargetSkeleton_{index}", joints,
            data["skeleton_parents"], smpl_material,
        )
        for index, joints in enumerate(data["target_skeleton_joints"])
    ]
    target_overlays = []
    for index, (parent, child) in enumerate(data["target_frame_edges"]):
        target_overlays.append(_segment(
            f"target_bone_{index}", data["target_frame_pos"][parent],
            data["target_frame_pos"][child], 0.012, target_material,
        ))
    for index, (joint, target) in enumerate(zip(data["target_smpl_joints"], data["target_frame_pos"])):
        target_overlays.append(_segment(f"offset_{index}", joint, target, 0.004, error_material))
        target_overlays.append(_sphere(f"target_{index}", target, 0.026, target_material))
    if args.target_axes:
        for index in data["axis_indices"]:
            position = data["target_frame_pos"][index]
            rotation = data["target_frame_rot"][index]
            for axis in range(3):
                target_overlays.append(_segment(
                    f"frame_{index}_{axis}", position,
                    position + rotation[:, axis] * 0.09,
                    0.006, axis_materials[axis],
                ))
    groups[1].extend(obj for obj in target_overlays if obj is not None)
    _render_layered(output, "stage2_targets", groups, floor)


def main():
    args = _parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    source_paths = [str(path) for path in data["source_usd"]]
    target_paths = [str(path) for path in data["target_usd"]]
    _smpl_panel(args, output, "stage2_smpl", data["original_smpl_vertices"], data["faces"])
    _target_panel(args, output, data)
    _robot_panel(args, output, "stage2_robot", target_paths)
    _robot_panel(args, output, "stage3_source", source_paths)
    _skeleton_panel(
        args, output, "stage3_smpl", data["source_skeleton_joints"],
        data["skeleton_parents"],
    )
    _robot_panel(args, output, "stage3_target", target_paths)
    (output / ".retargeting_stages_complete").touch()


if __name__ == "__main__":
    main()
