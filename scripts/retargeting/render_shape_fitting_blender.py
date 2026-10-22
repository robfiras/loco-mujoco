"""Blender-side renderer for the Stage 1 shape-fitting comparison."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-usd", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--camera-scale", type=float, default=2.35)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _material(name, color, roughness=0.75, alpha=1.0, emission=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )
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


def _look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _configure_scene(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    if hasattr(scene, "cycles"):
        scene.cycles.samples = args.samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Khronos PBR Neutral"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.render.film_transparent = False

    if scene.world is None:
        scene.world = bpy.data.worlds.new("ShapeFittingWorld")
    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.92, 0.94, 0.97, 1.0)
    background.inputs["Strength"].default_value = 0.5

    for obj in [obj for obj in scene.objects if obj.type in {"LIGHT", "CAMERA"}]:
        bpy.data.objects.remove(obj, do_unlink=True)
    for name, energy, location, size in (
        ("Key", 600.0, (4.0, -5.0, 7.0), 5.0),
        ("Fill", 380.0, (-4.0, -2.0, 5.0), 4.0),
        ("Rim", 450.0, (2.0, 4.0, 6.0), 3.0),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = location
        _look_at(light, (0.0, 0.0, 2.0))

    camera_data = bpy.data.cameras.new("ShapeFittingCamera")
    camera = bpy.data.objects.new("ShapeFittingCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (4.2, -6.5, 2.05)
    _look_at(camera, (0.0, 0.0, 0.85))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = args.camera_scale
    camera.data.lens = 55
    scene.camera = camera


def _style_robot():
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        if "floor" in obj.name.lower():
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        for material in obj.data.materials:
            if material is None or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Metallic"].default_value = 0.0
                    node.inputs["Roughness"].default_value = 0.72


def _place_robot_on_floor():
    """Translate the complete imported hierarchy until its visual feet touch z=0."""
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    min_z = min(
        (obj.matrix_world @ Vector(corner)).z
        for obj in meshes
        for corner in obj.bound_box
    )
    imported = set(bpy.context.scene.objects)
    for obj in imported:
        if obj.parent is None and obj.type not in {"CAMERA", "LIGHT"}:
            obj.location.z -= min_z
    return -min_z


def _add_floor():
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, -0.006))
    floor = bpy.context.object
    floor.name = "ShapeFittingFloor"
    floor.data.materials.append(_material("Floor", (0.72, 0.75, 0.79), roughness=0.92))


def _sphere(name, position, radius, material, collection):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)


def _segment(name, start, end, radius, material, collection):
    start, end = Vector(start), Vector(end)
    delta = end - start
    if delta.length < 1e-7:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=delta.length)
    obj = bpy.context.object
    obj.name = name
    obj.location = (start + end) / 2
    obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    for parent in list(obj.users_collection):
        parent.objects.unlink(obj)
    collection.objects.link(obj)


def _add_overlay(vertices, faces, smpl_joints, target_joints, z_offset):
    collection = bpy.data.collections.new("ShapeFittingOverlay")
    bpy.context.scene.collection.children.link(collection)
    offset = np.asarray([0.0, 0.0, z_offset])
    vertices = vertices + offset
    smpl_joints = smpl_joints + offset
    target_joints = target_joints + offset
    smpl_material = _material("SMPLBody", (0.08, 0.42, 0.82), alpha=0.20)
    smpl_joint_material = _material("SMPLJoints", (0.02, 0.32, 0.80), emission=0.15)
    robot_joint_material = _material("RobotSites", (1.0, 0.34, 0.035), emission=0.1)
    error_material = _material("CorrespondenceError", (0.82, 0.20, 0.12), alpha=0.85)

    mesh = bpy.data.meshes.new("SMPLBodyMesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    body = bpy.data.objects.new("SMPLBody", mesh)
    collection.objects.link(body)
    body.data.materials.append(smpl_material)
    for polygon in body.data.polygons:
        polygon.use_smooth = True

    for index, (smpl, target) in enumerate(zip(smpl_joints, target_joints)):
        _segment(f"error_{index}", smpl, target, 0.0045, error_material, collection)
        _sphere(f"smpl_joint_{index}", smpl, 0.021, smpl_joint_material, collection)
        _sphere(f"robot_site_{index}", target, 0.025, robot_joint_material, collection)
    return collection


def _remove_collection(collection):
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def main():
    args = _args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(Path(args.robot_usd).resolve()))
    _style_robot()
    z_offset = _place_robot_on_floor()
    _add_floor()
    _configure_scene(args)
    data = np.load(args.data)

    for state in ("before", "after"):
        overlay = _add_overlay(
            data[f"{state}_vertices"],
            data["faces"],
            data[f"{state}_joints"],
            data["target_joints"],
            z_offset,
        )
        bpy.context.scene.render.filepath = str(output / f"{state}.png")
        bpy.ops.render.render(write_still=True)
        _remove_collection(overlay)
    (output / ".shape_fitting_render_complete").touch()


if __name__ == "__main__":
    main()
