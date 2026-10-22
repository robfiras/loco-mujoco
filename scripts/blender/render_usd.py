"""Standalone Blender-side USD import and render script.

This file is executed by Blender, not by the normal LocoMuJoCo interpreter.
Arguments after ``--`` are parsed to avoid consuming Blender's own flags.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--lighting", choices=("uniform", "studio", "model"), default="uniform")
    parser.add_argument("--sun-energy", type=float, default=5.0)
    parser.add_argument("--fill-ratio", type=float, default=0.55)
    parser.add_argument("--robot-roughness", type=float, default=0.9)
    parser.add_argument("--robot-specular", type=float, default=0.05)
    parser.add_argument("--floor-style", choices=("viser", "model"), default="viser")
    parser.add_argument("--floor-cell-size", type=float, default=1.0)
    parser.add_argument("--floor-extent", type=float, default=10000.0)
    parser.add_argument("--floor-roughness", type=float, default=0.35)
    parser.add_argument("--sky-style", choices=("daylight", "model"), default="daylight")
    parser.add_argument("--sky-saturation", type=float, default=1.35)
    parser.add_argument("--environment-strength", type=float, default=0.0)
    parser.add_argument("--view-transform", default="Khronos PBR Neutral")
    parser.add_argument("--exposure", type=float, default=0.7)
    parser.add_argument("--render-frame", type=int, default=-1)
    parser.add_argument("--overlay-interval-frames", type=int, default=0)
    parser.add_argument("--overlay-opacity", type=float, default=0.3)
    parser.add_argument("--hide-geom", action="append", default=[])
    parser.add_argument("--camera-position", nargs=3, type=float, required=True)
    parser.add_argument("--camera-look-at", nargs=3, type=float, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _configure_camera(position, look_at):
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    if cameras:
        camera = cameras[0]
    else:
        camera_data = bpy.data.cameras.new("LocoMuJoCoCamera")
        camera = bpy.data.objects.new("LocoMuJoCoCamera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector(position)
    target = bpy.data.objects.get("LocoMuJoCoCameraTarget")
    if target is None:
        camera.rotation_euler = (
            Vector(look_at) - camera.location
        ).to_track_quat("-Z", "Y").to_euler()
    else:
        # Follow horizontal root translation without copying the gait's small
        # vertical bob, and continuously point toward the animated root.
        follow = camera.constraints.new(type="COPY_LOCATION")
        follow.name = "Follow robot"
        follow.target = target
        follow.use_offset = True
        follow.use_x = True
        follow.use_y = True
        follow.use_z = False
        track = camera.constraints.new(type="TRACK_TO")
        track.name = "Look at robot"
        # Follow root XY while keeping the aim height fixed. Tracking the
        # animated root Z made the horizon move with each gait's vertical bob.
        bpy.context.scene.frame_set(0)
        aim = bpy.data.objects.new("LocoMuJoCoCameraAim", None)
        aim.location = Vector((target.matrix_world.translation.x,
                               target.matrix_world.translation.y,
                               look_at[2]))
        bpy.context.scene.collection.objects.link(aim)
        aim_follow = aim.constraints.new(type="COPY_LOCATION")
        aim_follow.target = target
        aim_follow.use_x = True
        aim_follow.use_y = True
        aim_follow.use_z = False
        track.target = aim
        track.track_axis = "TRACK_NEGATIVE_Z"
        track.up_axis = "UP_Y"
    camera.data.lens = 35
    camera.data.clip_end = 1000
    bpy.context.scene.camera = camera
    return target


def _configure_uniform_lighting(sun_energy, fill_ratio):
    """Replace local USD lights with a soft directional key and fill."""
    scene = bpy.context.scene
    for light in [obj for obj in scene.objects if obj.type == "LIGHT"]:
        bpy.data.objects.remove(light, do_unlink=True)

    # Sun lights have no distance falloff, so the robot and floor receive the
    # same illumination throughout the trajectory. A dim opposing fill avoids
    # crushed shadows without the flat gray wash caused by world illumination.
    lights = (
        ("LocoMuJoCoKey", sun_energy, (25.0, -20.0, -25.0), 18.0),
        ("LocoMuJoCoFill", sun_energy * fill_ratio, (50.0, 25.0, 145.0), 25.0),
    )
    for name, energy, rotation, angle in lights:
        light_data = bpy.data.lights.new(name, type="SUN")
        light_data.energy = energy
        light_data.angle = math.radians(angle)
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.rotation_euler = tuple(math.radians(value) for value in rotation)


def _configure_studio_lighting(target):
    """Use the soft key/fill/rim rig from the shape-fitting figure."""
    scene = bpy.context.scene
    for light in [obj for obj in scene.objects if obj.type == "LIGHT"]:
        bpy.data.objects.remove(light, do_unlink=True)

    target_position = target.matrix_world.translation if target is not None else Vector((0, 0, 1))
    lights = (
        ("LocoMuJoCoStudioKey", 600.0, (4.0, -5.0, 5.0), 5.0),
        ("LocoMuJoCoStudioFill", 380.0, (-4.0, -2.0, 3.0), 4.0),
        ("LocoMuJoCoStudioRim", 450.0, (2.0, 4.0, 4.0), 3.0),
    )
    for name, energy, offset, size in lights:
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = target_position + Vector(offset)
        if target is not None:
            follow = light.constraints.new(type="COPY_LOCATION")
            follow.target = target
            follow.use_offset = True
            track = light.constraints.new(type="TRACK_TO")
            track.target = target
            track.track_axis = "TRACK_NEGATIVE_Z"
            track.up_axis = "UP_Y"
        else:
            light.rotation_euler = (
                target_position - light.location
            ).to_track_quat("-Z", "Y").to_euler()


def _srgb(rgb):
    """Convert an 8-bit sRGB color to Blender's scene-linear representation."""
    result = []
    for channel in rgb:
        value = channel / 255.0
        result.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return (*result, 1.0)


def _configure_robot_materials(roughness, specular):
    """Use matte dielectric shading while preserving exported base colors."""
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            node.inputs["Metallic"].default_value = 0.0
            node.inputs["Roughness"].default_value = roughness
            specular_input = node.inputs.get("Specular IOR Level") or node.inputs.get("Specular")
            if specular_input is not None:
                specular_input.default_value = specular


def _configure_viser_floor(cell_size, extent, roughness):
    """Give the exported ground plane viser's charcoal grid appearance."""
    floors = [
        obj for obj in bpy.context.scene.objects
        if obj.type == "MESH" and "floor" in obj.name.lower()
    ]
    for floor in floors:
        # MuJoCo's exported plane is finite. Enlarge its mesh rather than its
        # object scale so procedural coordinates remain measured in metres.
        xs = [vertex.co.x for vertex in floor.data.vertices]
        ys = [vertex.co.y for vertex in floor.data.vertices]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        for vertex in floor.data.vertices:
            if width > 0:
                vertex.co.x *= extent / width
            if height > 0:
                vertex.co.y *= extent / height
        floor.data.update()

        material = bpy.data.materials.new(f"{floor.name}_ViserMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputMaterial")
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = roughness
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])

        coordinates = nodes.new("ShaderNodeTexCoord")

        def line_mask(direction, scale, threshold_value):
            wave = nodes.new("ShaderNodeTexWave")
            wave.wave_type = "BANDS"
            wave.wave_profile = "SAW"
            wave.bands_direction = direction
            wave.inputs["Scale"].default_value = scale
            threshold = nodes.new("ShaderNodeMath")
            threshold.operation = "LESS_THAN"
            threshold.inputs[1].default_value = threshold_value
            links.new(coordinates.outputs["Object"], wave.inputs["Vector"])
            links.new(wave.outputs["Fac"], threshold.inputs[0])
            return threshold.outputs[0]

        def combine_grid(scale, threshold):
            combine = nodes.new("ShaderNodeMath")
            combine.operation = "MAXIMUM"
            links.new(line_mask("X", scale, threshold), combine.inputs[0])
            links.new(line_mask("Y", scale, threshold), combine.inputs[1])
            return combine.outputs[0]

        # Stronger section lines are placed every five cells.
        fine_grid = combine_grid(1.0 / cell_size, 0.035)
        section_grid = combine_grid(1.0 / (cell_size * 5.0), 0.007)
        fine_mix = nodes.new("ShaderNodeMixRGB")
        fine_mix.inputs[1].default_value = _srgb((28, 30, 34))
        fine_mix.inputs[2].default_value = _srgb((68, 70, 74))
        links.new(fine_grid, fine_mix.inputs[0])
        section_mix = nodes.new("ShaderNodeMixRGB")
        section_mix.inputs[2].default_value = _srgb((100, 102, 105))
        links.new(fine_mix.outputs[0], section_mix.inputs[1])
        links.new(section_grid, section_mix.inputs[0])
        links.new(section_mix.outputs[0], shader.inputs["Base Color"])

        floor.data.materials.clear()
        floor.data.materials.append(material)


def _configure_daylight_sky(saturation, environment_strength=0.0):
    """Create a clear physical daylight sky visible only to the camera."""
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("LocoMuJoCoWorld")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    ray_type = nodes.new("ShaderNodeLightPath")
    mix_shader = nodes.new("ShaderNodeMixShader")
    dark_background = nodes.new("ShaderNodeBackground")
    dark_background.inputs["Color"].default_value = (0.92, 0.94, 0.97, 1.0)
    dark_background.inputs["Strength"].default_value = environment_strength
    visible_background = nodes.new("ShaderNodeBackground")
    visible_background.inputs["Strength"].default_value = 0.5

    sky = nodes.new("ShaderNodeTexSky")
    # Blender 5 renamed Nishita's multi-scattering mode.
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation = math.radians(55.0)
    sky.sun_rotation = math.radians(135.0)
    sky.altitude = 0.2
    sky.air_density = 1.0
    if hasattr(sky, "aerosol_density"):
        sky.aerosol_density = 0.0
    else:
        sky.dust_density = 0.0
    sky.ozone_density = 1.0

    sky_color = nodes.new("ShaderNodeHueSaturation")
    sky_color.inputs["Saturation"].default_value = saturation
    links.new(sky.outputs["Color"], sky_color.inputs["Color"])
    links.new(sky_color.outputs["Color"], visible_background.inputs["Color"])
    links.new(ray_type.outputs["Is Camera Ray"], mix_shader.inputs[0])
    links.new(dark_background.outputs[0], mix_shader.inputs[1])
    links.new(visible_background.outputs[0], mix_shader.inputs[2])
    links.new(mix_shader.outputs[0], output.inputs["Surface"])


def _render_trajectory_layers(output, current_frame, interval_frames):
    """Render opaque pose layers for later whole-silhouette compositing."""
    scene = bpy.context.scene
    camera = scene.camera
    scene.frame_set(current_frame)
    bpy.context.view_layer.update()
    camera_matrix = camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
    for constraint in list(camera.constraints):
        camera.constraints.remove(constraint)
    camera.parent = None
    camera.matrix_world = camera_matrix

    floors = [
        obj for obj in scene.objects if obj.type == "MESH" and "floor" in obj.name.lower()
    ]
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(output / "_overlay_base.png")
    bpy.ops.render.render(write_still=True)
    scene.render.film_transparent = True
    for floor in floors:
        if hasattr(floor, "visible_camera"):
            # Keep the floor in indirect lighting/reflections while removing
            # it from the transparent pose pass itself.
            floor.visible_camera = False
        else:
            floor.hide_render = True
    ghost_frames = list(range(0, current_frame, interval_frames))
    ghost_frames += list(range(current_frame + interval_frames, scene.frame_end + 1,
                               interval_frames))
    for frame in ghost_frames:
        scene.frame_set(frame)
        scene.render.filepath = str(output / f"_overlay_ghost_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    scene.frame_set(current_frame)
    scene.render.filepath = str(output / "_overlay_current.png")
    bpy.ops.render.render(write_still=True)


def _hide_geoms(names):
    """Hide imported mesh objects corresponding to selected MuJoCo geoms."""
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and any(name in obj.name for name in names):
            obj.hide_render = True


def main():
    args = _parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(Path(args.usd).resolve()))
    _hide_geoms(args.hide_geom)

    scene = bpy.context.scene
    try:
        scene.render.engine = args.engine
    except TypeError:
        # Blender < 4.2 used the older identifier.
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    if hasattr(scene, "cycles"):
        scene.cycles.samples = args.samples

    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = round(args.fps)
    scene.render.fps_base = round(args.fps) / args.fps
    scene.view_settings.view_transform = args.view_transform
    scene.view_settings.exposure = args.exposure
    scene.frame_start = 0
    scene.frame_end = max(0, args.frames - 1)

    _configure_robot_materials(args.robot_roughness, args.robot_specular)
    if args.floor_style == "viser":
        _configure_viser_floor(args.floor_cell_size, args.floor_extent, args.floor_roughness)
    if args.sky_style == "daylight":
        _configure_daylight_sky(args.sky_saturation, args.environment_strength)
    target = _configure_camera(args.camera_position, args.camera_look_at)
    if args.lighting == "uniform":
        _configure_uniform_lighting(args.sun_energy, args.fill_ratio)
    elif args.lighting == "studio":
        _configure_studio_lighting(target)
    if args.render_frame >= 0:
        scene.frame_start = args.render_frame
        scene.frame_end = args.render_frame

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output / "frame_")
    if args.render_frame >= 0 and args.overlay_interval_frames > 0:
        # Restore the full authored range while selecting ghost sample frames.
        scene.frame_end = max(0, args.frames - 1)
        _render_trajectory_layers(output, args.render_frame, args.overlay_interval_frames)
    else:
        bpy.ops.render.render(animation=True)
    (output / ".blender_render_complete").touch()


if __name__ == "__main__":
    main()
