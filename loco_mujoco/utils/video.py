"""
Video conversion helpers.
"""

import os


def video2gif(file_path, fps=20, scale=720, skip=0.1, duration=4.0, rectangular=True):
    """
    Convert a video file to a GIF with ``ffmpeg``.

    The GIF is written next to the video, with the same file name and the extension ``.gif``; an existing file is
    overwritten. ``ffmpeg`` has to be available on the system path. A color palette is generated from the video to
    improve the quality of the GIF.

    Args:
        file_path (str): Path to the video file.
        fps (int): Frame rate of the GIF.
        scale (int): Width of the GIF in pixels; the height follows from the aspect ratio.
        skip (float): Time in seconds to skip at the beginning of the video.
        duration (float): Duration in seconds of the part of the video that is converted.
        rectangular (bool): If True, the frames are center-cropped to 720x720 pixels before scaling.

    """
    gif_path = ".".join(file_path.split(".")[:-1]) + ".gif"
    if rectangular:
        crop_cmd = ",crop=720:720"
    else:
        crop_cmd = ""
    cmd = f"ffmpeg -ss {skip} -t {duration} -i {file_path} -filter_complex \"fps={fps}{crop_cmd}," \
          f"scale={scale}:-1[s]; [s]split[a][b]; " \
          f"[a]palettegen[palette]; [b][palette]paletteuse\" -y {gif_path}"

    os.system(cmd)
