import os


def video2gif(file_path, fps=20, scale=720, skip=0.1, duration=4.0, rectangular=True):
    gif_path = ".".join(file_path.split(".")[:-1]) + ".gif"
    if rectangular:
        crop_cmd = ",crop=1080:1080"
    else:
        crop_cmd = ""
    cmd = f"ffmpeg -ss {skip} -t {duration} -i {file_path} -filter_complex \"fps={fps}{crop_cmd}," \
          f"scale={scale}:-1[s]; [s]split[a][b]; " \
          f"[a]palettegen[palette]; [b][palette]paletteuse\" -y {gif_path}"

    os.system(cmd)
