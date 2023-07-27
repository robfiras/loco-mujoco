import os
import shutil
import numpy as np

N_SAMPLES = 500


for env_type in os.listdir():
    if env_type in ["humanoids", "quadrupeds"]:
        env_type_path = "./" + env_type
        mini_dataset_path = env_type_path + "/mini_datasets"
        if os.path.isdir(mini_dataset_path):
            shutil.rmtree(mini_dataset_path)
        os.mkdir(mini_dataset_path)

        files = os.listdir(env_type_path)
        for file in files:
            if file.endswith("npz"):
                data = np.load(env_type_path + "/" + file, allow_pickle=True)

                # modify split points
                if "split_points" in data.keys():
                    # if we have multiple trajectories, take from each N_SAMPLES
                    split_points = data["split_points"]

                    assert split_points[1] >= N_SAMPLES
                    short_data = {k: d[:N_SAMPLES] for k, d in data.items() if k != "split_points"}
                    created_split_points = False
                    new_split_points = [0]
                    for i in range(1, len(split_points)-1):
                        for k, v in short_data.items():
                            short_data[k] = np.concatenate([short_data[k], data[k][split_points[i]:split_points[i]+N_SAMPLES]])
                        new_split_points.append(new_split_points[-1] + N_SAMPLES)
                        created_split_points = True
                    new_split_points.append(new_split_points[-1] + N_SAMPLES)
                    short_data["split_points"] = np.array(new_split_points)
                else:
                    short_data = {k: d[:N_SAMPLES] for k, d in data.items() if k != "split_points"}

                # save new files
                new_path = mini_dataset_path + "/" + file
                np.savez(new_path, **short_data)
