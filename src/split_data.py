from config import mykeys, mydata, testset, trainset, goals
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import csv

num_segments = 0
os.system("mkdir ../data/processed/")
for dataset in mydata:
    print(f"Processing: {dataset}")

    os.system(f"mkdir ../data/processed/{dataset}")
    with open(f"../data/{dataset}/objects.pkl", "rb") as f:
        objects = pickle.load(f)
    with open(f"../data/{dataset}/joints_orientations.pkl", "rb") as f:
        orientations = pickle.load(f)
    with open(f"../data/{dataset}/joints_positions.pkl", "rb") as f:
        positions = pickle.load(f)
    with open(f"../data/{dataset}/segmentation.pkl", "rb") as f:
        segmentation = pickle.load(f)
    with open(f"../data/{dataset}/gaze.pkl", "rb") as f:
        gaze = pickle.load(f)

    i = 0
    while i < len(segmentation) - 1:
        os.system(f"mkdir ../data/processed/{dataset}/{str(num_segments)}")
        if dataset == "6_1" and i == 248:  # error in the dataset, null twice in a row
            i += 1
        if dataset == "7_2" and i == 0:  # error in the dataset, null twice in a row
            i += 1
        current_segment = segmentation[i]
        next_segment = segmentation[i + 1]
        curr_goal = next_segment[2].decode("utf-8")

        if curr_goal == "null":
            print(i, current_segment, next_segment)
            print("Warning: goal should never be null")
        num_examples = 0
        for goal in goals:
            savedict = {}
            for key in mykeys:
                euc_distance = (
                    objects[goal][current_segment[0] : current_segment[1], 0:3]
                    - positions[key][current_segment[0] : current_segment[1]]
                )
                orientation_distance = orientations[key][
                    current_segment[0] : current_segment[1]
                ]
                orientation_distance = (
                    orientation_distance
                    / np.linalg.norm(orientation_distance, axis=1)[:, None]
                )
                orientation_distance = np.linalg.norm(
                    orientation_distance
                    - euc_distance / np.linalg.norm(euc_distance, axis=1)[:, None],
                    axis=1,
                )
                euc_distance = np.linalg.norm(euc_distance, axis=1)
                savedict[key + " euc"] = euc_distance
                savedict[key + " ori"] = orientation_distance

            goggles_distance = (
                objects[goal][current_segment[0] : current_segment[1], 0:3]
                - objects["goggles"][current_segment[0] : current_segment[1], 0:3]
            )
            goggles_distance = (
                goggles_distance / np.linalg.norm(goggles_distance, axis=1)[:, None]
            )
            gaze_distance = gaze[current_segment[0] : current_segment[1]]
            gaze_distance = np.linalg.norm(gaze_distance - goggles_distance, axis=1)
            savedict["gaze"] = gaze_distance
            savedict["label"] = goal == curr_goal
            with open(
                f"../data/processed/{dataset}/{str(num_segments)}/{goal}.pkl", "wb"
            ) as f:
                pickle.dump(savedict, f)
            num_examples += 1

        i += 2
        num_segments += 1

print("Processed data")
