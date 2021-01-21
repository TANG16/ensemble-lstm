import numpy as np
from config import mydata, trainset, testset, goals, EVAL_TIME
import csv
import os
import pickle
import matplotlib.pyplot as plt


def evaluate(good_predictons, total_examples, prediction, truth, eval_time):
    int_prediction = [1 if truth[x] else 0 for x in prediction]
    int_prediction.reverse()
    for i in range(min(eval_time, len(int_prediction))):
        good_predictons[0, i] += int_prediction[i]
        total_examples[0, i] += 1
    return good_predictons, total_examples


def evaluate_baseline(baseline, testset):
    good_predictons = np.zeros((1, EVAL_TIME))
    total_examples = np.zeros((1, EVAL_TIME))
    segments = os.listdir(f"../data/processed/{testset[0]}")
    for dataset in testset:
        segments = os.listdir(f"../data/processed/{dataset}")
        for segment in segments:
            baseline_list = []
            truth = []
            for goal_id in range(len(goals)):
                goal = goals[goal_id]
                with open(
                    f"../data/processed/{dataset}/{segment}/{goal}.pkl", "rb"
                ) as f:
                    data = pickle.load(f)
                    baseline_list.append(data[baseline])
                    truth.append(data["label"])

            baseline_array = np.array(baseline_list)
            prediction = np.argmin(baseline_array, axis=0)
            good_predictons, total_examples = evaluate(
                good_predictons, total_examples, prediction, truth, EVAL_TIME
            )
    good_predictons = np.flip(good_predictons)
    total_examples = np.flip(total_examples)
    print(f"{baseline} AUC = {sum(good_predictons[0, :] / total_examples[0, :])}")
    plt.plot(range(EVAL_TIME), good_predictons[0, :] / total_examples[0, :])
    plt.title(f"{baseline}")
    plt.show()
