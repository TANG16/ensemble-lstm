import numpy as np
from config import (
    mykeys,
    mydata,
    trainset,
    testset,
    goals,
    EVAL_TIME,
    EVAL_STEP,
    colors,
    myfeatures,
    my_mean,
    calculate_diff,
    diff_mean,
    CULL_COEFF,
)
import csv
import os
import pickle
import torch
import matplotlib.pyplot as plt
import time


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def save_pickle(data, myfile):
    with open(myfile, "wb") as f:
        pickle.dump(data, f)


def load_pickle(myfile):
    with open(myfile, "rb") as f:
        return pickle.load(f)


def pad_start(batch):
    maxlen = len((max(batch, key=lambda x: len(x[0])))[0])
    for i in range(len(batch)):
        arr = np.pad(batch[i][0], ((maxlen - len(batch[i][0]), 0), (0, 0)), "edge")
        batch[i] = (arr, batch[i][1])
    return torch.utils.data._utils.collate.default_collate(batch)


def cull_trainset(x_train, y_train, cull_coeff):
    init_len = len(y_train)
    while len(y_train) > init_len * cull_coeff:
        rand_id = np.random.randint(len(y_train))
        if not y_train[rand_id]:
            x_train.pop(rand_id)
            y_train.pop(rand_id)

    sort_ids = [i[0] for i in sorted(enumerate(x_train), key=lambda x: len(x[1]))]
    x_train = [x_train[x] for x in sort_ids]
    y_train = [y_train[x] for x in sort_ids]
    return x_train, y_train


def load_train_data(trainset, save_flag=False):
    x_train = []
    y_train = []

    for dataset in trainset:
        print(f"Dataset: {dataset}")
        segments = os.listdir(f"../data/processed/{dataset}")
        for segment in segments:
            for goal_id in range(len(goals)):
                goal = goals[goal_id]
                x_train, y_train = get_train_data(
                    x_train,
                    y_train,
                    f"../data/processed/{dataset}/{segment}/{goal}.pkl",
                )
    x_train, y_train = cull_trainset(x_train, y_train, CULL_COEFF)
    if save_flag:
        save_pickle(x_train, "../data/train_x.pkl")
        save_pickle(np.transpose(np.array(y_train)), "../data/train_y.pkl")
        print("Saved data")
    return x_train, np.transpose(np.array(y_train))


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
    plt.plot(
        range(EVAL_TIME),
        good_predictons[0, :] / total_examples[0, :],
        label=f"{baseline} AUC = {sum(good_predictons[0, :] / total_examples[0, :])}",
    )
    plt.title(f"{baseline}")


def get_diff(data, segment_length):
    diff = []
    j = EVAL_STEP
    mystep = EVAL_STEP
    while j <= segment_length:
        diff.extend((data[j - mystep : j] - data[j - mystep]) / diff_mean)
        if j >= segment_length:
            break
        if j + EVAL_STEP < segment_length:
            mystep = EVAL_STEP
        else:
            mystep = segment_length - j
        j = min(j + EVAL_STEP, segment_length)
    return diff


def get_train_data(x, y, myfile):
    data = load_pickle(myfile)
    truth = data["label"]
    segment_length = len(data[myfeatures[0]])
    cutoff = max(0, int(segment_length - EVAL_TIME * 1.5))
    segment_length = segment_length - cutoff
    features = []
    for i in range(len(myfeatures)):
        feature = myfeatures[i]
        features.append(data[feature][cutoff:] / my_mean[i])
    if calculate_diff:
        features.append(get_diff(data["rWristRotX euc"], segment_length))

    j = max(EVAL_STEP * 2, segment_length - EVAL_TIME)
    while j <= segment_length:
        features_sliced = np.transpose(np.array(features))[:][j - EVAL_STEP * 2 : j]
        x.append(features_sliced)
        y.append(int(truth))
        if j == segment_length:
            break
        j = min(j + EVAL_STEP, segment_length)

    return x, y


def get_test_data(x, y, myfile, plot_flag=False):
    data = load_pickle(myfile)
    segment_length = len(data[myfeatures[0]])
    cutoff = max(0, int(segment_length - EVAL_TIME * 1.5))
    segment_length = segment_length - cutoff
    features = []
    for i in range(len(myfeatures)):
        feature = myfeatures[i]
        features.append(data[feature][cutoff:] / my_mean[i])
    if calculate_diff:
        features.append(get_diff(data["rWristRotX euc"], segment_length))

    two_d = torch.from_numpy(np.transpose(np.array(features))).type(torch.FloatTensor)
    three_d = two_d.unsqueeze(0).repeat([1, 1, 1])
    x.append(three_d)
    truth = data["label"]
    y.append(int(truth))
    if plot_flag:
        ax1.plot(
            features[0],
            label="rWristRotX euc",
            color=colors[int(truth)],
            linewidth=int(truth) * 2 + 1,
        )
        ax2.plot(
            features[1],
            label="headRotX ori",
            color=colors[int(truth)],
            linewidth=int(truth) * 2 + 1,
        )
    return x, y


def get_prediction(x, y, net, plot_flag):
    y_candidate = []
    y_pred = []
    for i in range(len(x)):
        y_candidate.clear()
        x_example = x[i]
        seq_length = x_example.size()[1]
        time_step = EVAL_STEP * 2
        while time_step < seq_length:
            x_sliced = x_example[:, time_step - 2 * EVAL_STEP : time_step]
            y_candidate.extend([net(x_sliced)] * min(EVAL_STEP, seq_length - time_step))
            time_step = min(time_step + EVAL_STEP, seq_length)
        if plot_flag:
            ax3.plot(
                y_candidate,
                label=str(y[i]),
                color=colors[y[i]],
                linewidth=y[i] * 2 + 1,
            )

        y_pred.append(y_candidate.copy())

    y_array = np.array(y_pred)
    prediction = np.argmax(y_array, axis=0)
    return prediction


def evaluate_net(net, testset, plot_flag=False, verbose=True):
    good_predictons = np.zeros((1, EVAL_TIME))
    total_examples = np.zeros((1, EVAL_TIME))
    x = []
    y = []
    for dataset in testset:  # testset
        segments = os.listdir(f"../data/processed/{dataset}")
        for segment_id in range(round(len(segments) * 1.0)):
            t0 = time.time()
            segment = segments[segment_id]
            if verbose:
                print(f"Processing segment {segment_id} of {len(segments)}")

            if plot_flag:
                fig, (ax1, ax2, ax3) = plt.subplots(
                    3, 1
                )  # TODO: enable input-output plotting via plot_flag

            x.clear()
            y.clear()

            for goal_id in range(len(goals)):
                goal = goals[goal_id]
                x, y = get_test_data(
                    x, y, f"../data/processed/{dataset}/{segment}/{goal}.pkl", plot_flag
                )
            prediction = get_prediction(x, y, net, plot_flag)

            if plot_flag:
                plt.show()

            good_predictons, total_examples = evaluate(
                good_predictons, total_examples, prediction, y, EVAL_TIME
            )
            t1 = time.time()
            if verbose:
                print(f"Total time for segment {t1-t0}")
            print(f"AUC: {sum(good_predictons[0, :] / total_examples[0, :])}")
    return good_predictons, total_examples