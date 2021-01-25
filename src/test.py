import utils as ut
import torch
import numpy as np
from LSTMBasic import LSTMBasic
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argument_parser as ap
from config import mykeys, mydata, myfeatures, trainset, testset, goals, EVAL_TIME
import os
import pickle
import matplotlib.pyplot as plt


# ut.evaluate_baseline("gaze", testset)
for feature in myfeatures:
    ut.evaluate_baseline(feature, testset)

# net = ut.load_pickle("../models/lstm_all.pkl")
net = ut.load_pickle("../models/lstm.pkl")

good_predictons, total_examples = ut.evaluate_net(net, testset)
print(good_predictons[0, :], total_examples)
print(sum(good_predictons[0, :] / total_examples[0, :]))
good_predictons = np.flip(good_predictons)
total_examples = np.flip(total_examples)
plt.plot(
    range(EVAL_TIME),
    good_predictons[0, :] / total_examples[0, :],
    label=f"LSTM AUC = {sum(good_predictons[0, :] / total_examples[0, :])}",
)
plt.title("LSTM accuracy")
plt.legend()
plt.show()