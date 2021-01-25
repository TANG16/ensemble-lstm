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
from config import (
    mykeys,
    mydata,
    trainset,
    testset,
    goals,
    EVAL_TIME,
    STOP_TRAIN,
    EVAL_STEP,
)
import os
import pickle
import matplotlib.pyplot as plt


class DatasetMaper(Dataset):
    """
    Handles batches of dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Execute:

    def __init__(self, args):
        self.__init_data__(args.reload_data)
        self.args = args
        self.batch_size = args.batch_size
        self.model = LSTMBasic(args)

    def __init_data__(self, reload_data):

        self.x_train, self.y_train = self.load_train_data(trainset, reload_data)

    def load_train_data(self, trainset, reload_data):
        if reload_data == True:
            return ut.load_train_data(trainset)
        x_train = ut.load_pickle("../data/train_x.pkl")
        y_train = ut.load_pickle("../data/train_y.pkl")
        return x_train, y_train

    def train(self):

        training_set = DatasetMaper(self.x_train, self.y_train)
        self.loader_training = DataLoader(
            training_set, batch_size=self.batch_size, collate_fn=ut.pad_start
        )
        optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):

            predictions = []
            self.model.train()
            for x_batch, y_batch in self.loader_training:
                x = x_batch.type(torch.FloatTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze(1)

                y_pred = self.model(x)

                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                predictions += list(y_pred.squeeze().detach().numpy())

            train_accuary = self.calculate_accuray(self.y_train, predictions)
            print(
                f"Epoch: {epoch + 1}, loss: {loss.item():.5f}, Train accuracy: {train_accuary:.5f}"
            )
            # if train_accuary > STOP_TRAIN:
            #    break

        with open(f"../models/lstm.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # this code is for devel purposes:
        # good_predictons, total_examples = ut.evaluate_net(self.model, testset)
        # test_accuracy = sum(good_predictons[0, :] / total_examples[0, :])
        # plt.plot(range(EVAL_TIME), good_predictons[0, :] / total_examples[0, :])
        # plt.show()

    def evaluation(self):

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.Tensor)
                y = y_batch.type(torch.Tensor)

                y_pred = self.model(x)
                predictions += list(y_pred.detach().numpy())

        return predictions

    @staticmethod
    def calculate_accuray(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0

        for true, pred in zip(grand_truth, predictions):
            if (pred > 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass

        return (true_positives + true_negatives) / len(grand_truth)


if __name__ == "__main__":

    args = ap.parameter_parser()

    execute = Execute(args)
    execute.train()
