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
from config import mykeys, mydata, trainset, testset, goals, EVAL_TIME
import os
import pickle


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
    """
    Class for execution. Initializes the preprocessing as well as the
    Tweet Classifier model
    """

    def __init__(self, args):
        self.__init_data__(args)

        self.args = args
        self.batch_size = args.batch_size

        self.model = LSTMBasic(args)

    def __init_data__(self, args):
        """
        Initialize preprocessing from raw dataset to dataset split into training and testing
        Training and test datasets are index strings that refer to tokens
        """

        self.x_train, self.y_train = self.get_data(trainset)

    def get_data(self, trainset):
        x_train = []
        y_train = []
        for dataset in ["1_1"]:  # trainset
            segments = os.listdir(f"../data/processed/{dataset}")
            for segment in segments:
                for goal_id in range(len(goals)):
                    goal = goals[goal_id]
                    with open(
                        f"../data/processed/{dataset}/{segment}/{goal}.pkl", "rb"
                    ) as f:
                        data = pickle.load(f)
                        features = []
                        for key in mykeys:
                            features.append(data[key + " euc"])
                            features.append(data[key + " ori"])
                        x_train.append(np.transpose(np.array(features)))
                        truth = data["label"]
                        y_train.append(int(truth))

        # print(len(x_train), len(y_train))
        # print(x_train[0])
        return x_train, np.transpose(np.array(y_train))

    def train(self):

        training_set = DatasetMaper(self.x_train, self.y_train)
        # test_set = DatasetMaper(self.x_test, self.y_test)
        # print(training_set.x)
        self.loader_training = DataLoader(training_set, batch_size=self.batch_size)
        # self.loader_test = DataLoader(test_set)

        optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):

            predictions = []
            self.model.train()
            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.FloatTensor)
                y = y_batch.type(torch.FloatTensor).unsqueeze(1)

                y_pred = self.model(x)

                # print(y, y_pred)

                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                predictions += list(y_pred.squeeze().detach().numpy())

            # test_predictions = self.evaluation()

            train_accuary = self.calculate_accuray(self.y_train, predictions)
            # test_accuracy = self.calculate_accuray(self.y_test, test_predictions)

            print(
                "Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f"
                % (epoch + 1, loss.item(), train_accuary, 0)
            )

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
