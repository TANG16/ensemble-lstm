import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Tweet Classification")

    parser.add_argument(
        "--reload_data",
        dest="reload_data",
        type=int,
        default=1,
        help="1 for new data generation, 0 for using saved data.",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=5,
        help="Number of gradient descent iterations. Default is 5.",
    )

    parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.01,
        help="Gradient descent learning rate. Default is 0.01.",
    )

    parser.add_argument(
        "--hidden_dim",
        dest="hidden_dim",
        type=int,
        default=80,
        help="Number of neurons by hidden layer. Default is 80.",
    )

    parser.add_argument(
        "--lstm_layers",
        dest="lstm_layers",
        type=int,
        default=1,
        help="Number of LSTM layers, default is 1",
    )

    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=5,
        help="Batch size, default is 5.",
    )

    return parser.parse_args()
