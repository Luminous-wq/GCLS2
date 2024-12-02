import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-gpu","--cuda", default=True, type=bool, help="use cuda for training")
    parser.add_argument("-B", "--batch_size",default=32, type=int, help="# of datasets batch size")
    parser.add_argument("-S", "--seed", default=42, type=int, help="random seed")
    parser.add_argument("-E", "--epochs", default=100, type=int, help="# of training epochs")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help="the learning rate")
    parser.add_argument("-dr", "--dropout", default=0.5, type=float, help="the rate of dropout")
    parser.add_argument("-k", "--cluster", default=6, type=int, help="# of cluster for dataset")

    args = parser.parse_args()

    return args