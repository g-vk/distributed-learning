import sys
sys.path.insert(0, './')

import numpy as np

import torch
import dc_framework


def train_simple_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )
    criterion = torch.nn.BCELoss()

    train_data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }

    model_dc_framework = dc_framework.init(model, criterion)
    model_dc_framework.train(train_data=train_data, validation_data=train_data)
    model_dc_framework.save("tmp.pt")


def test_simple_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.Sigmoid()
    )

    def accuracy(preds, labels):
        return np.sum(preds == labels) / len(labels)

    test_data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "target": np.array([1, 0, 0, 1])
    }

    model_dc_framework = dc_framework.init(model)
    model_dc_framework.load("tmp.pt")
    assert model_dc_framework.test(test_data=test_data, metric=accuracy) == 2.0


if __name__ == "__main__":
    train_simple_model()
    test_simple_model()
