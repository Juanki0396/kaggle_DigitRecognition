
import numpy as np
import torch
import torch.nn as nn


def batch_training(model: nn.Module, X_batch: torch.FloatTensor, Y_batch: torch.FloatTensor, loss_function, optimizer, device):
    """
    This function will train the model during one batch. It returns the loss obtained in the batch.
    """
    # Locate data in the corresponding device
    X_batch.to(device)
    Y_batch.to(device)

    # Reset Gradients
    optimizer.zero_grad()

    # Forward step: Making predictions and computing the loss
    model.train()
    Y_pred = model(X_batch)
    loss = loss_function(Y_pred, Y_batch)

    # Backward step: Computing gradientes and update weights
    loss.backward()
    optimizer.step()

    # Cleaning memory to avoid GPU runs out of memory
    del X_batch, Y_batch, Y_pred

    return loss


def batch_testing(model: nn.Module, X_batch: torch.FloatTensor, Y_batch: torch.IntTensor, loss_function, metric_function, device):
    """
    This function will test the model during one batch.
    """
    # Locate data in the corresponding device
    X_batch.to(device)
    Y_batch.to(device)

    # Forward step: Making predictions and computing the loss
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_batch)
        loss = loss_function(Y_pred, Y_batch)
        metric = metric_function(Y_pred, Y_batch)

    # Cleaning memory to avoid GPU runs out of memory
    del X_batch, Y_batch, Y_pred

    return loss, metric


def training_epoch(model, trainDataLoader, testDataLoader, loss_function, optimizer, metric_function, device):
    """ 
    This function will go along one training epoch
    """
    batch_loss = []

    # Training one epoch
    for X_batch, Y_batch in trainDataLoader:
        training_loss = batch_training(
            model, X_batch, Y_batch, loss_function, optimizer, device)
        batch_loss.append(training_loss)

    average_loss = np.mean(np.array(batch_loss))

    batch_test_loss = []
    batch_metric = []

    # Testing one epoch
    for X_batch, Y_batch in testDataLoader:
        test_loss, metric = batch_testing(
            model, X_batch, Y_batch, loss_function, metric_function, device)
        batch_test_loss.append(test_loss)
        batch_metric.append(metric)

    average_test_loss = np.mean(np.array(batch_test_loss))
    average_metric = np.mean(np.array(batch_metric))

    epoch_loss = {
        'training_batchs': batch_loss,
        'training_average': average_loss,
        'testing_average': average_test_loss,
        'metric_average': average_metric
    }

    return epoch_loss
