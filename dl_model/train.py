import torch


def fit(model, 
        x, 
        y_true, 
        signature, 
        epoch, 
        i,
    ):
    """
    A complete learning trial of the clustering model.
    """
    y_pred = model(
        x, epoch=epoch, i=i, signature=signature, y_true=y_true
    )
    loss_value = model.loss_fn(y_pred, y_true)
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    item_proberror = 1. - torch.max(y_pred * y_true)
    update_params(model, loss_value, model.optim)
    return model, item_proberror


def update_params(model, loss_value, optimizer):
    """
    Update trainable params in the model.
    """
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
