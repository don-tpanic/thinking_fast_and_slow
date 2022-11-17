import torch
    

def fit(model, 
        x, 
        y_true, 
        ):
    """
    Fit model to data.
    """
    model.train()
    y_pred = model(x)
    loss_value = model.loss_fn(y_pred, y_true)
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    # Convert loss to proberror used in SUSTAIN.
    item_proberror = 1. - torch.max(y_pred * y_true)
    # Update trainable parameters.
    update_params(model, loss_value, model.optim)
    return model, item_proberror


def update_params(model, loss_value, optimizer):
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()