import torch
    

def fit(model, 
        x, y_true, 
        # signature, 
        loss_fn, optimizer, lr, 
        epoch, i,
        problem_type,
        run,
        config_version,
        ):
    """
    Fit model to data.
    """
    model.train()
    y_pred = model(x)
    loss_value = loss_fn(y_pred, y_true)
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    # Convert loss to proberror used in SUSTAIN.
    item_proberror = 1. - torch.max(y_pred * y_true)
    # Update trainable parameters.
    update_params(model, loss_value, optimizer)
    return model, item_proberror


def update_params(model, loss_value, optimizer):
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()