import torch 

"""
To be consistent..
"""

def fit(model, x, y_true, epoch, i):
    y_logits, act, win_ind = model(x, epoch, i, y_true)
    y_pred = torch.nn.functional.softmax(y_logits, dim=1)
    item_proberror = 1. - torch.max(y_pred * y_true)
    update_params(model, y_logits, act, win_ind, x, y_true)
    return model, item_proberror


def update_params(model, y_logits, act, win_ind, x, y_true):
    """
    Update trainable params in the model.
    """
    model.optim.zero_grad()
    loss_value = model.loss_fn(y_logits, y_true)
    loss_value.backward(retain_graph=True)
    model.DimensionWiseAttnLayer.weight.grad[:] = 0
    model.optim.step()

    # model.update_assoc(y_logits, y_true, x)
    model.update_attn(act, win_ind)
    model.update_units(x, win_ind)