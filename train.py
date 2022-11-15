import torch


def fit(model, x, y_true, signature, 
        loss_fn, optimizer, 
        epoch, i,
        ):
    """
    A complete learning trial of the clustering model.
    """
    # TODO: model.train()  think about how dropout comes into play
    y_pred_fast, y_pred_slow, y_pred = model(
        inp=x, epoch=epoch, i=i, signature=signature, y_true=y_true
    )

    loss_value = loss_fn(y_pred, y_true)
    y_pred_fast = torch.argmax(y_pred_fast, dim=1)
    y_pred_slow = torch.argmax(y_pred_slow, dim=1)
    y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    # Convert loss to proberror used in SUSTAIN.
    item_proberror_fast = 1 - torch.max(y_pred_fast * y_true)
    item_proberror_slow = 1 - torch.max(y_pred_slow * y_true)
    item_proberror = 1. - torch.max(y_pred * y_true)
    # Update trainable parameters.
    update_params(model, loss_value, optimizer)
    return model, item_proberror_fast, item_proberror_slow, item_proberror


def update_params(model, loss_value, optimizer):
    """
    Update trainable params in the model.
    """
    print('[Check] ... update_params ...')
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    # print('\n\n updated params \n\n')
    # for param in model.parameters():
    #     print(param, param.grad)