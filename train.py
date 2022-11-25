import torch


def fit(model, x, y_true, signature, 
        # loss_fn, optimizer, 
        epoch, i,
        ):
    """
    A complete learning trial of the clustering model.
    """
    # TODO: model.train()  think about how dropout comes into play
    y_pred_fast, y_pred_slow, y_pred_total = model(
        inp=x, epoch=epoch, i=i, signature=signature, y_true=y_true
    )

    # TODO: careful if there is momentum, loss_fn can not be reused for different branches.
    loss_value_fast = model.loss_fn(y_pred_fast, y_true)
    loss_value_slow = model.loss_fn(y_pred_slow, y_true)
    loss_value = model.loss_fn(y_pred_total, y_true)

    # Convert logits to proba and then to proberror used in SUSTAIN.
    y_pred_fast = torch.nn.functional.softmax(y_pred_fast, dim=1)
    y_pred_slow = torch.nn.functional.softmax(y_pred_slow, dim=1)
    y_pred_total = torch.nn.functional.softmax(y_pred_total, dim=1)
    item_proberror_fast =  1. - torch.max(y_pred_fast * y_true)
    item_proberror_slow =  1. - torch.max(y_pred_slow * y_true)
    item_proberror_total = 1. - torch.max(y_pred_total * y_true)
    # Update trainable parameters.
    update_params(model, loss_value, model.optim)
    return model, \
        item_proberror_fast, item_proberror_slow, item_proberror_total, \
        loss_value_fast, loss_value_slow, loss_value


def update_params(model, loss_value, optimizer):
    """
    Update trainable params in the model.
    """
    # print('[Check] ... update_params ...')
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    # print('\n\n updated params \n\n')
    # for param in model.parameters():
    #     print(param, param.grad)