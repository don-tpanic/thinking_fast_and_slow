import torch


def fit(model, x, y_true, signature, 
        # loss_fn, optimizer, 
        epoch, i,
        ):
    """
    A complete learning trial of the clustering model.
    """
    # TODO: model.train()  think about how dropout comes into play
    y_logits_fast, y_logits_slow, y_logits_total, extra_stuff = model(
        inp=x, epoch=epoch, i=i, signature=signature, y_true=y_true
    )

    # TODO: careful if there is momentum, loss_fn can not be reused for different branches.
    loss_value_fast = model.loss_fn(y_logits_fast, y_true)
    loss_value_slow = model.loss_fn(y_logits_slow, y_true)
    loss_value = model.loss_fn(y_logits_total, y_true)

    # Convert logits to proba and then to proberror used in SUSTAIN.
    y_pred_fast = torch.nn.functional.softmax(y_logits_fast, dim=1)
    y_pred_slow = torch.nn.functional.softmax(y_logits_slow, dim=1)
    y_pred_total = torch.nn.functional.softmax(y_logits_total, dim=1)
    item_proberror_fast =  1. - torch.max(y_pred_fast * y_true)
    item_proberror_slow =  1. - torch.max(y_pred_slow * y_true)
    item_proberror_total = 1. - torch.max(y_pred_total * y_true)
    # Update trainable parameters.
    
    # TODO: handle extra stuff more flexibly.
    if len(extra_stuff) == 0:
        update_params(model, loss_value, model.optim)
    else:
        TEMP__update_params(model, y_logits_total, extra_stuff[0], extra_stuff[1], x, y_true)

    return model, \
        item_proberror_fast, item_proberror_slow, item_proberror_total, \
        loss_value_fast, loss_value_slow, loss_value


def update_params(model, loss_value, optimizer):
    """
    Update trainable params in the model.
    """
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()


def TEMP__update_params(model, y_logits, act, win_ind, x, y_true):
    """
    Update trainable params in the model.
    """
    # first update params that are updated by SGD,
    # i.e. in fast model, the assoc weights 
    # and  in slow model, all nn weights.
    model.optim.zero_grad()
    loss_value = model.loss_fn(y_logits, y_true)
    loss_value.backward(retain_graph=True)
    model.FastModel.DimensionWiseAttnLayer.weight.grad[:] = 0
    model.optim.step()
    # model.update_assoc(y_logits, y_true, x)
    
    # and then we update the rest of slow model
    # where local update rule is used.
    model.FastModel.update_attn(act, win_ind)
    model.FastModel.update_units(x, win_ind)