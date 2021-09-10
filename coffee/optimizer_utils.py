__all__ = ["get_optimizer_params"]


def get_optimizer_params(model, type="s"):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())  # noqa: F841
    learning_rate = 5e-5
    no_decay = ["bias", "gamma", "beta"]
    if type == "s":
        optimizer_parameters = filter(lambda x: x.requires_grad, model.parameters())
    elif type == "i":

        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "transformer" not in n
                ],
                "lr": 1e-3,
                "weight_decay_rate": 0.01,
            },
        ]
    elif type == "a":
        group1 = ["layer.0.", "layer.1.", "layer.2.", "layer.3."]
        group2 = ["layer.4.", "layer.5.", "layer.6.", "layer.7."]
        group3 = ["layer.8.", "layer.9.", "layer.10.", "layer.11."]
        group_all = [
            "layer.0.",
            "layer.1.",
            "layer.2.",
            "layer.3.",
            "layer.4.",
            "layer.5.",
            "layer.6.",
            "layer.7.",
            "layer.8.",
            "layer.9.",
            "layer.10.",
            "layer.11.",
        ]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group1)
                ],
                "weight_decay_rate": 0.01,
                "lr": learning_rate / 2.6,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group2)
                ],
                "weight_decay_rate": 0.01,
                "lr": learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(nd in n for nd in group3)
                ],
                "weight_decay_rate": 0.01,
                "lr": learning_rate * 2.6,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in group_all)
                ],
                "weight_decay_rate": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate / 2.6,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.transformer.named_parameters()
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)
                ],
                "weight_decay_rate": 0.0,
                "lr": learning_rate * 2.6,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "transformer" not in n
                ],
                "lr": 1e-3,
                # "momentum": 0.99,
            },
        ]
    return optimizer_parameters
