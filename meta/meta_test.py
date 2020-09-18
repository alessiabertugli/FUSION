import torch
from torch.nn import functional as F
import torch.nn as nn
from scipy import stats
from utils.rehearsal import ReservoirSampler
import model.learner as learner
from model.modelfactory import get_model


def model_loader(args, classes, device):
    config = get_model(args, classes)
    if args.scratch:
        maml = learner.Learner(config, args)
    else:
        maml_old = learner.Learner(config, args)
        maml = torch.load(args.model, map_location="cpu")

        if maml.config[-1]['config']['out-channels'] != classes:
            maml.config[-1]['config']['out-channels'] = classes
            key_last_vars = [*maml.vars._parameters.keys()][-2:]
            maml.vars._parameters[key_last_vars[0]] = nn.Parameter(
                torch.ones(classes, maml.vars._parameters[key_last_vars[0]].shape[1]))
            maml.vars._parameters[key_last_vars[1]] = nn.Parameter(torch.zeros(classes))

        for (n1, old_model), (n2, loaded_model) in zip(maml_old.named_parameters(), maml.named_parameters()):
            loaded_model.adaptation = old_model.adaptation
            loaded_model.meta = old_model.meta

        maml.reset_vars()

    maml = maml.to(device)

    return maml


def train(args, maml, iterator_sorted, aggregation, device, lr):
    maml.train()
    res_sampler = ReservoirSampler(args.windows, args.buffer_size)
    opt = torch.optim.Adam(maml.get_adaptation_parameters(), lr=lr)
    for _ in range(0, args.epoch):
        for img, y in iterator_sorted:
            # Rehearsal
            res_sampler.add((img, y))
            if args.rehearsal and len(res_sampler.buffer) >= 10:
                coreset = res_sampler.sample(5)
                x_coreset, y_coreset = [], []
                for example in coreset:
                    x_coreset.append(example[0])
                    y_coreset.append(example[1])
                img_coreset = torch.cat(x_coreset)
                y_coreset = torch.cat(y_coreset)

                img = torch.cat([img, img_coreset], dim=0).to(device)
                y = torch.cat([y, y_coreset], dim=0).to(device)
            else:
                img = img.to(device)
                y = y.to(device)

            pred = maml(img, outer_att=aggregation)
            opt.zero_grad()
            loss = F.cross_entropy(pred, y)
            loss.backward()
            opt.step()
    return maml


def test(logger, maml, iterator, aggregation, device):
    maml.eval()
    correct = 0
    for img, target in iterator:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            logits_q = maml(img, vars=None, outer_att=aggregation)

            pred_q = (logits_q).argmax(dim=1)

            correct += torch.eq(pred_q, target).sum().item() / len(img)

    current_acc = correct / len(iterator)
    logger.info(str(current_acc))
    return current_acc


def lr_search(args, classes, iterator_sorted, iterator, logger, aggregation, device):
    lr_list = [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001, 0.0000003, 0.0000001]
    lr_all = []
    max_lr = 0
    for lr_search in range(0, 5):
        max_acc = -10
        for lr in lr_list:
            maml = model_loader(args, classes, device)
            args.epoch = 1
            maml = train(args, maml, iterator_sorted, aggregation, device, lr)
            logger.info("Result after one epoch for LR = %f", lr)
            current_acc = test(logger, maml, iterator, aggregation, device)
            if current_acc > max_acc:
                max_acc = current_acc
                max_lr = lr
                print("max lr ", max_lr)
                print("mac acc ", max_acc)
        lr_all.append(max_lr)
    best_lr = float(stats.mode(lr_all)[0][0])
    logger.info("BEST LR %s= ", str(best_lr))
    return best_lr