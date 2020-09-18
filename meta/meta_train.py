import torch
from torch.nn import functional as F
import numpy as np


def train(args, traj_classes, sampler, maml):
    maml.train()
    t1 = np.random.choice(traj_classes, args.tasks, replace=False)

    d_traj_iterators = []
    for t in t1:
        d_traj_iterators.append(sampler.sample_task([t]))

    d_rand_iterator = sampler.get_complete_iterator()

    x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                                      reset=not args.no_reset, rehearsal=args.rehearsal)
    if torch.cuda.is_available():
        x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

    accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
    return maml, accs, loss


def train_iid(opt, maml, iterator, device):
    maml.train()
    correct = 0
    for img, y in iterator:
        img = img.to(device)
        y = y.to(device)
        pred = maml(img, vars=None, outer_att=True)
        pred_q = (pred).argmax(dim=1)
        correct += torch.eq(pred_q, y).sum().item() / len(img)
        opt.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        opt.step()
    accs = correct / len(iterator)
    return maml, accs, loss


def test(maml, iid, iterator, device):
    maml.eval()
    correct = 0
    for img, target in iterator:
        with torch.no_grad():
            img = img.to(device)
            target = target.to(device)
            if not iid:
                logits_q = maml.net(img, vars=None, outer_att=True)
            else:
                logits_q = maml(img, vars=None, outer_att=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct += torch.eq(pred_q, target).sum().item() / len(img)
    return correct