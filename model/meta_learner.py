import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import model.learner as Learner
from utils.rehearsal import ReservoirSampler
from utils.utils import find_index

logger = logging.getLogger("experiment")


class MetaLearnerClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config, balance_vector):

        super(MetaLearnerClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.dataset = args.dataset
        self.step_count = 0
        if args.attention or args.mean:
            self.aggregation = True
        else:
            self.aggregation = False
        if balance_vector is not None:
            self.balance_param = torch.from_numpy(balance_vector).cuda()
        else:
            self.balance_param = None
        self.rehearsal = args.rehearsal

        self.res_sampler = ReservoirSampler(args.windows, args.buffer_size)
        self.buffer_size = args.buffer_size
        self.net = Learner.Learner(config, args)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def clip_grad_params(self, params, norm=500):

        for p in params.parameters():
            g = p.grad
            # print(g)
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            # print(g)
            p.grad = g

    def inner_update(self, x, fast_weights, y, outer_att):
        adaptation_weight_counter = 0
        logits = self.net(x, fast_weights, outer_att=outer_att)
        if self.balance_param is not None:
            loss = self.balance_param[y] * F.cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()

        # Computes and returns the sum of gradients of outputs w.r.t. the inputs
        grad = torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights), create_graph=True)

        new_weights = []
        for p in fast_weights:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                temp_weight = p - self.update_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def meta_loss(self, x, fast_weights, y, outer_att):

        logits = self.net(x, fast_weights, outer_att=outer_att)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random 1batch of data
        :return:
        """

        if self.rehearsal:
            x_traj = torch.cat((x_traj, x_rand.squeeze(0).unsqueeze(1)))
            y_traj = torch.cat((y_traj, y_rand.squeeze(0).unsqueeze(1)))

            [self.res_sampler.add((x_traj[idx], y_traj[idx])) for idx in range(len(x_traj))]
            if self.rehearsal and len(self.res_sampler.buffer) >= 30:
                index = find_index(y_traj, y_rand)
                x_rand = x_rand[:, index:]
                y_rand = y_rand[:, index:]
                coreset = self.res_sampler.sample(10)
                x_coreset, y_coreset = [], []
                for example in coreset:
                    x_coreset.append(example[0])
                    y_coreset.append(example[1])
                x_coreset = torch.cat(x_coreset)
                y_coreset = torch.cat(y_coreset)
                x_rand = torch.cat((x_coreset, x_rand[0]), 0).unsqueeze(0)
                y_rand = torch.cat((y_coreset, y_rand[0]), 0).unsqueeze(0)

        if self.aggregation:
            self.update_step = 1
            dim = 2
        else:
            self.update_step = len(x_traj)
            dim = 1

        meta_losses = [0 for _ in range(self.update_step + dim)]  # losses_q[i] is the loss on step i
        accuracy_meta_set = [0 for _ in range(self.update_step + dim)]

        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)

        with torch.no_grad():
            # Meta loss before any inner updates
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], self.net.parameters(), y_rand[0], self.aggregation)
            meta_losses[0] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[0] = accuracy_meta_set[0] + classification_accuracy

            # Meta loss after a single inner update
            meta_loss, last_layer_logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], self.aggregation)
            meta_losses[1] += meta_loss

            classification_accuracy = self.eval_accuracy(last_layer_logits, y_rand[0])
            accuracy_meta_set[1] = accuracy_meta_set[1] + classification_accuracy

        if self.aggregation:
            fast_weights = self.inner_update(x_traj.squeeze(1), fast_weights, y_traj[0], False)

            # Computing meta-loss with respect to latest weights
            meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], self.aggregation)
            meta_losses[-1] += meta_loss
        else:
            for k in range(1, self.update_step):
                # Doing inner updates using fast weights
                fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], False)

                # Computing meta-loss with respect to latest weights
                meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], self.aggregation)
                meta_losses[k+1] += meta_loss

                # Computing accuracy on the meta and traj set for understanding the learning
                with torch.no_grad():
                    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                    classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy
                    accuracy_meta_set[k + 1] = accuracy_meta_set[k + 1] + classification_accuracy

        # Taking the meta gradient step
        self.optimizer.zero_grad()

        meta_loss = meta_losses[-1]

        meta_loss.backward()

        self.clip_grad_params(self.net, norm=5)

        self.optimizer.step()
        accuracies = np.array(accuracy_meta_set) / len(x_rand[0])
        return accuracies, meta_losses

    def sample_training_data(self, traj_iterators, rand_iterator, reset=True, rehearsal=False):
        # Sample data for inner and meta updates
        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []

        counter = 0
        x_rand_temp = []
        y_rand_temp = []

        for it1 in traj_iterators:
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                counter += 1

                if counter <= int(len(it1) * 2 / 3):
                    x_traj.append(img)
                    y_traj.append(data)
                else:
                    x_rand_temp.append(img)
                    y_rand_temp.append(data)

        # Sampling the random batch of data
        counter = 0
        for img, data in rand_iterator:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        if not rehearsal:
            x_rand = torch.cat([x_rand, x_rand_temp], 1)
            y_rand = torch.cat([y_rand, y_rand_temp], 1)
        else:
            x_rand = x_rand_temp
            y_rand = y_rand_temp

        return x_traj, y_traj, x_rand, y_rand