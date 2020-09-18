import logging
import torch
from torch import nn
from torch.nn import functional as F
from model import layers

logger = logging.getLogger("experiment")


class Learner(nn.Module):

    def __init__(self, config, args):
        super(Learner, self).__init__()

        self.config = config
        self.attention = args.attention
        # this dict contains all tensors needed to be optimized
        self.vars, self.vars_bn = self.parse_config(self.config, nn.ParameterList(), nn.ParameterList())
        self.attention_coeff = None

    def reset_vars(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
        :return:
        """
        num_vars = len(self.vars)
        last_w, last_b = num_vars-2, num_vars-1
        for idx, var in enumerate(self.vars):
            if idx == last_w or idx == last_b:
                if len(var.shape) > 1:
                    logger.info("Resetting weight")
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)

    def parse_config(self, config, vars_list, var_bn_list):

        for i, info_dict in enumerate(config):

            if info_dict["name"] == 'conv2d':
                w, b = layers.conv2d(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = layers.linear(param_config["out-channels"], param_config["in-channels"], info_dict["adaptation"],
                                     info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict['name'] == 'bn':
                param_config = info_dict["config"]
                w, running_mean, running_var = layers.batch_norm(param_config["in-channels"])
                vars_list.append(w)
                var_bn_list.append(running_mean)
                var_bn_list.append(running_var)

            elif info_dict["name"] in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'leakyrelu', 'sigmoid', 'sum', 'softmax', 'mean']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list, var_bn_list

    def forward(self, x, vars=None, config=None, outer_att=False):
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0
        bn_idx = 0
        features_vector = None
        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]

            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])

                if self.attention and idx == 10:
                    features_vector = x

                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'relu':
                x = F.relu(x)

            elif name == 'tanh':
                x = F.tanh(x)

            elif name == 'softmax':
                if outer_att:
                    x = F.softmax(x.unsqueeze(1), dim=1)
                else:
                    x = F.softmax(x, dim=0)
                self.attention_coeff = x

            elif name == 'sum':
                features_vector = features_vector.view(x.size(0), -1)
                if outer_att:
                    features_vector = features_vector.unsqueeze(1)
                    x = torch.sum(features_vector * x, dim=1)
                else:
                    x = torch.sum(features_vector * x, dim=0)
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
            elif name == 'mean':
                if not outer_att:
                    x = torch.mean(x, dim=0)
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)

            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=True)
                idx += 2
                bn_idx += 2

            else:
                raise NotImplementedError
        assert idx == len(vars)
        return x

    def get_adaptation_parameters(self, vars=None):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))

    def get_forward_meta_parameters(self):
        """
        :return: adaptation parameters i.e. parameters changed in the inner loop
        """
        return list(filter(lambda x: x.meta, list(self.vars)))

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
