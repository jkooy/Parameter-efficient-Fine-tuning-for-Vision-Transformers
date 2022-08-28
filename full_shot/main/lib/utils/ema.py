# adopted from https://anmoljoshi.com/Pytorch-Dicussions/
import logging

import torch


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                name = self._remove_prefix(name)
                self.shadow[name] = param.data.clone()

    def cuda(self):
        if not self.shadow:
            return

        for name, param in self.shadow.items():
            self.shadow[name] = self.shadow[name].cuda()

    def load_shadow_states(self, shadow_states):
        for name, param in shadow_states.items():
            if name in shadow_states:
                self.shadow[name] = param.data.clone()
            else:
                logging.info('{} are not used in ema'.format(name))

    @torch.no_grad()
    def __call__(self, model):
        decay = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                name = self._remove_prefix(name)
                assert name in self.shadow
                self.shadow[name].copy_(
                    (1.0-decay)*param.detach().data + decay*self.shadow[name]
                )

    def _remove_prefix(self, name):
        prefix = 'module.'
        if name.startswith(prefix):
            return name[len(prefix):]

        return name

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                name = self._remove_prefix(name)
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                name = self._remove_prefix(name)
                assert name in self.shadow
                param.data = self.original[name]
