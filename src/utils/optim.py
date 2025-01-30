import torch
from torch.optim import Optimizer

class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        """
        LAMB optimizer (cfr. You et. Al, Large Batch Optimization for Deep Learning: Training BERT in 76 minutes, ICLR 2020)
        
        Args:
            params (iterable)
            lr (float)
            betas (tuple)
            eps (float)
            weight_decay (float)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LAMB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        torch.nn.utils.clip_grad_norm_(
            parameters=[
                p for group in self.param_groups for p in group['params']],
            max_norm=1.0,
            norm_type=2
        )

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                scaled_lr = group['lr']
                update = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(update)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        w_norm / g_norm,
                        torch.ones_like(w_norm)
                    )
                    scaled_lr *= trust_ratio.item()

                p.data.add_(update, alpha=-scaled_lr)

        return loss




class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, eps=1e-6, trust_coefficient=0.3):
        """
        LARS optimizer (cfr. You et. Al, Large Batch Training of Convolutional Networks, arXiv 2017)
        
        Args:
            params (iterable)
            lr (float)
            momentum (float)
            weight_decay (float)
            eps (float)
            trust_coefficient (float)
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, trust_coefficient=trust_coefficient)
        super(LARS, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1

                # apply weight decay
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # compute trust ratio
                param_norm = p.data.norm(2)
                grad_norm = grad.norm(2)
                trust_ratio = group['trust_coefficient'] * param_norm / (grad_norm + group['eps']) if param_norm > 0 and grad_norm > 0 else 1.0

                # compute scaled gradient
                scaled_lr = group['lr'] * trust_ratio
                grad = grad.mul(scaled_lr)

                # update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(group['momentum']).add_(grad)

                # update parameter
                p.data.add_(-momentum_buffer)

        return loss
