import torch
from torch.optim.optimizer import Optimizer, required

class Lion(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.99), weight_decay=0.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                # --- AMP 対応：すべての演算を float32 に昇格 ---
                dtype = p.dtype
                param_data = p.data.float()
                grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(param_data)

                exp_avg = state['exp_avg']

                # Weight decay（L2 regularization）
                if wd != 0:
                    param_data.mul_(1 - lr * wd)

                # EMA update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Parameter update (sign-based)
                update = exp_avg.sign()
                param_data.add_(update, alpha=-lr)

                # 書き戻し：AMPで動作するように元の dtype へ
                p.data.copy_(param_data.to(dtype))

        return loss

class LionAMP(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(p.dtype)
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg = state['exp_avg']
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                if wd != 0.0:
                    p.data.mul_(1 - lr * wd)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add(grad, alpha=1 - beta2)
        return loss
