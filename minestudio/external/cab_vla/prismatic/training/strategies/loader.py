"""
loader.py

"""

import random

class RatioInterleavedLoader:
    #
    def __init__(self, loaders, ratios):
        assert len(loaders) == len(ratios), "Loaders and ratios must match in length"
        self.loaders = loaders
        self.ratios = ratios
        self.epoch = None
        print(f"RatioInterleavedLoader ratios : {self.ratios}")
    #
    def set_epoch(self, epoch):
        self.epoch = epoch
        for loader in self.loaders:
            sampler = getattr(loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
    #
    def __iter__(self):
        def init_iter(idx):
            loader = self.loaders[idx]
            sampler = getattr(loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch') and self.epoch is not None:
                sampler.set_epoch(self.epoch)
            return iter(loader)

        iters = [init_iter(i) for i in range(len(self.loaders))]
        # max_batches = max(len(loader) for loader in self.loaders)
        # total_ratio = sum(self.ratios)
        # total_steps = max_batches * total_ratio
        total_steps = sum(len(loader) for loader in self.loaders)

        for _ in range(total_steps):
            idx = random.choices(range(len(iters)), weights=self.ratios)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = init_iter(idx)
                yield next(iters[idx])

    def __len__(self):
        # max_batches = max(len(loader) for loader in self.loaders)
        # total_ratio = sum(self.ratios)
        # total_steps = max_batches * total_ratio        
        total_steps = sum(len(loader) for loader in self.loaders)
        return total_steps

