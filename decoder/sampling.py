"""Checkpointable batch order with an explicitly committed training cursor."""
from contextlib import contextmanager, nullcontext
from numbers import Integral

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    """Yield epoch indices; only the training loop commits consumed batches.

    __len__ describes the full epoch, not the remaining suffix. Workers may
    request indices ahead of training; issued indices never advance position.
    Exact resume currently requires deterministic data and num_workers=0.
    """
    def __init__(self, dataset, batch_size, *, shuffle=True, seed=456, drop_last=False):
        for name, value in (('batch_size', batch_size), ('seed', seed)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if name == 'batch_size' else 0):
                raise ValueError(f'Invalid {name}')
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise ValueError('shuffle and drop_last must be booleans')
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.batch_size, self.shuffle, self.seed, self.drop_last = batch_size, shuffle, seed, drop_last
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.position = 0, 0
        self.order = None
        self._issued_position = 0
        self._read_only = False

    @property
    def effective_samples(self):
        n = self.dataset_length
        return n - n % self.batch_size if self.drop_last else n

    def __len__(self):
        return (self.effective_samples + self.batch_size - 1) // self.batch_size

    def config_dict(self):
        return dict(dataset_length=self.dataset_length, batch_size=self.batch_size,
                    shuffle=self.shuffle, seed=self.seed, drop_last=self.drop_last)

    def set_epoch(self, epoch, *, reset=False):
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        if len(self.dataset) != self.dataset_length:
            raise ValueError('Dataset length changed after sampler construction')
        if epoch != self.epoch or self.order is None:
            self.generator.manual_seed((self.seed + epoch) % (2 ** 63))
            self.order = (torch.randperm(self.dataset_length, generator=self.generator)
                          if self.shuffle else torch.arange(self.dataset_length))
            self.position = 0
        elif reset:
            self.position = 0
        self.epoch = epoch
        self._issued_position = self.position

    def __iter__(self):
        if self.order is None:
            self.set_epoch(self.epoch)
        read_only = self._read_only
        position = 0 if read_only else self.position
        if not read_only:
            self._issued_position = position
        while position < self.effective_samples:
            end = min(position + self.batch_size, self.effective_samples)
            batch = self.order[position:end].tolist()
            position = end
            if not read_only:
                self._issued_position = end
            yield batch

    def commit_batches(self, count):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('Committed batch count must be a nonnegative integer')
        target = min(self.position + count * self.batch_size, self.effective_samples)
        if target > self._issued_position:
            raise ValueError('Cannot commit examples that the loader has not requested')
        self.position = target

    @contextmanager
    def evaluation(self):
        previous = self._read_only
        self._read_only = True
        try:
            yield
        finally:
            self._read_only = previous

    def state_dict(self):
        # Order is immutable within an epoch; checkpoint saving makes the CPU copy.
        return dict(format_version=1, **self.config_dict(), epoch=self.epoch,
                    position=self.position, order=self.order,
                    generator_state=self.generator.get_state())

    def validate_state_dict(self, state):
        if state.get('format_version') != 1 or any(state.get(k) != v for k, v in self.config_dict().items()):
            raise ValueError('Sampler configuration differs from checkpoint')
        epoch, position, order = state['epoch'], state['position'], state['order']
        if (isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0 or
                isinstance(position, bool) or not isinstance(position, Integral) or
                not 0 <= position <= self.effective_samples or
                (position != self.effective_samples and position % self.batch_size)):
            raise ValueError('Invalid sampler epoch/committed position')
        if order is None:
            if position != 0:
                raise ValueError('A nonzero position requires a saved order')
        else:
            if (not isinstance(order, torch.Tensor) or order.device.type != 'cpu' or
                    order.dtype != torch.int64 or order.shape != (self.dataset_length,)):
                raise ValueError('Invalid saved sampler order')
            if self.dataset_length and (order.min() < 0 or order.max() >= self.dataset_length or
                    not torch.bincount(order, minlength=self.dataset_length).eq(1).all()):
                raise ValueError('Saved order must contain each dataset index exactly once')

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.epoch, self.position = state['epoch'], state['position']
        self.order = state['order']
        self.generator.set_state(state['generator_state'].cpu())
        self._issued_position = self.position


def resumable_sampler(loader):
    sampler = getattr(loader, 'batch_sampler', None)
    return sampler if type(sampler) is ResumableBatchSampler else None


def evaluation_view(loader):
    sampler = resumable_sampler(loader)
    return sampler.evaluation() if sampler is not None else nullcontext()


def commit_loader_position(loader, progress):
    """Commit actual read batches, including zero-target batches, at a safe boundary."""
    sampler = resumable_sampler(loader)
    if sampler is None or progress is None:
        return
    previously_committed_batches = (sampler.position + sampler.batch_size - 1) // sampler.batch_size
    old_position = sampler.position
    sampler.commit_batches(progress['batches_consumed_epoch'] - previously_committed_batches)
    progress['examples_consumed_epoch'] = sampler.position
    progress['examples_consumed'] = progress.get('examples_consumed', 0) + sampler.position - old_position
    progress['sampler_state'] = sampler.state_dict()
