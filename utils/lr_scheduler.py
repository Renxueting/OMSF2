import math

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    """

    def __init__(self, mode, init_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
        print('Using {} LR Scheduler!'.format(mode))
        if mode == 'step':
            assert lr_step

        self.epoch = -1
        self.mode = mode
        self.lr = init_lr
        self.lr_step = lr_step
        self.N = num_epochs * iters_per_epoch  # total iteration counts
        self.iters_per_epoch = iters_per_epoch

        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):

        T = epoch * self.iters_per_epoch + i  # current iteration count
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1**(epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.8f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0

        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = 3*lr
            optimizer.param_groups[3]['lr'] = 3*lr