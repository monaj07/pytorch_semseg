def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    lr = init_lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import pdb

def adjust_learning_rate_v2(optimizer, optimizer_init, epoch, step=20):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    for (param_group, param_group_init) in zip(optimizer.param_groups, optimizer_init.param_groups):
        param_group['lr'] = param_group_init['lr'] * (0.5 ** (epoch // step))