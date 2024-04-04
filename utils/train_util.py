import os
import time
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F
from utils.losses import cal_metrics


def tens2gray(oup):
    oup = oup.detach().cpu().numpy()
    return oup


def creat_save_path(args):
    train_info = '{},{}epochs,b{},lr{}'.format(
        args.solver, args.epochs, args.batch_size, args.lr)
    timestamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    save_path = os.path.join(os.getcwd(), timestamp, train_info)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        txt_path = save_path + "loss.txt"
        file = open(txt_path,"w")
        file.close()
    return save_path, txt_path


def save_checkpoint(state, is_best, save_path):
    print("Best loss:", state["best_loss"])
    torch.save(state, os.path.join(save_path, 'checkpoint.pth'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth'),
                        os.path.join(save_path, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def train(model, optimizer, epoch, iters_per_epoch, train_loader, device, lr_scheduler, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    flag = time.time()

    for i, (img,gt) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - flag)

        img = img.to(device)
        gt = gt.to(device)
 
        lr_scheduler(optimizer, i, epoch)
        pred_img = model(img)

        loss = criterion(pred_img, gt)

        # loss = criterion(pred_img, label_img)
        losses.update(loss.item(), gt.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - flag)
        flag = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t Lr {6}'
                  .format(epoch, i, iters_per_epoch, batch_time, data_time, losses, optimizer.param_groups[0]["lr"]))
    return losses.avg


@torch.no_grad()
def val(model, val_loader, device):
    psnres = AverageMeter()
    ssimes = AverageMeter()

    model.eval()
    for i, (corr, intn, target, t) in enumerate(val_loader):
        corr = corr.to(device)
        intn = intn.to(device)
        target = target.to(device)
        
        pred = model(corr, intn, t)
        output = torch.clamp(pred, 0, 1)
        psnr = cal_metrics(output, target)
        psnres.update(psnr, target.size(0))
        # ssimes.update(ssim, target.size(0))

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t PSNR {2}'.format(i, len(val_loader), psnres))
    print(' * PSNR {:.3f}'.format(psnres.avg))

    return psnres.avg, ssimes.avg


@torch.no_grad()
def test(model, val_loader, device):
    psnres = AverageMeter()
    ssimes = AverageMeter()

    model.eval()
    for i, (corr, intn, target) in enumerate(val_loader):
        corr = corr.to(device)
        intn = intn.to(device)
        target = target.to(device)
        pred = model(corr, intn)
        output = torch.clamp(pred[2], 0, 1)

        psnr, ssim = cal_metrics(output, target)
        psnres.update(psnr, target.size(0))
        ssimes.update(ssim, target.size(0))

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t PSNR {2}\t SSIM {3}'.format(i,
                  len(val_loader), psnres, ssimes))

    print(' * PSNR {:.3f} SSIM {:.3f}'.format(psnres.avg, ssimes.avg))

    return psnres.avg, ssimes.avg
