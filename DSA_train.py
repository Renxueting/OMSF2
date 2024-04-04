import os
import argparse

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from nets.DSA import unet_generator
from utils.dataset import creat_dataset, creat_dataloader
from utils.train_util import creat_save_path, save_checkpoint, train, val
from utils.losses import CharbonnierLoss
from utils.lr_scheduler import LR_Scheduler


cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    save_path,loss_txt_path = creat_save_path(args)
    train_set, val_set = creat_dataset(args.data_path)
    train_loader, val_loader = creat_dataloader(train_set, val_set, args.batch_size)
    iters_per_epoch = len(train_loader)
    try:
        weights_dict = torch.load(r"./checkpoint.pth")
        start_epoch = weights_dict['epoch']
    except:
        weights_dict = None
        start_epoch = 0
    model = unet_generator(weights_dict)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.9))
    lr_scheduler = LR_Scheduler(mode='cos', init_lr=args.lr, num_epochs=args.epochs, iters_per_epoch=iters_per_epoch, warmup_epochs=args.wp)
    criterion = CharbonnierLoss()
    for epoch in range(start_epoch, args.epochs):
        # ---------- train for one epoch ---------- #
        losses = train(model, optimizer, epoch, iters_per_epoch, train_loader, device, lr_scheduler, criterion)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_PSNR': PSNR}, False, save_path)
        with open(loss_txt_path,'a') as file:
            file.write(str(losses) + '\n')
        # ---------- evaluate on validation set ---------- #
        psnres = val(model, epoch, val_loader, device)
        # ---------- save model ---------- #
        if psnres > PSNR:
            is_best = True
            PSNR = psnres
        else:
            is_best = False
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_PSNR': PSNR}, is_best, save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CorNet')
    parser.add_argument(
        '--data-path', default=r"D:/multi_tsr")
    parser.add_argument('--solver', default='adam',
                        choices=['adam', 'sgd'], help='solver algorithms')
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4,
                        type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=2e-4,
                        type=float, help='initial learning rate')
    parser.add_argument('--wp', '--warm-up', default=0,
                        type=int, help='number of warm up epochs')

    args = parser.parse_args()

    torch.cuda.manual_seed_all(100)

    main(args)
