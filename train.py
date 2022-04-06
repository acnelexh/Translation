from random import seed
import torch
from torch import distributed
from torch.utils import data
import torchvision
import time
import os
import datetime

import presets
import utils
from pathlib import Path
import transforms as T

from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

import models
from engine import train_one_epoch #, evaluate

torch.autograd.set_detect_anomaly(False)  
torch.autograd.profiler.profile(False)  
torch.autograd.profiler.emit_nvtx(False)

def get_MNIST_dataset():
    '''
    Get train, validation, test dataset partion from pytorch dataset.
    '''
    train_set = torchvision.datasets.MNIST(root="datasets/MNIST", train=True, transform=T.ToTensor(), download=False)
    test_set = torchvision.datasets.MNIST(root="datasets/MNIST", train=False, transform=T.ToTensor(), download=False)
    train_set, valid_set = torch.utils.data.random_split(train_set, [50000, 10000])
    return train_set, valid_set, test_set


def get_MNIST_dataloader(train_set, valid_set, test_set):
    '''
    Create dataloader for training, validation, and test set.
    '''
    def batching_fn(batch):
        '''
        Somehow the torchvision MNIST dataset cant load into dataloader
        Need to write our own batching/collating function
        '''
        target_list = []
        img_list = []
        for i in batch:
            image = i[0][0]
            label = i[1]
            img_list.append(image)
            target_list.append(label)
        return torch.stack(img_list), torch.tensor(target_list)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn = batching_fn,
        shuffle=True,
        num_workers=args.workers)
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        collate_fn = batching_fn,
        shuffle=True,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    return train_loader, valid_loader, test_loader

def file_print(evaluate, model, data_loader_test, device, epoch, output_dir):
    import sys
    print('saving results')
    original_stdout = sys.stdout
    (Path(output_dir)/'logs').mkdir(parents=True, exist_ok=True)
    output = os.path.join(output_dir, 'logs', 'output' + str(epoch) + '.txt')
    with open(output, 'w') as f:
        sys.stdout = f
        evaluate(model, data_loader_test, device=device)
        sys.stdout = original_stdout

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    #parser.add_argument('--data-path', default='/data/datasets/coco', help='dataset')
    parser.add_argument('--dataset', default='mnist', help='dataset')
    parser.add_argument('--model', default='ssd_frozen', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--log-epochs', default=5, type=int, help='specific which how often to evalute and store logs file')
    return parser


def main(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # Data Loading code
    print('loading data')
    
    #train_set, valid_set, test_set = get_MNIST_dataset()

    print('creating data loaders')

    #train_loader, valid_loader, test_loader = get_MNIST_dataloader(
    #    train_set, valid_set, test_set)

    '''
    # Sanity Check
    for batch_idx, (x, target) in enumerate(train_loader):
        print(target.shape)
        print(x.shape)
        print(target)
        return
    '''
    
    print("Creating model")
    num_class = 10
    model = models.__dict__[args.model](num_class)

    model.to(device)
    return
    model_without_ddp = model
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                            "are supported.".format(args.lr_scheduler))


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, writer)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        # evaluate after every 5 epoch or at the final epoch
        if (epoch + 1) % args.log_epochs == 0 or (epoch + 1) == args.epochs:
            file_print(evaluate, model, data_loader_test, device, epoch, args.output_dir)
            #evaluate(model, data_loader_test, device=device)

    writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)