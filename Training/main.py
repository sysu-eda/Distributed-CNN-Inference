import argparse
import os
import time
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

model_names = [
    'alexnet', 'alexnet_s', 'alexnet_s_addchannel','alexnet_s_addchannel_fullshuffle','alexnet_s_addchannelx2_fullshuffle', 'alexnet_s1_2x', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vgg16_bn_s3_addchannel' , 'vgg16_bn_s3_addchannel_v2','resnet18', 'resnet34', 'resnet34_s2_addchannel', 'resnet34_s3_x15', 'resnet34_s3_x175','resnet50','resnet50_s', 'resnet50_justaddgroup','resnet50_block_s','resnet50_block_s_addchannel','resnet50_s3_addchannel','resnet50_s3_addchannel175' , 'resnet101', 'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-g', '--gpu', default='0', type=str, metavar='GPU',
                    help='gpu to use')

best_prec1 = 0.0


def main():
    
    
    
    global args, best_prec1
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    #Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{}{:02}{}'.format(args.arch,local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'alexnet_s':
        model = alexnet_s(pretrained=args.pretrained)
    elif args.arch == 'alexnet_s_addchannel':
        model = alexnet_s_addchannel(pretrained=args.pretrained)    
    elif args.arch == 'alexnet_s_addchannel_fullshuffle':
        model = alexnet_s_addchannel_fullshuffle(pretrained=args.pretrained) 
    elif args.arch == 'alexnet_s_addchannelx2_fullshuffle':
        model = alexnet_s_addchannelx2_fullshuffle(pretrained=args.pretrained) 
    elif args.arch == 'alexnet_s1_2x':
        model = alexnet_s1_2x(pretrained=args.pretrained)     
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn_s3_addchannel':
        model = vgg16_bn_s3_addchannel(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn_s3_addchannel_v2':
        model = vgg16_bn_s3_addchannel_v2(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet34_s2_addchannel':
        model = resnet34_s2_addchannel(pretrained=args.pretrained)
    elif args.arch == 'resnet34_s3_x15':
        model = resnet34_s3_x15(pretrained=args.pretrained)
    elif args.arch == 'resnet34_s3_x175':
        model = resnet34_s3_x175(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet50_s':
        model = resnet50_s(pretrained=args.pretrained)
    elif args.arch == 'resnet50_block_s':
        model = resnet50_block_s(pretrained=args.pretrained)
    elif args.arch == 'resnet50_block_s_addchannel':
        model = resnet50_block_s_addchannel(pretrained=args.pretrained)        
    elif args.arch == 'resnet50_s3_addchannel':
        model = resnet50_s3_addchannel(pretrained=args.pretrained)
    elif args.arch == 'resnet50_s3_addchannel175':
        model = resnet50_s3_addchannel175(pretrained=args.pretrained)
    elif args.arch == 'resnet50_justaddgroup':
        model = resnet50_justaddgroup(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input = input.cuda(async=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            printInfo = 'Epoch: [{0}][{1}/{2}],\t'.format(epoch,i,len(train_loader))+\
                'Time {batch_time.val:.3f}({batch_time.avg:.3f}),\t'.format(batch_time=batch_time) + \
            'Data {data_time.val:.3f} ({data_time.avg:.3f}),\t'.format(data_time=data_time) + \
            'Loss {loss.val:.4f} ({loss.avg:.4f}),\t'.format(loss=losses) + \
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}),\t'.format(top1=top1) + \
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f}),\t'.format(top5=top5)
                                                          
            logging.info(printInfo)
                                                          

def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()         
            
            if i % print_freq == 0:
                printInfo = 'Test: [{0}/{1}],\t'.format(i,len(val_loader))+\
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}),\t'.format(batch_time=batch_time)+\
                    'Loss {loss.val:.4f} ({loss.avg:.4f}),\t'.format(loss=losses)+\
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}),\t'.format(top1=top1)+\
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f}),\t'.format(top5=top5)
                logging.info(printInfo)

    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
