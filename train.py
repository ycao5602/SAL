from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import Tensor

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs, optimizer_kwargs_sc, optimizer_kwargs_pt, \
    optimizer_kwargs_al, optimizer_kwargs_gf, optimizer_kwargs_df
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision, BCELoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer
from models import model_g, model_d, model_x, model_a, model_i2, model_da, model_ga, model_dx, model_gx
from random import sample, choices

'''
========================================================================================
gan: triangle structure. Three discriminators.
attention: None
UNIT
https://arxiv.org/pdf/1703.00848.pdf
triange with reverse from the shared space. orders changed.
========================================================================================
'''

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args

    torch.manual_seed(args.seed)

    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')
    else:
        print("Currently using CPU, however, GPU is highly recommended")
        device = torch.device('cpu')

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    w = dm.w
    average = dm.average
    print("Initializing model: {}".format(args.arch))

    size_embed = args.size_embed
    num_attr = dm.num_train_attrs

    # image branch extractor
    model_I1 = models.init_model(name=args.arch, num_classes=512, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model_I1)))

    # attribute branch extractor
    model_G = model_g(dm.num_train_attrs, 512).to(device)

    # attribute classifier
    model_I2 = model_i2(size_embed, dm.num_train_attrs).to(device)

    # category classifier
    model_I2_sem = model_i2(size_embed, dm.num_train_sems).to(device)

    ######## Synthesis-GAN ##################
    model_GA = model_ga(512, 512).to(device)

    model_GX = model_gx(512+args.noise, 512).to(device)

    model_DC = model_d(2 * 512).to(device)


    ######## Alignment-GAN ##################
    model_DS = model_d(size_embed).to(device)

    model_xencoder = model_x(512, size_embed).to(device) # Ev

    model_aencoder = model_a(512, size_embed).to(device) # Ea


    if use_gpu:
        model_I1 = nn.DataParallel(model_I1).cuda()
        model_I2 = nn.DataParallel(model_I2).cuda()
        model_I2_sem = nn.DataParallel(model_I2_sem).cuda()
        model_G = nn.DataParallel(model_G).cuda()
        model_DC = nn.DataParallel(model_DC).cuda()
        model_DS = nn.DataParallel(model_DS).cuda()
        model_xencoder = nn.DataParallel(model_xencoder).cuda()
        model_aencoder = nn.DataParallel(model_aencoder).cuda()
        model_GA = nn.DataParallel(model_GA).cuda()
        model_GX = nn.DataParallel(model_GX).cuda()

    criterion = BCELoss(num_classes=dm.num_train_attrs, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_sem = CrossEntropyLoss(num_classes=dm.num_train_sems, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_gan = BCELoss(num_classes=1, use_gpu=use_gpu, label_smooth=args.label_smooth)

    # consistency loss, l2 or l1
    if args.loss == 'l2':
        criterion_mse = nn.MSELoss()
    elif args.loss == 'l1':
        criterion_mse = nn.L1Loss()
    else:
        criterion_mse = nn.MSELoss()
        print('mse loss is used since the argument is not correctly given from l1 and l2')

    # optimizer for the image branch in single branch pretrain
    optimizer = init_optimizer(
        list(model_I1.parameters()) + list(model_xencoder.parameters()) + list(model_I2.parameters()) + list(
            model_I2_sem.parameters()), **optimizer_kwargs_pt(args))
    # optimizer for the attribute branch in joint pretrain
    optimizer_G = init_optimizer(
        list(model_G.parameters()) + list(model_aencoder.parameters()) + list(model_I2.parameters()) + list(
            model_I2_sem.parameters()), **optimizer_kwargs_sc(args))
    # optimizer for the image branch in joint pretrain
    optimizer_I = init_optimizer(
        list(model_I1.parameters()) + list(model_xencoder.parameters()) + list(model_I2.parameters()) + list(
            model_I2_sem.parameters()), **optimizer_kwargs(args))
    # optimizer for the attribute branch in SAL
    optimizer_A = init_optimizer(
        list(model_G.parameters()) + list(model_aencoder.parameters()) + list(model_I2.parameters()) + list(
            model_I2_sem.parameters()),
        **optimizer_kwargs_al(args))
    # optimizer for the image branch in SAL
    optimizer_X = init_optimizer(
        list(model_I1.parameters()) + list(model_xencoder.parameters()) + list(model_I2.parameters()) + list(
            model_I2_sem.parameters()),
        **optimizer_kwargs_al(args))
    # optimizer for the synthesis-GAN discriminator
    optimizer_DC = init_optimizer(model_DC.parameters(), **optimizer_kwargs_df(args))
    # optimizer for the synthesis-GAN generators
    optimizer_GC = init_optimizer(list(model_GX.parameters()) + list(model_GA.parameters()),
                                  **optimizer_kwargs_gf(args))
    # optimizer for the alignment-GAN discriminator
    optimizer_DS = init_optimizer(model_DS.parameters(), **optimizer_kwargs_al(args))
    # optimizer for Ea, Ev
    optimizer_ES = init_optimizer(list(model_xencoder.parameters()) + list(model_aencoder.parameters()),
                                  **optimizer_kwargs_al(args))

    # simple checkpoint loading
    if args.folder and args.resume_epoch:
        if args.resume_stage == 'pt':
            args.resume_i1 = 'log/' + args.folder + '/checkpoint_ep_pt_I1' + args.resume_epoch + '.pth.tar'
            args.resume_i2 = 'log/' + args.folder + '/checkpoint_ep_pt_I2' + args.resume_epoch + '.pth.tar'
            args.resume_i2_sem = 'log/' + args.folder + '/checkpoint_ep_pt_I2sem' + args.resume_epoch + '.pth.tar'
            args.resume_x = 'log/' + args.folder + '/checkpoint_ep_pt_x' + args.resume_epoch + '.pth.tar'
        elif args.resume_stage == 'jt':
            args.resume_i1 = 'log/' + args.folder + '/checkpoint_ep_jt_I1' + args.resume_epoch + '.pth.tar'
            args.resume_i2 = 'log/' + args.folder + '/checkpoint_ep_jt_I2' + args.resume_epoch + '.pth.tar'
            args.resume_i2_sem = 'log/' + args.folder + '/checkpoint_ep_jt_I2sem' + args.resume_epoch + '.pth.tar'
            args.resume_a = 'log/' + args.folder + '/checkpoint_ep_jt_a' + args.resume_epoch + '.pth.tar'
            args.resume_x = 'log/' + args.folder + '/checkpoint_ep_jt_x' + args.resume_epoch + '.pth.tar'
            args.resume_g = 'log/' + args.folder + '/checkpoint_ep_jt_G' + args.resume_epoch + '.pth.tar'
        elif args.resume_stage == 'al':
            args.resume_i1 = 'log/' + args.folder + '/checkpoint_ep_al_I1' + args.resume_epoch + '.pth.tar'
            args.resume_g = 'log/' + args.folder + '/checkpoint_ep_al_G' + args.resume_epoch + '.pth.tar'
            args.resume_i2 = 'log/' + args.folder + '/checkpoint_ep_al_I2' + args.resume_epoch + '.pth.tar'
            args.resume_i2_sem = 'log/' + args.folder + '/checkpoint_ep_al_I2sem' + args.resume_epoch + '.pth.tar'
            args.resume_a = 'log/' + args.folder + '/checkpoint_ep_al_a' + args.resume_epoch + '.pth.tar'
            args.resume_x = 'log/' + args.folder + '/checkpoint_ep_al_x' + args.resume_epoch + '.pth.tar'
            args.resume_ga = 'log/' + args.folder + '/checkpoint_ep_al_GA' + args.resume_epoch + '.pth.tar'
            args.resume_gx = 'log/' + args.folder + '/checkpoint_ep_al_GX' + args.resume_epoch + '.pth.tar'
            args.resume_dc = 'log/' + args.folder + '/checkpoint_ep_al_DC' + args.resume_epoch + '.pth.tar'
            args.resume_ds = 'log/' + args.folder + '/checkpoint_ep_al_DS' + args.resume_epoch + '.pth.tar'

    # load pretrained weights for the image extractor
    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model_I1.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        if use_gpu:
            model_I1.module.load_state_dict(model_dict)
        else:
            model_I1.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    # Functions for resuming from checkpoints individually
    if args.resume_i1 and check_isfile(args.resume_i1):
        checkpoint = torch.load(args.resume_i1)
        if use_gpu:
            model_I1.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_I1.load_state_dict(checkpoint['state_dict'])
        if not args.resume_g and not args.resume_gx:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
        args.start_epoch_pt = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume_i1))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_i2 and check_isfile(args.resume_i2):
        checkpoint = torch.load(args.resume_i2)
        if use_gpu:
            model_I2.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_I2.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint from '{}'".format(args.resume_i2))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_i2_sem and check_isfile(args.resume_i2_sem):
        checkpoint = torch.load(args.resume_i2_sem)
        if use_gpu:
            model_I2_sem.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_I2_sem.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint from '{}'".format(args.resume_i2_sem))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_x and check_isfile(args.resume_x):
        checkpoint = torch.load(args.resume_x)
        if use_gpu:
            model_xencoder.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_xencoder.load_state_dict(checkpoint['state_dict'])
        if args.resume_gx:
            args.start_epoch_jt = args.max_epoch_jt
            args.start_epoch_pt = args.max_epoch_pt
            optimizer_X.load_state_dict(checkpoint['optimizer_x'])
            for group in optimizer_X.param_groups:
                group['initial_lr'] = group['lr']
        print("Loaded checkpoint from '{}'".format(args.resume_x))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_a and check_isfile(args.resume_a):
        checkpoint = torch.load(args.resume_a)
        if use_gpu:
            model_aencoder.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_aencoder.load_state_dict(checkpoint['state_dict'])
        if args.resume_gx:
            args.start_epoch_jt = args.max_epoch_jt
            args.start_epoch_pt = args.max_epoch_pt
            optimizer_A.load_state_dict(checkpoint['optimizer_a'])
            for group in optimizer_A.param_groups:
                group['initial_lr'] = group['lr']
        print("Loaded checkpoint from '{}'".format(args.resume_a))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_g and check_isfile(args.resume_g):
        checkpoint = torch.load(args.resume_g)
        if use_gpu:
            model_G.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_G.load_state_dict(checkpoint['state_dict'])
        # loading saved generative model means the pretraining is completed
        if not args.resume_gx:
            args.start_epoch_pt = args.max_epoch_pt
            args.start_epoch_jt = checkpoint['epoch'] + 1
            optimizer_I.load_state_dict(checkpoint['optimizer_i'])
            for group in optimizer_I.param_groups:
                group['initial_lr'] = group['lr']
            optimizer_G.load_state_dict(checkpoint['optimizer_g'])
            for group in optimizer_G.param_groups:
                group['initial_lr'] = group['lr']
        print("Loaded checkpoint from '{}'".format(args.resume_g))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_ga and check_isfile(args.resume_ga):
        checkpoint = torch.load(args.resume_ga)
        if use_gpu:
            model_GA.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_GA.load_state_dict(checkpoint['state_dict'])
        # loading saved generative model means the pretraining is completed
        args.start_epoch_jt = args.max_epoch_jt
        args.start_epoch_pt = args.max_epoch_pt
        print("Loaded checkpoint from '{}'".format(args.resume_ga))

    if args.resume_gx and check_isfile(args.resume_gx):
        checkpoint = torch.load(args.resume_gx)
        if use_gpu:
            model_GX.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_GX.load_state_dict(checkpoint['state_dict'])
        # loading saved generative model means the pretraining is completed
        print("Loaded checkpoint from '{}'".format(args.resume_gx))

    if args.resume_dc and check_isfile(args.resume_dc):
        checkpoint = torch.load(args.resume_dc)
        if use_gpu:
            model_DC.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_DC.load_state_dict(checkpoint['state_dict'])
        optimizer_DC.load_state_dict(checkpoint['optimizer_dc'])
        for group in optimizer_DC.param_groups:
            group['initial_lr'] = group['lr']
        optimizer_GC.load_state_dict(checkpoint['optimizer_gc'])
        for group in optimizer_GC.param_groups:
            group['initial_lr'] = group['lr']
        print("Loaded checkpoint from '{}'".format(args.resume_dc))
        print("- rank1: {}".format(checkpoint['rank1']))

    if args.resume_ds and check_isfile(args.resume_ds):
        checkpoint = torch.load(args.resume_ds)
        if use_gpu:
            model_DS.module.load_state_dict(checkpoint['state_dict'])
        else:
            model_DS.load_state_dict(checkpoint['state_dict'])
        optimizer_DS.load_state_dict(checkpoint['optimizer_ds'])
        for group in optimizer_DS.param_groups:
            group['initial_lr'] = group['lr']
        optimizer_ES.load_state_dict(checkpoint['optimizer_es'])
        for group in optimizer_ES.param_groups:
            group['initial_lr'] = group['lr']
        print("Loaded checkpoint from '{}'".format(args.resume_ds))

    if args.evaluate:
        print("Evaluate only")

        for name in args.dataset:
            print("Evaluating {} ...".format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']

            distmat = test_WOadv(model_I1, model_G, model_xencoder, model_aencoder, queryloader, galleryloader, use_gpu,
                                 return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    dm.label,
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    if args.attribute_prediction:

        print('evaluate the attribute predicting ability of the model')
        for name in args.dataset:
            print("Evaluating {} ...".format(name))
            galleryloader = testloader_dict[name]['gallery']
            test_attr(model_I1, model_I2, model_G, galleryloader, use_gpu)

        return

    start_time = time.time()
    ranklogger = RankLogger(args.dataset, args.dataset)
    train_time = 0
    print("=> Start training")

    if args.fixbase_epoch > 0:
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train_I(epoch, model_I1, model_I2, model_I2_sem, model_xencoder, criterion, criterion_sem, optimizer,
                    trainloader, use_gpu, fixbase=False)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch_pt))
        optimizer.load_state_dict(initial_optim_state)

    # learning rate decay
    # singe pretrain
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    # joint pretrain
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones=args.stepsize, gamma=args.gamma)
    scheduler_I = lr_scheduler.MultiStepLR(optimizer_I, milestones=args.stepsize, gamma=args.gamma)
    # sal
    scheduler_GC = lr_scheduler.MultiStepLR(optimizer_GC, milestones=args.stepsize_sal, gamma=args.gamma)
    scheduler_ES = lr_scheduler.MultiStepLR(optimizer_ES, milestones=args.stepsize_sal, gamma=args.gamma)
    scheduler_DC = lr_scheduler.MultiStepLR(optimizer_DC, milestones=args.stepsize_sal, gamma=args.gamma)
    scheduler_DS = lr_scheduler.MultiStepLR(optimizer_DS, milestones=args.stepsize_sal, gamma=args.gamma)
    scheduler_X = lr_scheduler.MultiStepLR(optimizer_X, milestones=args.stepsize_sal, gamma=args.gamma)
    scheduler_A = lr_scheduler.MultiStepLR(optimizer_A, milestones=args.stepsize_sal, gamma=args.gamma)

    print("***************stage 1: Pretrain the single image network******************")
    for epoch in range(args.start_epoch_pt, args.max_epoch_pt):
        start_train_time = time.time()
        train_I(epoch, model_I1, model_I2, model_I2_sem, model_xencoder, criterion, criterion_sem, optimizer,
                trainloader, use_gpu, fixbase=False)

        scheduler.step()

        if (epoch + 1) % args.save_pt == 0 or (epoch + 1) == args.max_epoch_pt:
            print("=> Save pretrained model")

            if use_gpu:
                state_dict_I1 = model_I1.module.state_dict()
                state_dict_I2 = model_I2.module.state_dict()
                state_dict_I2_sem = model_I2_sem.module.state_dict()
                state_dict_xencoder = model_xencoder.module.state_dict()
            else:
                state_dict_I1 = model_I1.state_dict()
                state_dict_I2 = model_I2.state_dict()
                state_dict_I2_sem = model_I2_sem.state_dict()
                state_dict_xencoder = model_xencoder.state_dict()

            optim_state_dict = optimizer.state_dict()

            if not args.no_save:
                save_checkpoint({
                    'state_dict': state_dict_I1,
                    'rank1': 0,
                    'epoch': epoch,
                    'optimizer': optim_state_dict,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_pt_I1' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_I2,
                    'rank1': 0,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_pt_I2' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_I2_sem,
                    'rank1': 0,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_pt_I2sem' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_xencoder,
                    'rank1': 0,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_pt_x' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    for name in args.dataset:
        print("Evaluating {} ...".format(name))
        queryloader = testloader_dict[name]['query']
        galleryloader = testloader_dict[name]['gallery']
        rank1 = test_WOadv(model_I1, model_G, model_xencoder, model_aencoder, queryloader, galleryloader, use_gpu,
                           ranks=[1, 5, 10, 20],
                           return_distmat=False)
        ranklogger.write(name, args.max_epoch_pt, rank1)
    ranklogger.show_summary()

    start_time = time.time()
    ranklogger = RankLogger(args.dataset, args.dataset)
    train_time = 0

    print("***************stage 2: jointly pretrain both branches******************")
    for epoch in range(args.start_epoch_jt, args.max_epoch_jt):
        start_train_time = time.time()
        train_WOadv(epoch, model_I1, model_I2, model_I2_sem, model_G, model_xencoder, model_aencoder, criterion,
                    criterion_sem, optimizer_I,
                    optimizer_G, trainloader, use_gpu, fixbase=False)
        train_time += round(time.time() - start_train_time)

        scheduler_G.step()
        scheduler_I.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch_jt:
            print("=> Test")

            for name in args.dataset:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test_WOadv(model_I1, model_G, model_xencoder, model_aencoder, queryloader, galleryloader,
                                   use_gpu, ranks=[1, 5, 10, 20], return_distmat=False)
                ranklogger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict_I1 = model_I1.module.state_dict()
                state_dict_I2 = model_I2.module.state_dict()
                state_dict_I2_sem = model_I2_sem.module.state_dict()
                state_dict_G = model_G.module.state_dict()
                state_dict_xencoder = model_xencoder.module.state_dict()
                state_dict_aencoder = model_aencoder.module.state_dict()
            else:
                state_dict_I1 = model_I1.state_dict()
                state_dict_I2 = model_I2.state_dict()
                state_dict_I2_sem = model_I2_sem.state_dict()
                state_dict_G = model_G.state_dict()
                state_dict_xencoder = model_xencoder.state_dict()
                state_dict_aencoder = model_aencoder.state_dict()

            optimizer_I_state_dict = optimizer_I.state_dict()
            optimizer_G_state_dict = optimizer_G.state_dict()

            if not args.no_save:
                save_checkpoint({
                    'state_dict': state_dict_I1,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_I1' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_I2,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_I2' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_I2_sem,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_I2sem' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_G,
                    'rank1': rank1,
                    'epoch': epoch,
                    'optimizer_i': optimizer_I_state_dict,
                    'optimizer_g': optimizer_G_state_dict,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_G' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_xencoder,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_x' + str(epoch + 1) + '.pth.tar'))

                save_checkpoint({
                    'state_dict': state_dict_aencoder,
                    'rank1': rank1,
                    'epoch': epoch,
                }, False, osp.join(args.save_dir, 'checkpoint_ep_jt_a' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()
    train_time = 0

    print("***************stage 3: train the full SAL******************")
    for epoch in range(args.start_epoch_al, args.max_epoch_al):
        start_train_time = time.time()
        train_gan(epoch, w, average, model_I1, model_I2, model_I2_sem, model_G, model_xencoder, model_aencoder, model_GA,
                  model_GX, model_DC,
                  model_DS, criterion, criterion_sem, criterion_gan, criterion_mse, optimizer_GC,
                  optimizer_DS,
                  optimizer_ES, optimizer_X, optimizer_A, optimizer_DC, trainloader, use_gpu,
                  fixbase=False)
        train_time += round(time.time() - start_train_time)

        scheduler_X.step()
        scheduler_A.step()
        scheduler_DS.step()
        scheduler_ES.step()
        scheduler_DC.step()
        scheduler_GC.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch_al:
            print("=> Test")

            for name in args.dataset:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test_WOadv(model_I1, model_G, model_xencoder, model_aencoder, queryloader, galleryloader,
                                   use_gpu, ranks=[1, 5, 10, 20], return_distmat=False)
                ranklogger.write(name, epoch + 1, rank1)



            if use_gpu:
                state_dict_I1 = model_I1.module.state_dict()
                state_dict_G = model_G.module.state_dict()
                state_dict_I2 = model_I2.module.state_dict()
                state_dict_I2_sem = model_I2_sem.module.state_dict()
                state_dict_xencoder = model_xencoder.module.state_dict()
                state_dict_aencoder = model_aencoder.module.state_dict()
                state_dict_GA = model_GA.module.state_dict()
                state_dict_GX = model_GX.module.state_dict()
                state_dict_DC = model_DC.module.state_dict()
                state_dict_DS = model_DS.module.state_dict()
            else:
                state_dict_I1 = model_I1.state_dict()
                state_dict_G = model_G.state_dict()
                state_dict_I2_sem = model_I2_sem.state_dict()
                state_dict_xencoder = model_xencoder.state_dict()
                state_dict_aencoder = model_aencoder.state_dict()
                state_dict_GA = model_GA.state_dict()
                state_dict_GX = model_GX.state_dict()
                state_dict_DC = model_DC.module.state_dict()
                state_dict_DS = model_DS.module.state_dict()

            optimizer_A_state_dict = optimizer_A.state_dict()
            optimizer_X_state_dict = optimizer_X.state_dict()
            optimizer_DC_state_dict = optimizer_DC.state_dict()
            optimizer_GC_state_dict = optimizer_GC.state_dict()
            optimizer_DS_state_dict = optimizer_DS.state_dict()
            optimizer_ES_state_dict = optimizer_ES.state_dict()

            save_checkpoint({
                'state_dict': state_dict_I1,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_I1' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_G,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_G' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_I2,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_I2' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_I2_sem,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_I2sem' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_xencoder,
                'rank1': rank1,
                'epoch': epoch,
                'optimizer_x': optimizer_X_state_dict,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_x' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_aencoder,
                'rank1': rank1,
                'epoch': epoch,
                'optimizer_a': optimizer_A_state_dict,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_a' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_GX,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_GX' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_GA,
                'rank1': rank1,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_GA' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_DS,
                'rank1': rank1,
                'epoch': epoch,
                'optimizer_ds': optimizer_DS_state_dict,
                'optimizer_es': optimizer_ES_state_dict,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_DS' + str(epoch + 1) + '.pth.tar'))

            save_checkpoint({
                'state_dict': state_dict_DC,
                'rank1': rank1,
                'epoch': epoch,
                'optimizer_dc': optimizer_DC_state_dict,
                'optimizer_gc': optimizer_GC_state_dict,
            }, False, osp.join(args.save_dir, 'checkpoint_ep_al_DC' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print('ordinary testing')
    ranklogger.show_summary()


# stage 1: pretrain with the image branch only
def train_I(epoch, model_I1, model_I2, model_I2_sem, model_xencoder, criterion, criterion_sem, optimizer, trainloader,
            use_gpu, fixbase=False):
    ########################
    # Training image network#
    ########################

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_I1.train()
    model_I2.train()
    model_xencoder.train()

    end = time.time()

    for batch_idx, (imgs, _, attrs, _, _, _, sems) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, attrs, sems = imgs.cuda(), attrs.cuda(), sems.cuda()

        sems = sems.long()
        attrs = attrs.float()

        img_feature = model_xencoder(model_I1(imgs))

        output_attr = model_I2(img_feature)
        output_sem = model_I2_sem(img_feature)
        loss = (2 - args.lamb_sem) * criterion(output_attr, attrs) + args.lamb_sem * criterion_sem(output_sem, sems)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), sems.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        end = time.time()


# stage 2: jointly train the two branches without adversarial learning
def train_WOadv(epoch, model_I1, model_I2, model_I2_sem, model_G, model_xencoder, model_aencoder, criterion,
                criterion_sem, optimizer_I,
                optimizer_G, trainloader, use_gpu, fixbase=False):
    losses = AverageMeter()
    losses_G = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_I1.train()
    model_I2.train()
    model_G.train()
    model_aencoder.train()
    model_xencoder.train()

    end = time.time()

    for batch_idx, (imgs, _, attrs, _, _, _, sems) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, attrs, sems = imgs.cuda(), attrs.cuda(), sems.cuda()

        sems = sems.long()
        attrs = attrs.float()

        # update image branch
        img_feature = model_xencoder(model_I1(imgs))

        loss = criterion(model_I2(img_feature), attrs) \
               + criterion_sem(model_I2_sem(img_feature),
                               sems)

        optimizer_I.zero_grad()
        loss.backward()
        optimizer_I.step()

        # update attribute branch

        attr_feature = model_aencoder(model_G(attrs))
        loss_G = (2 - args.lamb_sem) * criterion(model_I2(attr_feature), attrs) \
                 + args.lamb_sem * criterion_sem(model_I2_sem(attr_feature),
                                                 sems)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ###########################end of GAN structure############################ #

        batch_time.update(time.time() - end)

        losses.update(loss.item(), sems.size(0))
        losses_G.update(loss_G.item(), sems.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f}+{loss_g.val:.4f} ({loss.avg:.4f}+{loss_g.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_g=losses_G))

        end = time.time()


def train_gan(epoch, w, average, model_I1, model_I2, model_I2_sem, model_G, model_xencoder, model_aencoder, model_GA, model_GX,
              model_DC,
              model_DS, criterion, criterion_sem, criterion_gan, criterion_mse, optimizer_GC,
              optimizer_DS,
              optimizer_ES, optimizer_X, optimizer_A, optimizer_DC, trainloader, use_gpu,
              fixbase=False):
    #################################################################################
    # 1. extract image concept by removing softmax and adding fc to resnet-50.      #
    # 2. generate image-analogous concept by adding multiple fully connected layers.#
    # 3. concept discriminator                                                      #
    #################################################################################
    losses_I = AverageMeter()
    losses_G = AverageMeter()
    losses_DC = AverageMeter()
    losses_GX = AverageMeter()
    losses_GA = AverageMeter()
    losses_CYCLE = AverageMeter()
    losses_DS = AverageMeter()
    losses_ES = AverageMeter()
    losses_EMSE = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_I1.train()
    model_I2.train()
    model_G.train()
    model_xencoder.eval()
    model_aencoder.eval()
    model_GX.train()
    model_GA.train()
    model_DC.train()
    model_DS.train()

    ''' DISABLING FIXBASE
    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)
    '''
    end = time.time()

    ratio = args.ssl_loss_ratio

    if epoch >= args.start_mse:
        lamb_mse = args.lamb_mse
    else:
        lamb_mse = 0

    ratio_l = args.train_batch_size / (args.train_batch_size + args.num_fake)
    ratio_u = 1 - ratio_l

    if epoch >= args.epoch_add_fake:
        for batch_idx, (imgs, _, attrs, _, _, _, sems) in enumerate(trainloader):
            # print('iter')
            data_time.update(time.time() - end)

            ############################################################################
            #  sample fake attributes from prior distribution (training distribution)  #
            ############################################################################
            fake_attrs = fake_attributes(w, average, args.num_fake)

            ua_valid = Variable(Tensor(fake_attrs.size(0), 1).fill_(1.0), requires_grad=False)
            ua_fake = Variable(Tensor(fake_attrs.size(0), 1).fill_(0.0), requires_grad=False)

            if batch_idx == 0:
                print('train with randomly sampled (based on prior distribution) fake attributes:')

            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            if use_gpu:
                imgs, attrs, sems, fake_attrs \
                    = imgs.cuda(), attrs.cuda(), sems.cuda(), fake_attrs.cuda()
                valid, fake, ua_valid, ua_fake \
                    = valid.cuda(), fake.cuda(), ua_valid.cuda(), ua_fake.cuda()

            sems = sems.long()
            attrs = attrs.float()
            fake_attrs = fake_attrs.float()

            #############################
            #  train the image network  #
            #############################
            x_open = model_I1(imgs)
            a_open = model_G(attrs)
            faa_open = model_G(fake_attrs)
            x = x_open.detach()
            a = a_open.detach()
            faa = faa_open.detach()

            # ############################# Synthesis-GAN structure ################################# #

            # ########################## D(A,I) ############################ #
            # update the discriminator for the synthesis-GAN                 #
            ##################################################################
            model_GX.train()
            model_GA.train()
            model_xencoder.eval()
            model_aencoder.eval()

            noise = torch.randn(attrs.size(0), args.noise)
            noise2 = torch.randn(fake_attrs.size(0), args.noise)

            if use_gpu:
                noise = noise.cuda()
                noise2 = noise2.cuda()

            # add noise to image feature generator
            x_hat = model_GX(torch.cat((a, noise), 1).detach())
            fax_hat = model_GX(torch.cat((faa, noise2), 1).detach())

            a_hat = model_GA(x)

            pair_ax = torch.cat((a, x), 1)
            pair_ax1 = torch.cat((a, x_hat), 1)
            pair_a1x = torch.cat((a_hat, x), 1)
            pair_faafax1 = torch.cat((faa, fax_hat), 1)

            loss_DC = criterion_gan(model_DC(pair_ax.detach()), valid)\
                      + ratio_l*criterion_gan(model_DC(pair_ax1.detach()),fake) \
                      + ratio_u*criterion_gan(model_DC(pair_faafax1.detach()),ua_fake) \
                      + criterion_gan(model_DC(pair_a1x.detach()), fake)

            optimizer_DC.zero_grad()
            loss_DC.backward()
            optimizer_DC.step()

            # ########################## GX and GA ############################ #
            # update the generators for the synthesis-GAN with both             #
            # adversarial and consistency loss                                  #             #
            ################################################################### #

            encoded_x_hat = model_xencoder(x_hat)
            encoded_a_hat = model_aencoder(a_hat)
            encoded_fax_hat = model_xencoder(fax_hat)
            encoded_x = model_xencoder(x)
            encoded_a = model_aencoder(a)
            encoded_faa = model_aencoder(faa)

            loss_GX = ratio_l*criterion_gan(model_DC(pair_ax1), valid) \
                      + ratio_l*criterion_mse(encoded_x_hat, encoded_a.detach()) \
                      + criterion_mse(encoded_x_hat, encoded_x.detach()) \
                      + ratio_u*ratio * criterion_gan(model_DC(pair_faafax1), ua_valid) \
                      + ratio_u*ratio * criterion_mse(encoded_fax_hat, encoded_faa.detach())

            loss_GA = criterion_gan(model_DC(pair_a1x), valid) \
              + criterion_mse(encoded_a_hat, encoded_x.detach()) \
              + criterion_mse(encoded_a_hat, encoded_a.detach())

            loss_cycle = ratio_l*criterion_mse(model_GA(x_hat), a) \
                         + ratio_u*ratio * criterion_mse(model_GA(fax_hat), faa)

            optimizer_GC.zero_grad()
            loss_GC = loss_GX + loss_GA + loss_cycle
            loss_GC.backward()
            optimizer_GC.step()

            # ############################# Alignment-GAN structure ################################# #

            # ########################## D(Sa,Sx) ############################ #
            # update the discriminator in the alignment-GAN                    #
            # ################################################################ #
            model_GX.eval()
            model_GA.eval()
            model_xencoder.train()
            model_aencoder.train()

            encoded_x_hat = model_xencoder(x_hat.detach())
            encoded_a_hat = model_aencoder(a_hat.detach())
            encoded_x = model_xencoder(x)
            encoded_a = model_aencoder(a)
            encoded_fax_hat = model_xencoder(fax_hat.detach())
            encoded_faa = model_aencoder(faa.detach())

            loss_DS = ratio_u*criterion_gan(model_DS(encoded_a.detach()), fake) \
                      + criterion_gan(model_DS(encoded_x.detach()), valid) \
                      + lamb_mse*criterion_gan(model_DS(encoded_a_hat.detach()), fake) \
                      + lamb_mse*ratio_u*criterion_gan(model_DS(encoded_x_hat.detach()), valid) \
                      + ratio_u*ratio * criterion_gan(model_DS(encoded_faa.detach()), ua_fake) \
                      + ratio_u*ratio * criterion_gan(model_DS(encoded_fax_hat.detach()), ua_valid)

            optimizer_DS.zero_grad()
            loss_DS.backward()
            optimizer_DS.step()

            # ########################## E(Sa,Sx) ############################ #
            # update the encoders in the alignment-GAN                         #
            # ################################################################ #
            encoded_a = model_aencoder(a)
            encoded_faa = model_aencoder(faa)

            loss_ES = ratio_l*criterion_gan(model_DS(encoded_a), valid) \
                      + lamb_mse*criterion_gan(model_DS(encoded_a_hat), valid) \
                      + ratio_u*ratio * criterion_gan(model_DS(encoded_faa), ua_valid)

            optimizer_ES.zero_grad()
            (loss_ES).backward()
            optimizer_ES.step()

            ##########################
            #  train the classifier  #
            ##########################
            x_feature = model_xencoder(x_open)
            x_hat_feature = model_xencoder(x_hat.detach())
            loss_I = (2 - args.lamb_sem) * (criterion(model_I2(x_feature), attrs)
                                            + lamb_mse * criterion(model_I2(x_hat_feature),attrs)) / 2 \
                     + args.lamb_sem * (criterion_sem(model_I2_sem(x_feature), sems)
                                        + lamb_mse * criterion_sem(model_I2_sem(x_hat_feature), sems)) / 2

            optimizer_X.zero_grad()
            loss_I.backward()
            optimizer_X.step()

            #######################################
            #  regularising the attribute branch  #
            #######################################
            a_feature = model_aencoder(a_open)
            faa_feature = model_aencoder(faa_open)
            a_hat_feature = model_aencoder(a_hat.detach())
            optimizer_A.zero_grad()

            loss_G = (2 - args.lamb_sem) * (ratio_l*criterion(model_I2(a_feature), attrs)
                                            + lamb_mse * criterion(model_I2(a_hat_feature), attrs)
                                            + ratio_u*criterion(model_I2(faa_feature), fake_attrs)) / 2 \
                     + args.lamb_sem * (criterion_sem(model_I2_sem(a_feature), sems)
                                        + lamb_mse * criterion_sem(model_I2_sem(a_hat_feature), sems)) / 2
            loss_G.backward()
            optimizer_A.step()

            # ##########################end of updating process############################ #
            batch_time.update(time.time() - end)

            losses_I.update(loss_I.item(), sems.size(0))
            losses_G.update(loss_G.item(), sems.size(0))
            losses_DC.update(loss_DC.item(), sems.size(0))
            losses_GX.update(loss_GX.item(), sems.size(0))
            losses_GA.update(loss_GA.item(), sems.size(0))
            losses_CYCLE.update(loss_cycle.item(), sems.size(0))
            losses_DS.update(loss_DS.item(), sems.size(0))
            losses_ES.update(loss_ES.item(), sems.size(0))

            del loss_I
            del loss_G
            del loss_GC
            del loss_DC
            del loss_DS
            del loss_ES

            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                      'Loss i {loss_i.val:.4f}+g {loss_g.val:.4f}'
                      '+dc {loss_dc.val:.4f}+gx {loss_gx.val:.4f}'
                      '+ga {loss_ga.val:.4f}+cycle {loss_cycle.val:.4f}'
                      '+ds {loss_ds.val:.4f}+es {loss_es.val:.4f}'
                      '(i {loss_i.avg:.4f}+g {loss_g.avg:.4f}'
                      '+dc {loss_dc.avg:.4f}+gx {loss_gx.avg:.4f}'
                      '+ga {loss_ga.avg:.4f}+cycle {loss_cycle.avg:.4f}'
                      '+ds {loss_ds.avg:.4f}+es {loss_es.avg:.4f})\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss_i=losses_I, loss_g=losses_G, loss_dc=losses_DC, loss_gx=losses_GX,
                    loss_ga=losses_GA, loss_cycle=losses_CYCLE, loss_ds=losses_DS, loss_es=losses_ES))

            end = time.time()
            torch.cuda.empty_cache()
    else:
        for batch_idx, (imgs, _, attrs, _, _, _, sems) in enumerate(trainloader):
            # print('iter')
            data_time.update(time.time() - end)
            if batch_idx == 0:
                print('train with only labelled data:')

            valid = Variable(Tensor(attrs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(attrs.size(0), 1).fill_(0.0), requires_grad=False)

            if use_gpu:
                imgs, attrs, sems = imgs.cuda(), attrs.cuda(), sems.cuda()
                valid, fake = valid.cuda(), fake.cuda()

            sems = sems.long()
            attrs = attrs.float()

            #############################
            #  train the image network  #
            #############################

            x_open = model_I1(imgs)
            a_open = model_G(attrs)
            x = x_open.detach()
            a = a_open.detach()

            # ############################# Synthesis-GAN structure ################################# #

            # ########################## D(A,I) ############################ #
            # update the discriminator for the synthesis-GAN                 #
            ##################################################################
            model_GX.train()
            model_GA.train()
            model_xencoder.eval()
            model_aencoder.eval()

            noise = torch.randn(attrs.size(0), args.noise)

            if use_gpu:
                noise = noise.cuda()

            # add noise to image feature generator
            x_hat = model_GX(torch.cat((a, noise), 1).detach())
            a_hat = model_GA(x)

            pair_ax = torch.cat((a, x), 1)
            pair_ax1 = torch.cat((a, x_hat), 1)
            pair_a1x = torch.cat((a_hat, x), 1)

            loss_DC = criterion_gan(model_DC(pair_ax.detach()), valid) + criterion_gan(model_DC(pair_ax1.detach()),
                                                                                       fake) \
                      + criterion_gan(model_DC(pair_a1x.detach()), fake)

            optimizer_DC.zero_grad()
            loss_DC.backward()
            optimizer_DC.step()

            # ########################## GX and GA ############################ #
            # update the generators for synthesis-GAN with both                 #
            # adversarial and consistency loss                                  #             #
            ################################################################### #

            encoded_x_hat = model_xencoder(x_hat)
            encoded_a_hat = model_aencoder(a_hat)
            encoded_x = model_xencoder(x)
            encoded_a = model_aencoder(a)

            loss_GX = criterion_gan(model_DC(pair_ax1), valid) \
                      + criterion_mse(encoded_x_hat, encoded_a.detach()) \
                      + criterion_mse(encoded_x_hat, encoded_x.detach())


            loss_GA = criterion_gan(model_DC(pair_a1x), valid) \
                      + criterion_mse(encoded_a_hat, encoded_x.detach()) \
                      + criterion_mse(encoded_a_hat, encoded_a.detach())

            loss_cycle = criterion_mse(model_GA(x_hat), a)

            optimizer_GC.zero_grad()
            loss_GC = loss_GX + loss_GA + loss_cycle
            loss_GC.backward()
            optimizer_GC.step()

            # ############################# Alignment-GAN structure ################################# #

            # ########################## D(Sa,Sx) ############################ #
            # update the discriminator in the alignment-GAN                    #
            # ################################################################ #
            model_GX.eval()
            model_GA.eval()
            model_xencoder.train()
            model_aencoder.train()

            encoded_x_hat = model_xencoder(x_hat.detach())
            encoded_a_hat = model_aencoder(a_hat.detach())

            loss_DS = criterion_gan(model_DS(encoded_a.detach()), fake) \
                      + criterion_gan(model_DS(encoded_x.detach()), valid) \
                      + lamb_mse * criterion_gan(model_DS(encoded_a_hat.detach()), fake) \
                      + lamb_mse * criterion_gan(model_DS(encoded_x_hat.detach()), valid)

            optimizer_DS.zero_grad()
            loss_DS.backward()
            optimizer_DS.step()

            # ########################## E(Sa,Sx) ############################ #
            # update the encoders in the alignment-GAN                         #
            # ################################################################ #
            encoded_a = model_aencoder(a)

            loss_ES = criterion_gan(model_DS(encoded_a), valid) \
                      + lamb_mse * criterion_gan(model_DS(encoded_a_hat), valid)

            optimizer_ES.zero_grad()
            (loss_ES).backward()
            optimizer_ES.step()

            ##########################
            #  train the classifier  #
            ##########################
            x_feature = model_xencoder(x_open)
            x_hat_feature = model_xencoder(x_hat.detach())
            loss_I = (2 - args.lamb_sem) * (
                        criterion(model_I2(x_feature), attrs) + lamb_mse * criterion(model_I2(x_hat_feature),
                                                                                     attrs)) / 2 \
                     + args.lamb_sem * (criterion_sem(model_I2_sem(x_feature), sems) + lamb_mse * criterion_sem(
                model_I2_sem(x_hat_feature), sems)) / 2

            optimizer_X.zero_grad()
            loss_I.backward()
            optimizer_X.step()

            #######################################
            #  regularising the attribute branch  #
            #######################################
            a_feature = model_aencoder(a_open)
            a_hat_feature = model_aencoder(a_hat.detach())
            optimizer_A.zero_grad()
            loss_G = (2 - args.lamb_sem) * (
                        criterion(model_I2(a_feature), attrs) + lamb_mse * criterion(model_I2(a_hat_feature),
                                                                                     attrs)) / 2 \
                     + args.lamb_sem * (criterion_sem(model_I2_sem(a_feature), sems) + lamb_mse * criterion_sem(
                model_I2_sem(a_hat_feature), sems)) / 2
            loss_G.backward()
            optimizer_A.step()

            # ##########################end of updating process############################ #
            batch_time.update(time.time() - end)

            losses_I.update(loss_I.item(), sems.size(0))
            losses_G.update(loss_G.item(), sems.size(0))
            losses_DC.update(loss_DC.item(), sems.size(0))
            losses_GX.update(loss_GX.item(), sems.size(0))
            losses_GA.update(loss_GA.item(), sems.size(0))
            losses_CYCLE.update(loss_cycle.item(), sems.size(0))
            losses_DS.update(loss_DS.item(), sems.size(0))
            losses_ES.update(loss_ES.item(), sems.size(0))

            del loss_I
            del loss_G
            del loss_GC
            del loss_DC
            del loss_DS
            del loss_ES

            if (batch_idx + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                      'Loss i {loss_i.val:.4f}+g {loss_g.val:.4f}'
                      '+dc {loss_dc.val:.4f}+gx {loss_gx.val:.4f}'
                      '+ga {loss_ga.val:.4f}+cycle {loss_cycle.val:.4f}'
                      '+ds {loss_ds.val:.4f}+es {loss_es.val:.4f}'
                      '(i {loss_i.avg:.4f}+g {loss_g.avg:.4f}'
                      '+dc {loss_dc.avg:.4f}+gx {loss_gx.avg:.4f}'
                      '+ga {loss_ga.avg:.4f}+cycle {loss_cycle.avg:.4f}'
                      '+ds {loss_ds.avg:.4f}+es {loss_es.avg:.4f})\t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss_i=losses_I, loss_g=losses_G, loss_dc=losses_DC, loss_gx=losses_GX,
                    loss_ga=losses_GA, loss_cycle=losses_CYCLE, loss_ds=losses_DS, loss_es=losses_ES))

            end = time.time()
            torch.cuda.empty_cache()


# ################################## generate sampled attributes ############################# #
def fake_attributes(w, average, batch_size):
    dataset = args.dataset[0]
    fake_attrs = None
    if dataset == 'Market-1501':
        # first 4 as a group, next 9 binary, next 8 as a group, next 9 as a group
        w = np.array(w)
        if args.prob == 'inverse':
            w[w == 0] = 0.99999
            w[w == 1] = 0.00001
            w_bin = w[4:13]
            prob = 1 / w_bin / (1 / w_bin + 1 / (1 - w_bin))  # generate infrequent attributes
            prob_binary = torch.FloatTensor(prob).unsqueeze(0).repeat(batch_size, 1)
            binary_attrs = torch.bernoulli(prob_binary)
            age_prob = 1 / torch.tensor(w[:4])
            age_dist = torch.distributions.categorical.Categorical(age_prob / age_prob.sum())
            age = age_dist.sample(sample_shape=(batch_size,)).long()
            up_prob = 1 / torch.tensor(w[13:21])
            up_dist = torch.distributions.categorical.Categorical(up_prob / up_prob.sum())
            up = up_dist.sample(sample_shape=(batch_size,)).long()
            down_prob = 1 / torch.tensor(w[21:])
            down_dist = torch.distributions.categorical.Categorical(down_prob / down_prob.sum())
            down = down_dist.sample(sample_shape=(batch_size,)).long()
        elif args.prob == 'prior':
            w_bin = w[4:13]
            bin_prob = torch.tensor(w_bin)  # generate infrequent attributes
            # prob_binary = torch.FloatTensor(prob).unsqueeze(0).repeat(batch_size, 1)
            # binary_attrs = torch.bernoulli(prob_binary)
            age_prob = torch.tensor(w[:4])
            age_dist = torch.distributions.categorical.Categorical(age_prob / age_prob.sum())
            age = age_dist.sample(sample_shape=(batch_size,)).long()
            up_prob = torch.tensor(w[13:21])
            up_dist = torch.distributions.categorical.Categorical(up_prob / up_prob.sum())
            up = up_dist.sample(sample_shape=(batch_size,)).long()
            down_prob = torch.tensor(w[21:])
            down_dist = torch.distributions.categorical.Categorical(down_prob / down_prob.sum())
            down = down_dist.sample(sample_shape=(batch_size,)).long()
        bin_list = []
        for i in range(batch_size):
            bin_list.append(torch.multinomial(bin_prob / bin_prob.sum(), int(average - 2)).unsqueeze(0))
        bin = torch.cat(bin_list, 0).long()
        bin_vector = torch.zeros(len(bin), len(w_bin)).scatter_(1, bin, 1.)
        age_one_hot = torch.zeros(len(age), 4).scatter_(1, age.unsqueeze(1), 1.)
        up_one_hot = torch.zeros(len(up), 8).scatter_(1, up.unsqueeze(1), 1.)
        down_one_hot = torch.zeros(len(down), 9).scatter_(1, down.unsqueeze(1), 1.)
        fake_attrs = torch.cat((age_one_hot, bin_vector, up_one_hot, down_one_hot), 1)
    elif dataset == 'PETA':
        # first 8 are binary, next 7 as a group, next 8 as a group
        w = np.array(w)
        if args.prob=='inverse':
            w[w == 0] = 0.99999
            w[w == 1] = 0.00001
            w_bin = np.concatenate((w[4:35], w[79:]))
            bin_prob = 1 / torch.tensor(w_bin)
            age_prob = 1 / torch.tensor(w[0:4])
            up_prob = 1 / torch.tensor(w[35:46])
            down_prob = 1 / torch.tensor(w[46:57])
            hair_prob = 1 / torch.tensor(w[57:68])
            foot_prob = 1 / torch.tensor(w[68:79])
        elif args.prob=='equal':
            w[w == 0] = 0.99999
            w[w == 1] = 0.00001
            w_bin = np.concatenate((w[4:35], w[79:]))
            bin_prob = torch.ones_like(torch.tensor(w_bin))
            age_prob = torch.ones_like(torch.tensor(w[0:4]))
            up_prob = torch.ones_like(torch.tensor(w[35:46]))
            down_prob = torch.ones_like(torch.tensor(w[46:57]))
            hair_prob = torch.ones_like(torch.tensor(w[57:68]))
            foot_prob = torch.ones_like(torch.tensor(w[68:79]))
        elif args.prob=='prior':
            w_bin = np.concatenate((w[4:35], w[79:]))
            bin_prob = torch.tensor(w_bin)
            age_prob = torch.tensor(w[0:4])
            up_prob = torch.tensor(w[35:46])
            down_prob = torch.tensor(w[46:57])
            hair_prob = torch.tensor(w[57:68])
            foot_prob = torch.tensor(w[68:79])

        bin_list = []
        for i in range(batch_size):
            bin_list.append(torch.multinomial(bin_prob / bin_prob.sum(), int(average-5)).unsqueeze(0))
        bin = torch.cat(bin_list, 0).long()


        # actually no need to divide by the sum of probabilities.

        age_dist = torch.distributions.categorical.Categorical(age_prob / age_prob.sum())
        age = age_dist.sample(sample_shape=(batch_size,)).long()


        up_dist = torch.distributions.categorical.Categorical(up_prob / up_prob.sum())
        up = up_dist.sample(sample_shape=(batch_size,)).long()


        # actually no need to divide by the sum of probabilities.
        down_dist = torch.distributions.categorical.Categorical(down_prob / down_prob.sum())
        down = down_dist.sample(sample_shape=(batch_size,)).long()


        # actually no need to divide by the sum of probabilities.
        hair_dist = torch.distributions.categorical.Categorical(hair_prob / hair_prob.sum())
        hair = hair_dist.sample(sample_shape=(batch_size,)).long()


        # actually no need to divide by the sum of probabilities.
        foot_dist = torch.distributions.categorical.Categorical(foot_prob / foot_prob.sum())
        foot = foot_dist.sample(sample_shape=(batch_size,)).long()

        bin_vector = torch.zeros(len(bin), len(w_bin)).scatter_(1, bin, 1.)
        age_one_hot = torch.zeros(len(age), 4).scatter_(1, age.unsqueeze(1), 1.)
        down_one_hot = torch.zeros(len(down), 11).scatter_(1, down.unsqueeze(1), 1.)
        up_one_hot = torch.zeros(len(up), 11).scatter_(1, up.unsqueeze(1), 1.)
        hair_one_hot = torch.zeros(len(hair), 11).scatter_(1, hair.unsqueeze(1), 1.)
        foot_one_hot = torch.zeros(len(foot), 11).scatter_(1, foot.unsqueeze(1), 1.)

        fake_attrs = torch.cat((age_one_hot,bin_vector[:,:31], up_one_hot, down_one_hot, hair_one_hot, foot_one_hot, bin_vector[:,31:]), 1)

    return fake_attrs


def test_WOadv(model_I1, model_G, model_xencoder, model_aencoder, queryloader, galleryloader, use_gpu,
               ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_xencoder.eval()
    model_aencoder.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            features = model_aencoder(model_G(attrs))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()
            features = model_xencoder(model_I1(imgs))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]

#################### The following test functions are only used for reference ########################

def test_Image_space(model_I1, model_G, model_GX, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20],
                     return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_GX.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            noise = torch.randn(attrs.size(0), args.noise)

            a = model_G(attrs)

            if use_gpu:
                noise = noise.cuda()
            # add noise to image feature generator
            features = model_GX(torch.cat((a, noise), 1).detach())

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()
            features = model_I1(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


def test_attribute_space(model_I1, model_G, model_GA, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20],
                         return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_GA.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            features = model_G(attrs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()
            features = model_GA(model_I1(imgs))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


def test_2way(model_I1, model_G, model_xencoder, model_aencoder, model_GX, model_GA, queryloader, galleryloader,
              use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_xencoder.eval()
    model_aencoder.eval()
    model_GA.eval()
    model_GX.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            features1 = model_aencoder(model_G(attrs))

            noise = torch.randn(attrs.size(0), args.noise)

            a = model_G(attrs)

            if use_gpu:
                noise = noise.cuda()
            # add noise to image feature generator
            x_hat = model_GX(torch.cat((a, noise), 1).detach())
            features2 = model_xencoder(x_hat)

            features = features1 + features2

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()
            features1 = model_xencoder(model_I1(imgs))
            features2 = model_aencoder(model_GA(model_I1(imgs)))

            features = features1 + features2

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


def test_1way(model_I1, model_G, model_xencoder, model_aencoder, model_GA, queryloader, galleryloader, use_gpu,
              ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_xencoder.eval()
    model_aencoder.eval()
    model_GA.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            features = model_aencoder(model_G(attrs))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()

            features = model_aencoder(model_GA(model_I1(imgs)))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


def test_1way_alt(model_I1, model_G, model_xencoder, model_aencoder, model_GA, model_GX, queryloader, galleryloader,
                  use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()
    model_xencoder.eval()
    model_aencoder.eval()
    model_GA.eval()
    model_GX.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (_, attrs, ids, camids, names, sems) in enumerate(queryloader):
            if use_gpu: attrs = attrs.cuda()

            attrs = attrs.float()

            end = time.time()

            noise = torch.randn(attrs.size(0), args.noise)

            a = model_G(attrs)

            if use_gpu:
                noise = noise.cuda()
            # add noise to image feature generator
            x_hat = model_GX(torch.cat((a, noise), 1).detach())

            features = model_xencoder(x_hat)

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            qf.append(features)
            q_pids.extend(sems)
            q_camids.extend(camids)

        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, i, label, id, cam, name, sem
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()

            end = time.time()

            features = model_xencoder(model_I1(imgs))

            batch_time.update(time.time() - end)

            features = features.data.cpu()

            gf.append(features)
            g_pids.extend(sems)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)

    ########################################
    # change euclidean dist to cosine dist #
    ########################################
    qf_norm = qf / (qf.norm(dim=1).expand_as(qf.t()).t())
    gf_norm = gf / (gf.norm(dim=1).expand_as(gf.t()).t())
    distmat = qf_norm.mm(gf_norm.t())
    distmat = -distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


def test_attr(model_I1, model_I2, model_G, galleryloader, use_gpu):
    batch_time = AverageMeter()

    model_G.eval()
    model_I1.eval()

    with torch.no_grad():
        gf, g_pids, g_camids = [], [], []
        end = time.time()
        # data, label, id, camid, img_path, sem
        tp_i = None  # true positive for image network
        pp_i = None  # positive prediction
        tp_g = None  # true positive for attribute network
        pp_g = None  # positive prediction
        for batch_idx, (imgs, attrs, id, camids, name, sems) in enumerate(galleryloader):
            ones = torch.ones(attrs.size())
            zeros = torch.zeros(attrs.size())
            if use_gpu:
                sems = sems.cuda()
                imgs = imgs.cuda()
                attrs = attrs.cuda()
                ones = ones.cuda()
                zeros = zeros.cuda()

            batch_tp_i = zeros.clone()
            batch_pp_i = zeros.clone()
            batch_tp_g = zeros.clone()
            batch_pp_g = zeros.clone()

            end = time.time()

            attrs = attrs.float()

            ###################################
            # calculate map for image network #
            ###################################

            image_predictions = model_I2(model_I1(imgs))
            # print('image_outputs size: ',image_outputs.size())
            # image_predictions = torch.argmax(image_outputs, dim=2)
            image_predictions = torch.where(image_predictions > 0, ones, zeros)
            image_results = image_predictions - attrs
            image_results = torch.where(image_results == 0, ones, zeros)
            image_predictions = torch.where(image_predictions == 1, ones, zeros)
            index = torch.where(image_results + image_predictions == 2, ones, zeros)
            batch_tp_i += index
            batch_pp_i += image_predictions
            if tp_i is not None:
                tp_i += batch_tp_i.sum(dim=0)
                pp_i += batch_pp_i.sum(dim=0)
            else:
                tp_i = batch_tp_i.sum(dim=0)
                pp_i = batch_pp_i.sum(dim=0)

            #######################################
            # calculate map for attribute network #
            #######################################

            attr_predictions = model_I2(model_G(attrs))
            attr_predictions = torch.where(attr_predictions > 0, ones, zeros)
            attr_results = attr_predictions - attrs
            attr_results = torch.where(attr_results == 0, ones, zeros)
            attr_predictions = torch.where(attr_predictions == 1, ones, zeros)
            index = torch.where(attr_results + attr_predictions == 2, ones, zeros)
            batch_tp_g += index
            batch_pp_g += attr_predictions
            if tp_g is not None:
                tp_g += batch_tp_g.sum(dim=0)
                pp_g += batch_pp_g.sum(dim=0)
            else:
                tp_g = batch_tp_g.sum(dim=0)
                pp_g = batch_pp_g.sum(dim=0)

        pp_i = torch.add(pp_i, 1e-10)
        pp_g = torch.add(pp_g, 1e-10)

        map_i = torch.div(tp_i, pp_i).mean()
        map_g = torch.div(tp_g, pp_g).mean()
        print('mean average precision of image network predictions: ', map_i)
        print('mean average precision of attribute network predictions: ', map_g)

    return


if __name__ == '__main__':
    main()
