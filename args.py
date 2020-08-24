import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    '''
    change the default root to dataset
    '''
    parser.add_argument('--root', type=str, default='dataset',
                        help="root path to data directory")
    parser.add_argument('-s', '--dataset', type=str, required=True, nargs='+',
                        help="dataset")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")
    parser.add_argument('--train-sampler', type=str, default='',
                        help="sampler for trainloader")
    parser.add_argument('--size-embed', type=int, default=128,
                        help="embedding size of concepts")
    parser.add_argument('--noise', type=int, default=400,
                        help="dimension of noise")
    parser.add_argument('--loss', type=str, default='l2',
                        help="type l2 or l1")
    # ************************************************************
    # fake settings
    # ************************************************************
    parser.add_argument('--num-train', type=int, default=15,
                        help="number of training semantic ids")
    parser.add_argument('--epoch-add-fake', default=0, type=int,
                        help="the epoch to start adding sampled attributes")
    parser.add_argument('--ssl-loss-ratio', default=1.0, type=float,
                        help="the ratio of ssl loss")
    parser.add_argument('--num-fake', default=128, type=int,
                        help="Number of fake attributes")
    parser.add_argument('--test-attr', action='store_true',
                        help="make use of the test-attr")
    parser.add_argument('--start-mse', default=0, type=int,
                        help="the epoch to start adding consistent loss")
    parser.add_argument('--prob', default='prior', type=str,
                        help="generate random fake attributes according to prior probability or randomly")
    parser.add_argument('--no-save', action='store_true',
                        help="do not save checkpoints")
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr-jt', default=0.001, type=float,
                        help="initial learning rate for training jointly")
    parser.add_argument('--lr-pt', default=0.00003, type=float,
                        help="initial learning rate for pre-training the image network")
    parser.add_argument('--lr-al', default=0.0001, type=float,
                        help="initial learning rate for training with adversarial learning")
    parser.add_argument('--lr-sc', default=0.01, type=float,
                        help="initial learning rate for optimising the semantic consistency loss")
    parser.add_argument('--lr-gf', default=0.0001, type=float,
                        help="initial learning rate for generating fake features")
    parser.add_argument('--lr-df', default=0.0001, type=float,
                        help="initial learning rate for discriminating fake features")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay")
    parser.add_argument('--ssl-ratio', default=0.2, type=float,
                        help="ratio for ssl")
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="momentum factor for sgd and rmsprop")
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help="sgd's dampening for momentum")
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help="whether to enable sgd's Nesterov momentum")
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help="rmsprop's smoothing constant")
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help="exponential decay rate for adam's first moment")
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help="exponential decay rate for adam's second moment")
    
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch-pt', default=100, type=int,
                        help="maximum epochs to run for single pretraining")
    parser.add_argument('--max-epoch-jt', default=100, type=int,
                        help="maximum epochs to run for joint pretraining")
    parser.add_argument('--max-epoch-al', default=60, type=int,
                        help="maximum epochs to run for SAL")
    parser.add_argument('--start-epoch-jt', default=0, type=int,
                        help="manual epoch number for jointly training (useful when restart)")
    parser.add_argument('--start-epoch-pt', default=0, type=int,
                        help="manual epoch number for pre-training (useful when restart)")
    parser.add_argument('--start-epoch-al', default=0, type=int,
                        help="manual epoch number for adversarial learning (useful when restart)")
    parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                        help="stepsize to decay learning rate for pretraining")
    parser.add_argument('--stepsize-sal', default=[20, 40], nargs='+', type=int,
                        help="stepsize to decay learning rate for sal")
    parser.add_argument('--gamma', default=0.5, type=float,
                        help="learning rate decay rate")
    parser.add_argument('--gamma-sc', default=0.5, type=float,
                        help="learning rate decay for attribute branch")

    parser.add_argument('--lamb-gx', default=1, type=float,
                        help="weight for generator loss")

    parser.add_argument('--lamb-dx', default=1, type=float,
                        help="weight for discriminator loss")

    parser.add_argument('--lamb-ga', default=1, type=float,
                        help="weight for generator loss")

    parser.add_argument('--lamb-da', default=1, type=float,
                        help="weight for discriminator loss")

    parser.add_argument('--lamb-mse', default=1, type=float,
                        help="weight for distance loss")

    parser.add_argument('--objective', default='xent', type=str,
                        help="set the objective")

    parser.add_argument('--train-batch-size', default=128, type=int,
                        help="training batch size")
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help="test batch size")
    
    parser.add_argument('--always-fixbase', action='store_true',
                        help="always fix base network and only train specified layers")
    parser.add_argument('--fixbase-epoch', type=int, default=0,
                        help="how many epochs to fix base network (only train randomly initialized classifier)")
    parser.add_argument('--open-layers', type=str, nargs='+', default=['classifier'],
                        help="open specified layers for training while keeping others frozen")

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help="use label smoothing regularizer in cross entropy loss")

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('--htri-only', action='store_true',
                        help="only use hard triplet loss")
    parser.add_argument('--lambda-xent', type=float, default=1,
                        help="weight to balance cross entropy loss")
    parser.add_argument('--lambda-htri', type=float, default=1,
                        help="weight to balance hard triplet loss")
    parser.add_argument('--lamb-attended', type=float, default=1,
                        help="weight to attended loss")
    parser.add_argument('--lamb-sem', type=float, default=1,
                        help="weight to semantic loss")
    
    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help="load pretrained weights but ignore layers that don't match in size")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluate only")
    parser.add_argument('--evaluate-train', action='store_true',
                        help="evaluate only (on training set)")
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help="evaluation frequency (set to -1 to test only in the end)")
    parser.add_argument('--start-eval', type=int, default=0,
                        help="start to evaluate after a specific epoch")
    parser.add_argument('--attribute-prediction', action='store_true',
                        help="evaluate the attribute prediction results")
    
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=10,
                        help="print frequency")
    parser.add_argument('--seed', type=int, default=1,
                        help="manual seed")
    parser.add_argument('--resume-i1', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-i2', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-i2-sem', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-g', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-d', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-ii', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-a', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-x', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-gx', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-dx', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-ga', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-da', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-ds', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--resume-dc', type=str, default='', metavar='PATH',
                        help="resume from a checkpoint")
    parser.add_argument('--folder', type=str, default='', metavar='PATH',
                        help="resume from a folder")
    parser.add_argument('--resume-epoch', type=str, default='', metavar='PATH',
                        help="resume from some epoch")
    parser.add_argument('--resume-pt', type=bool, default=False, metavar='PATH',
                        help="resume from pt or not")
    parser.add_argument('--resume-stage', type=str, default='pt',
                        help="resume from which stage")
    parser.add_argument('--save-pt', type=int, default=-1, metavar='PATH',
                        help="the frequency of saving the pretraining model")
    parser.add_argument('--save-dir', type=str, default='log',
                        help="path to save log and model weights")
    parser.add_argument('--use-cpu', action='store_true',
                        help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help="use available gpus instead of specified devices (useful when using managed clusters)")
    parser.add_argument('--visualize-ranks', action='store_true',
                        help="visualize ranked results, only available in evaluation mode")

    return parser


def image_dataset_kwargs(parsed_args):

    return {
        'source_names': parsed_args.dataset,
        'target_names': parsed_args.dataset,
        'root': parsed_args.root,
        'seed': parsed_args.seed,
        'num_train': parsed_args.num_train,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,
    }

def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_jt,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }

def optimizer_kwargs_pt(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_pt,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }

def optimizer_kwargs_sc(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_sc,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }



def optimizer_kwargs_al(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_al,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }

def optimizer_kwargs_gf(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_gf,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }


def optimizer_kwargs_df(parsed_args):
    """
    Build kwargs for optimizer in optimizer.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr_df,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2
    }