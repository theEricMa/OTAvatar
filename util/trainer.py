import random
import importlib
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler, SGD, AdamW

from util.distributed import master_only_print as print
from util.init_weight import weights_init

import os

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_trainer_4_warping_n_inversion(opt, net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_ema, opt_G, sch_G, train_dataset):
    module, trainer_name = opt.trainer.type.split('::')

    trainer_lib = importlib.import_module(module)
    trainer_class = getattr(trainer_lib, trainer_name)
    trainer = trainer_class(opt, net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_ema, opt_G, sch_G, train_dataset)
    return trainer

def get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset):
    module, trainer_name = opt.trainer.type.split('::')

    trainer_lib = importlib.import_module(module)
    trainer_class = getattr(trainer_lib, trainer_name)
    trainer = trainer_class(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset)
    return trainer

def gen_model_optimizer_4_warping_n_inversion(opt):
    # initialize warping module
    warp_module, warp_network_name = opt.warp.type.split('::')
    lib = importlib.import_module(warp_module)
    network = getattr(lib, warp_network_name)
    net_Warp = network(**opt.warp.param).to(opt.device)
    net_Warp_ema = network(**opt.warp.param).to(opt.device)
    net_Warp_ema.eval()

    if not hasattr(opt.warp, 'checkpoint') or opt.warp.checkpoint is None:
        # initialize net_Warp 
        init_bias = getattr(opt.trainer.init, 'bias', None)
        net_Warp.apply(
            weights_init(
            opt.trainer.init.type, opt.trainer.init.gain, init_bias
            ))
        print('net_Warp is initialized')
        # accumulate net_Warp_ema weights
        accumulate(net_Warp_ema, net_Warp, 0)
    else:
        # load net_Warp weights
        assert os.path.exists(opt.warp.checkpoint)
        checkpoint = torch.load(opt.warp.checkpoint, map_location=torch.device('cpu'))
        net_Warp.load_state_dict(checkpoint['net_Warp_ema'], strict=False)
        print('net_Warp pretrained model is loaded')
        # identically load net_Warp_ema weights
        net_Warp_ema.load_state_dict(checkpoint['net_Warp_ema'], strict=False)

    print('net [{}] parameter count: {:,}'.format(
        'Warp', _calculate_model_size(net_Warp)))    
    print('Initialize net_Warp weights using '
          'type: {} gain: {}'.format(opt.trainer.init.type,
                                     opt.trainer.init.gain))

    # set trainable parameters in opt_Warp
    if hasattr(opt.warp_optimizer,'refine_only') and opt.warp_optimizer.refine_only:
        assert opt.warp.param.use_refine # the net_Warp should have refine module
        net_Warp_trainable = [v for k,v in net_Warp.named_parameters() if 'refine_net' in k]
    else:
        net_Warp_trainable = net_Warp

    opt_Warp = get_optimizer(
        opt.warp_optimizer, 
        net_Warp_trainable)
    sch_Warp = get_scheduler(opt.warp_optimizer, opt_Warp)

    if opt.distributed:
        net_Warp = nn.parallel.DistributedDataParallel(
            net_Warp,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        net_Warp_ema = nn.parallel.DistributedDataParallel(
            net_Warp_ema,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # generation model
    gen_module, gen_network_name = opt.gen.type.split('::')
    lib = importlib.import_module(gen_module)
    network = getattr(lib, gen_network_name)

    # Initializing net_G
    net_G = network(**opt.gen.param).eval().to(opt.device)
    checkpoint = torch.load(opt.gen.checkpoint, map_location=torch.device('cpu'))
    net_G.load_state_dict(checkpoint['G_ema'], strict=False)
    print('net [{}] parameter count: {:,}'.format(
        'G', _calculate_model_size(net_G))) 

    if hasattr(opt, 'gen_optimizer'): # the generator will be finetuned
        # set trainable parameters in opt_G
        if hasattr(opt.gen_optimizer,'sr_only') and opt.gen_optimizer.sr_only:
            net_G_trainable = [v for k,v in net_G.named_parameters() if 'superresolution' in k]
        else:
            net_G_trainable = net_G

        opt_G = get_optimizer(opt.gen_optimizer, net_G_trainable) #[param for name, param in net_G.named_parameters() if 'superresolution' not in name])
        sch_G = get_scheduler(opt.gen_optimizer, opt_G)

        # therefor net_G_ema is needed
        net_G_ema = network(**opt.gen.param).eval().to(opt.device)
        net_G_ema.load_state_dict(checkpoint['G_ema'], strict=False)

    else:
        opt_G = None
        sch_G = None

        net_G_ema = None

    if opt.distributed:
        net_G = nn.parallel.DistributedDataParallel(
            net_G,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False
        )
        
        # maybe this would work
        net_G_ema = nn.parallel.DistributedDataParallel(
            net_G_ema,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    return net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_ema, opt_G, sch_G



def get_model_optimizer_and_scheduler(opt):
    gen_module, gen_network_name = opt.gen.type.split('::')
    lib = importlib.import_module(gen_module)
    network = getattr(lib, gen_network_name)
    net_G = network(**opt.gen.param).to(opt.device)
    init_bias = getattr(opt.trainer.init, 'bias', None)
    net_G.apply(weights_init(
        opt.trainer.init.type, opt.trainer.init.gain, init_bias))

    net_G_ema = network(**opt.gen.param).to(opt.device)
    net_G_ema.eval()
    accumulate(net_G_ema, net_G, 0)
    print('net [{}] parameter count: {:,}'.format(
        'net_G', _calculate_model_size(net_G)))
    print('Initialize net_G weights using '
          'type: {} gain: {}'.format(opt.trainer.init.type,
                                     opt.trainer.init.gain))

    opt_G = get_optimizer(opt.gen_optimizer, net_G)

    if opt.distributed:
        net_G = nn.parallel.DistributedDataParallel(
            net_G,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )


    # Scheduler
    sch_G = get_scheduler(opt.gen_optimizer, opt_G)
    return net_G, net_G_ema, opt_G, sch_G


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(opt_opt, opt):
    """Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """

    if opt_opt.lr_policy.type == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=opt_opt.lr_policy.step_size,
            gamma=opt_opt.lr_policy.gamma)
    elif opt_opt.lr_policy.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                format(opt_opt.lr_policy.type))
    return scheduler

def get_optimizer(opt_opt, net, lr_reduction = 1.0):
    if isinstance(net, torch.nn.Module):
        return get_optimizer_for_params(opt_opt, net.parameters(), lr_reduction)
    else:
        return get_optimizer_for_params(opt_opt, net, lr_reduction)


def get_optimizer_for_params(opt_opt, params, lr_reduction = 1):
    r"""Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        params (obj): Parameters to be trained by the parameters.

    Returns:
        (obj): Optimizer
    """
    # We will use fuse optimizers by default.
    if opt_opt.type == 'adam':
        opt = Adam(params,
                   lr=opt_opt.lr * lr_reduction,
                   betas=(opt_opt.adam_beta1, opt_opt.adam_beta2),
                   weight_decay=opt_opt.weight_decay if hasattr(opt_opt, 'weight_decay') else 0,
                   )
    elif opt_opt.type == 'sgd':
        opt = SGD(params, 
                  lr = opt_opt.lr * lr_reduction,
                  weight_decay=opt_opt.weight_decay if hasattr(opt_opt, 'weight_decay') else 0,)
    elif opt_opt.type == 'adamw':
        opt = AdamW(params,
                    lr=opt_opt.lr * lr_reduction,
                    betas=(opt_opt.adam_beta1, opt_opt.adam_beta2),
                    weight_decay=opt_opt.weight_decay if hasattr(opt_opt, 'weight_decay') else 0,
                    )
    else:
        raise NotImplementedError(
            'Optimizer {} is not yet implemented.'.format(opt_opt.type))
    return opt


