import os
import math
from re import sub
from time import monotonic
from turtle import update

from traitlets import HasTraits

from util.distributed import get_rank
import torch
from torch.nn import functional as F

from trainers.base import BaseTrainer
from util.trainer import accumulate, get_optimizer
from util.camera_utils import LookAtPoseSampler
from util.misc import to_cuda
from loss.perceptual  import PerceptualLoss

from util.distributed import is_master
from loss.local import LocalLoss
from loss.identity import IDLoss
from tqdm import tqdm
import cv2, numpy as np
from loss.local import lmk2mask
import random

class FaceTrainer(BaseTrainer):
    r"""Initialize lambda model trainer.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, opt, net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_ema, opt_G, sch_G,
                 train_data_loader, val_data_loader=None):
        super(FaceTrainer, self).__init__(opt, net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_ema, opt_G, sch_G, train_data_loader, val_data_loader)
        self.accum_Warp = self.opt.trainer.accum_ratio.Warp
        self.accum_G = self.opt.trainer.accum_ratio.G


        self.log_size = int(math.log(opt.data.resolution, 2))

        self._net_Warp_ema = self.net_Warp_ema
        self.net_Warp_ema = self.net_Warp_ema.module
        if self.net_G_ema is None :
            self.net_G_ema = self.net_G_module
            self.train_G = False
            print('We do not train Generator')
        else:
            assert self.opt_G is not None
            assert self.sch_G is not None
            self._net_G_ema = self.net_G_ema
            self.net_G_ema = self.net_G_ema.module
            self.train_G = True
            print('We train Generator')

        self.ws_stdv = torch.ones_like(torch.from_numpy(np.load('pretrained/ws_std.npy')))

    
    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_inverse_lr',
            PerceptualLoss(
                network=opt.trainer.vgg_param_lr.network,
                layers=opt.trainer.vgg_param_lr.layers,
                weights=getattr(opt.trainer.vgg_param_lr, 'weights', None),
                num_scales=getattr(opt.trainer.vgg_param_lr, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_lr, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_lr, 'style_to_perceptual', 0)
                ).to('cuda:{}'.format(get_rank())),
            opt.trainer.loss_weight.inverse)

        self._assign_criteria(
            'perceptual_inverse_sr',
            PerceptualLoss(
                network=opt.trainer.vgg_param_sr.network,
                layers=opt.trainer.vgg_param_sr.layers,
                weights=getattr(opt.trainer.vgg_param_sr, 'weights', None),
                num_scales=getattr(opt.trainer.vgg_param_sr, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_sr, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_sr, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.inverse)

        self._assign_criteria(
            'perceptual_refine_lr',
            PerceptualLoss(
                network=opt.trainer.vgg_param_lr.network,
                layers=opt.trainer.vgg_param_lr.layers,
                weights=getattr(opt.trainer.vgg_param_lr, 'weights', None),
                num_scales=getattr(opt.trainer.vgg_param_lr, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_lr, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_lr, 'style_to_perceptual', 0)
                ).to('cuda:{}'.format(get_rank())),
            opt.trainer.loss_weight.refine)

        self._assign_criteria(
            'perceptual_refine_sr',
            PerceptualLoss(
                network=opt.trainer.vgg_param_sr.network,
                layers=opt.trainer.vgg_param_sr.layers,
                weights=getattr(opt.trainer.vgg_param_sr, 'weights', None),
                num_scales=getattr(opt.trainer.vgg_param_sr, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_sr, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_sr, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.refine)            



        self._assign_criteria(
            'monotonic',
            self.monotonic_loss,
            opt.trainer.loss_weight.monotonic
        )
        self._assign_criteria(
            'TV',
            self.TV_loss,
            opt.trainer.loss_weight.TV
        )


        self._assign_criteria(
            'pixel',
            torch.nn.SmoothL1Loss(),
            opt.trainer.loss_weight.pixel
        )

        self._assign_criteria(
            'a_norm',
            lambda x: torch.relu(10 - torch.mean(x.norm(dim = 1))),
            opt.trainer.loss_weight.a_norm
        )

        self._assign_criteria(
            'a_mutual',
            lambda x, y: torch.relu(10 - torch.mean((x - y).norm(dim = 1))),
            opt.trainer.loss_weight.a_mutual
        )

        # the following should be treat differently for target and source data
        self._assign_criteria(
            'local',
            LocalLoss(),
            opt.trainer.loss_weight.local
        )

        self._assign_criteria(
            'local_s',
            LocalLoss(),
            opt.trainer.loss_weight.local
        )

        self._assign_criteria(
            'id',
            IDLoss().to('cuda'),
            opt.trainer.loss_weight.id
        )

        self._assign_criteria(
            'id_s',
            IDLoss().to('cuda'),
            opt.trainer.loss_weight.id
        )

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def _percept_loss(self, predict_images, target_images, percept_fn, same_size_as_pred = True, target_keypoints = None, mask_rate = 1):
        if same_size_as_pred:
            _, _, Hp, Wp = predict_images.shape
            if target_images.shape[2] != Hp or target_images.shape[3] != Wp:
                target_images = F.interpolate(target_images, size = (Hp, Wp), mode='area')
            if target_keypoints is not None:

                mask = lmk2mask(target_images, target_keypoints * Hp / self.opt.data.resolution, mask_rate)
                percept_loss = percept_fn(predict_images, target_images, mask)
            else:
                percept_loss =  percept_fn(predict_images, target_images)
        else:
            _, _, H, W = target_images.shape
            if predict_images.shape[2] != H or predict_images.shape[3] != W:
                predict_images = F.interpolate(predict_images, size = (H, W), mode = 'area')
            if target_keypoints is not None:
                mask = lmk2mask(target_images, target_keypoints * H / self.opt.data.resolution, mask_rate)
                percept_loss = percept_fn(predict_images, target_images, mask)
            else:
                percept_loss =  percept_fn(predict_images, target_images)

        return percept_loss

    def _local_loss(self, predict_images, target_images, target_keypoint, local_fn, same_size_as_pred = True, use_cache = False):
        if same_size_as_pred:
            _, _, Hp, Wp = predict_images.shape
            assert Hp == Wp
            if target_images.shape[2] != Hp or target_images.shape[3] != Wp:
                target_images = F.interpolate(target_images, size = (Hp, Wp), mode='area')
            local_loss = local_fn(predict_images, target_images, target_keypoint * Hp / self.opt.data.resolution, use_cache = use_cache)
        else:
            _, _, H, W = target_images.shape
            assert H == W
            if predict_images.shape[2] != H or predict_images.shape[3] != W:
                predict_images = F.interpolate(predict_images, size = (H, W), mode = 'area')
            local_loss = local_fn(predict_images, target_images, target_keypoint * H / self.opt.data.resolution, use_cache = use_cache)
        return local_loss

    def monotonic_loss(self, ws, planes, reduction = 1):
        # monotonic loss
        initial_coordinates = (torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1) / reduction # Front
        perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.net_G_module.rendering_kwargs['box_warp'] # Behind
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.net_G_module.renderer.run_model(planes, self.net_G_module.decoder, all_coordinates, torch.randn_like(all_coordinates), self.net_G_module.rendering_kwargs)['sigma']
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]
        monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean()
        return monotonic_loss

    def TV_loss(self, ws, planes, reduction = 1):
        # monotonic loss
        initial_coordinates = (torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1) / reduction # Front
        perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.net_G_module.rendering_kwargs['density_reg_p_dist']
        all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
        sigma = self.net_G_module.renderer.run_model(planes, self.net_G_module.decoder, all_coordinates, torch.randn_like(all_coordinates), self.net_G_module.rendering_kwargs)['sigma']
        sigma_initial = sigma[:, :sigma.shape[1]//2]
        sigma_perturbed = sigma[:, sigma.shape[1]//2:]
        TV_loss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.net_G_module.rendering_kwargs['density_reg']
        return TV_loss

    def pixel_loss(self, predict_images, target_images, pixel_fn, same_size_as_pred = True):
        if same_size_as_pred:
            _, _, Hp, Wp = predict_images.shape
            if target_images.shape[2] != Hp or target_images.shape[3] != Wp:
                target_images = F.interpolate(target_images, size = (Hp, Wp), mode='area')
        else:
            _, _, H, W = target_images.shape
            if predict_images.shape[2] != H or predict_images.shape[3] != W:
                predict_images = F.interpolate(predict_images, size = (H, W), mode = 'area')
        pixel_loss = pixel_fn(predict_images, target_images)
        return pixel_loss        

    def clip(self, ws):
        w_avg = self.net_G_ema.backbone.mapping.w_avg.clone()
        w_sdv = self.ws_stdv.to(ws)
        return torch.clamp(ws, w_avg - 5 * w_sdv, w_avg + 5 * w_sdv)


    def net_G_optimize(self, target_images, target_semantic, target_condition, target_keypoint, opt_Ws, _w_opt,
        w_opt, use_ema = False, sr_iters = 100, intri_bias = None, trans_bias = None,
        **kwargs):
        from copy import deepcopy
        from lpips import LPIPS

        # collect hyper paramters
        lpips_thres = kwargs.get('lpips_thres', 0.06)
        pti_lr = kwargs.get('pti_lr', 3e-4) 
        max_iters = kwargs.get('max_iters', 400)
        lpips_fn = LPIPS(net = 'alex').cuda().eval()

        _net_G = deepcopy(self.net_G_ema)
        net_Warp = self.net_Warp_ema if use_ema else self.net_Warp_module
        _opt_G = torch.optim.Adam(_net_G.parameters(), lr = pti_lr)

        inversion_opt = self.opt.trainer.inversion
        for sub_iter in tqdm(range(max_iters)):

            do_sr = sub_iter >= (inversion_opt.iterations - sr_iters)

            # model forward
            ws_scaling, ws_trans, alpha = net_Warp(target_semantic)
            ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
            ws_trans = ws_trans * self.ws_stdv.to(w_opt)

            planes = _net_G.before_planes(self.clip(w_opt * ws_scaling + ws_trans), noise_mode='const')
            synth_dict = _net_G.render_from_planes(
                self.add_bias(target_condition, intri_bias, trans_bias), 
                planes, 
                neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
            predict_images = synth_dict['image']
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                predict_images, predict_feature = synth_dict['image'], synth_dict['image_feature']
                sr_dict = _net_G.sr(predict_images, predict_feature, self.clip(w_opt * ws_scaling + ws_trans))
                sr_images = sr_dict['image']            

            percept_loss = self._percept_loss(predict_images, target_images, percept_fn = lpips_fn.forward,)
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                percept_loss += self._percept_loss(sr_images, target_images, percept_fn = lpips_fn.forward,)

            # pixel loss
            pixel_loss = self.pixel_loss(predict_images, target_images, self.criteria['pixel'])
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                pixel_loss += self.pixel_loss(sr_images, target_images, self.criteria['pixel'])


            # model backward
            loss = 0. * percept_loss + \
                   1 * pixel_loss 

            _opt_G.zero_grad()
            loss.backward()
            _opt_G.step()

            def resize(image, target_images):
                _, _, H, W = target_images.shape
                if image.shape[2] != H or image.shape[3] != W:
                    image = F.interpolate(image, (H, W))
                return image

            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                images = resize(sr_images, target_images)
            else:
                images = resize(predict_images, target_images)

            with torch.no_grad():
                lpips_score = lpips_fn.forward(images, target_images).squeeze().detach().cpu().numpy()
            if lpips_score <= lpips_thres:
                break
            print(lpips_score)
        
        _net_G.eval()
        return _net_G, intri_bias, trans_bias,

    def add_bias(self, c, intri_bias = None, extri_bias = None):
        extrinsics = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25]
        if intri_bias is not None:
            s, t = torch.chunk(intri_bias, 2, dim=1)
            intrinsics_ = intrinsics.data * (1 + torch.tanh(s)) + t
        else:
            intrinsics_ = intrinsics.data

        if extri_bias is not None:
            extrinsics_ = extrinsics.data
            s, t = torch.chunk(extri_bias, 2, dim=1)
            extrinsics_[:, :3, :3] =  (extrinsics.data[:, :3, :3].reshape(-1, 9) * (1 + torch.tanh(s) ) + t).reshape(-1, 3, 3)
        else:
            extrinsics_ = extrinsics.data

        return torch.cat(
            [
                extrinsics_.view(-1, 16),
                intrinsics_
            ],
            dim = 1
        )


    def inverse_optimize(self, target_images, target_semantic, target_condition, target_keypoint, opt_Ws, _w_opt,
        w_std, use_ema = False, source_images = None, source_semantic = None, source_condition = None, source_keypoint = None,
        sr_iters = 100,  
        if_optimize_intrinsic = True, intri_bias = None,
        if_optmize_translation = False, trans_bias = None):
        B, _, H, W = target_images.shape
        inverse_losses = {
            'inverse_perceptual': [],
            'inverse_local': [],
            'inverse_monotonic':[],
            'inverse_TV': [],
            'inverse_pixel':[],
            'inverse_a_norm': [],
            'inverse_a_mutual': [],
            'inverse_id': [],
        }

        do_update_camera = False
        if hasattr(self.opt, 'camera_optimizer') and (if_optimize_intrinsic or if_optmize_translation):
            do_update_camera = True
            if if_optimize_intrinsic:
                intri_bias = intri_bias if intri_bias is not None else torch.cat([torch.zeros_like(target_condition[:, -9:])] * 2, dim = 1)
                intri_bias.requires_grad = True
                opt_Intri = get_optimizer(
                    opt_opt = self.opt.camera_optimizer,
                    net = [intri_bias]
                )
            if if_optmize_translation:
                trans_bias = trans_bias if trans_bias is not None else torch.cat([torch.zeros_like(target_condition[:, -9:])] * 2, dim = 1)
                trans_bias.requires_grad = True
                opt_Trans = get_optimizer(
                    opt_opt = self.opt.camera_optimizer,
                    net = [trans_bias],
                    # lr_reduction=0.1
                )
            
            
            
        inversion_opt = self.opt.trainer.inversion
        for sub_iter in tqdm(range(inversion_opt.iterations)) if is_master() else range(inversion_opt.iterations):          # lr schedule        

            do_sr = sub_iter >= (inversion_opt.iterations - sr_iters)
            mask_ratio = 1

            if _w_opt.shape[1] == 1: # ws other wise ws_plus 
                w_opt = _w_opt.repeat(1, self.net_G_module.backbone.mapping.num_ws, 1)
            else:
                w_opt = _w_opt
            ws = w_opt

            
            # select net or net_ema
            net_G = self.net_G_ema if isinstance(self.net_G_ema, torch.nn.Module) else self.net_G_module
            net_Warp = self.net_Warp_ema if use_ema else self.net_Warp_module

            # model forward
            ws_scaling, ws_trans, alpha = net_Warp(target_semantic)
            ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
            ws_trans = ws_trans * self.ws_stdv.to(ws)
            
            planes = net_G.before_planes(self.clip(ws * ws_scaling + ws_trans), noise_mode='const')
            synth_dict = net_G.render_from_planes(
                target_condition if not do_update_camera else self.add_bias(target_condition, intri_bias if if_optimize_intrinsic else None, trans_bias if if_optmize_translation else None),
                planes, 
                neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
            predict_images = synth_dict['image']
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                predict_images, predict_feature = synth_dict['image'], synth_dict['image_feature']
                sr_dict = net_G.sr(predict_images, predict_feature, self.clip(ws * ws_scaling + ws_trans))
                sr_images = sr_dict['image']

            # debug
            # from PIL import Image; import numpy as np
            # Image.fromarray(((synth_dict['image'][0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/debug.png')
            # Image.fromarray(((target_images[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/pose.png')
            
            # perceputal loss
            percept_loss = self._percept_loss(predict_images, target_images, self.criteria['perceptual_inverse_lr'], target_keypoints= target_keypoint, mask_rate = mask_ratio)
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                percept_loss += self._percept_loss(sr_images, target_images, self.criteria['perceptual_inverse_sr'], target_keypoints= target_keypoint, mask_rate = mask_ratio)

            # local loss
            local_loss = self._local_loss(predict_images, target_images, target_keypoint, self.criteria['local'], use_cache=sub_iter > 0)
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                local_loss += self._local_loss(sr_images, target_images, target_keypoint, self.criteria['local'], use_cache=sub_iter > (inversion_opt.iterations - sr_iters))
        
            # pixel loss
            pixel_loss = self.pixel_loss(predict_images, target_images, self.criteria['pixel'])
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                pixel_loss += self.pixel_loss(sr_images, target_images, self.criteria['pixel'])

            # id loss
            id_loss = self._local_loss(predict_images, target_images, target_keypoint, self.criteria['id'], use_cache=sub_iter > 0)
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                id_loss += self._local_loss(sr_images, target_images, target_keypoint, self.criteria['id'], use_cache=sub_iter > (inversion_opt.iterations - sr_iters))
        

            # monotonic loss  
            monotonic_loss = torch.zeros_like(local_loss)

            # TV loss
            TV_loss = torch.zeros_like(local_loss)
            
            # a_norm loss
            a_norm_loss = torch.ones_like(local_loss)

            # a_mutual loss
            a_mutual_loss = torch.zeros_like(local_loss)

            if not use_ema: # which means we are training
                assert source_images is not None; assert source_semantic is not None; assert source_condition is not None; assert source_keypoint is not None          # model forward

                # model forward
                ws_scaling_s, ws_trans_s, alpha_s = net_Warp(source_semantic)
                ws_scaling_s = ws_scaling_s + 1 if ws_scaling_s is not None else 1
                ws_trans_s = ws_trans_s * self.ws_stdv.to(ws)
                
                planes_s = net_G.before_planes(self.clip(ws * ws_scaling_s + ws_trans_s), noise_mode='const')
                synth_dict_s = net_G.render_from_planes(
                    source_condition if not do_update_camera else self.add_bias(source_condition, intri_bias if if_optimize_intrinsic else None, trans_bias if if_optmize_translation else None), 
                    planes_s, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images_s = synth_dict_s['image']
                if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr :
                    predict_images_s, predict_feature_s = synth_dict_s['image'], synth_dict_s['image_feature']
                    sr_dict_s = net_G.sr(predict_images_s, predict_feature_s, self.clip(ws * ws_scaling_s + ws_trans_s))
                    sr_images_s = sr_dict_s['image']

                percept_loss += self._percept_loss(predict_images_s, source_images, self.criteria['perceptual_inverse_lr'], target_keypoints=source_keypoint, mask_rate=mask_ratio)
                if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr :
                    percept_loss += self._percept_loss(sr_images_s, source_images, self.criteria['perceptual_inverse_sr'], target_keypoints=source_keypoint, mask_rate=mask_ratio)

                # monotonic loss
                monotonic_loss = self.criteria['monotonic'](ws, planes) + \
                                 self.criteria['monotonic'](ws, planes_s)

                # local loss
                local_loss += self._local_loss(predict_images_s, source_images, source_keypoint, self.criteria['local_s'], use_cache=sub_iter > 0)
                if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                    local_loss += self._local_loss(sr_images_s, source_images, source_keypoint, self.criteria['local_s'], use_cache=sub_iter > (inversion_opt.iterations - sr_iters))

                # pixel loss
                pixel_loss += self.pixel_loss(predict_images_s, source_images, self.criteria['pixel'])
                if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                    pixel_loss += self.pixel_loss(sr_images_s, source_images, self.criteria['pixel'])

                # id loss
                id_loss += self._local_loss(predict_images_s, source_images, source_keypoint, self.criteria['id_s'], use_cache=sub_iter > 0)
                if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                    id_loss += self._local_loss(sr_images_s, source_images, source_keypoint, self.criteria['id_s'], use_cache=sub_iter > (inversion_opt.iterations - sr_iters))

                # TV loss
                TV_loss = self.criteria['TV'](ws, planes) + \
                          self.criteria['TV'](ws, planes_s)

                # a norm
                a_norm_loss = (self.criteria['a_norm'](alpha) + \
                              self.criteria['a_norm'](alpha_s)) / 2

                # a mutual loss
                a_mutual_loss = self.criteria['a_mutual'](alpha, alpha_s)

            # sum up loss
            loss = self.weights['perceptual_inverse_lr'] * percept_loss + \
                   self.weights['local'] * local_loss + \
                   self.weights['monotonic'] * monotonic_loss + \
                   self.weights['TV'] * TV_loss + \
                   self.weights['pixel'] * pixel_loss + \
                   self.weights['a_norm'] * a_norm_loss + \
                   self.weights['a_mutual'] * a_mutual_loss + \
                   self.weights['id'] * id_loss

            opt_Ws.zero_grad()
            self.opt_Warp.zero_grad()
            if self.train_G:
                self.opt_G.zero_grad()
            if if_optimize_intrinsic:
                opt_Intri.zero_grad()
            if if_optmize_translation:
                opt_Trans.zero_grad()

            loss.backward()

            if use_ema: # test time
                opt_Ws.step()
                if if_optimize_intrinsic:
                    opt_Intri.step()
                if if_optmize_translation:
                    opt_Trans.step()

            else: # train time 
                update_type = inversion_opt.asynchronous_update
                if update_type is not None: # do alternatively update latent code and refinement model
                    # mapping types
                    if update_type == 'alternatively':
                        do_update_warp = (sub_iter % inversion_opt.warp_update_iters == inversion_opt.warp_update_iters -1)            
                    elif update_type == 'successively':
                        do_update_warp = (sub_iter >= inversion_opt.iterations - inversion_opt.warp_update_iters)  
                    else:
                        raise ValueError

                    if do_update_warp:
                        torch.nn.utils.clip_grad_norm_(self.net_Warp_module.parameters(), 1)
                        self.opt_Warp.step()
                    else: 
                        torch.nn.utils.clip_grad_norm_(_w_opt, 1)
                        opt_Ws.step()
                        if if_optimize_intrinsic:
                            opt_Intri.step()
                        if if_optmize_translation:
                            opt_Trans.step()

                else: # update togather
                    torch.nn.utils.clip_grad_norm_(self.net_Warp_module.parameters(), 1)
                    opt_Ws.step()
                    self.opt_Warp.step()
                    if if_optimize_intrinsic:
                        opt_Intri.step()
                    if if_optmize_translation:
                        opt_Trans.step()
                        
            # record loss 
            inverse_losses['inverse_perceptual'].append(percept_loss.detach().cpu())
            inverse_losses['inverse_local'].append(local_loss.detach().cpu())
            inverse_losses['inverse_monotonic'].append(monotonic_loss.detach().cpu())
            inverse_losses['inverse_TV'].append(TV_loss.detach().cpu())
            inverse_losses['inverse_pixel'].append(pixel_loss.detach().cpu())
            inverse_losses['inverse_a_mutual'].append(a_mutual_loss.detach().cpu())
            inverse_losses['inverse_a_norm'].append(a_norm_loss.detach().cpu())
            inverse_losses['inverse_id'].append(id_loss.detach().cpu())

        # accumulate loss
        for term, values in inverse_losses.items():
            if isinstance(values, list):
                inverse_losses[term] = sum(values) / len(values) if len(values) != 0 else torch.tensor(0)
            else:
                inverse_losses[term] = values
                
        return w_opt.detach(), \
                intri_bias.detach() if if_optimize_intrinsic and do_update_camera else None, \
                trans_bias.detach() if if_optmize_translation and do_update_camera else None, \
                inverse_losses

    def inverse_setup(self, batch_size, ws = None): # type in ['w', 'w_plus', 'pti']              
        # sample ws
        w_avg, w_std = self.sample_zs()
        # setup inversion variables
        if ws is not None: #w_plus
            assert isinstance(ws, torch.Tensor)
            assert ws.shape[1] == self.net_G_module.backbone.mapping.num_ws
            w_opt = ws.detach()
        else: # w
            w_opt = w_avg.repeat(batch_size, 1, 1).detach()

        w_opt.requires_grad = True

        # setup optimizer
        opt_Ws = get_optimizer(
            opt_opt = self.opt.inverse_optimizer,
            net = [w_opt]
            )    

        return opt_Ws, w_opt, w_std.detach()


    def reset_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    
    def optimize_parameters(self, data):
        gen_losses = {
            'warp_perceptual':[],
            'warp_local':[],
            'warp_monotonic':[],
            'warp_TV':[],
            'warp_pixel':[],
            'warp_a_norm': [],
            'warp_id':[],
        }

        B, D, _, H, W = data['images'].shape

        for sub_iter in tqdm(range(D)) if is_master() else range(D):

            # load data to device
            target_images = to_cuda(data['images'][:, sub_iter])
            target_semantic = to_cuda(data['semantics'][:, sub_iter])
            target_condition = to_cuda(data['conditions'][:, sub_iter])
            target_keypoint = to_cuda(data['keypoint'][:, sub_iter])

            # decide whether to performe on sr images
            do_sr = sub_iter >= D - (self.opt.trainer.sr_iters if hasattr(self.opt.trainer, 'sr_iters') else 0)
            
            if sub_iter == 0:
                # Inversion ###########################
                accumulate(self.net_Warp_module, self.net_Warp_ema,  1)

                # lr of warping net parameter should decrease
                self.reset_lr(self.opt_Warp, self.opt.warp_optimizer.lr / self.opt.trainer.inversion.iterations * self.opt.trainer.inversion.warp_lr_mult)

                # setup inversion parameters and optimizers
                opt_Ws, w_opt, w_std = self.inverse_setup(B)

                # iteratively optimize
                w_opt, intri_bias, trans_bias, _inverse_losses = self.inverse_optimize(
                    target_images, target_semantic, target_condition, target_keypoint, opt_Ws, w_opt, w_std,
                    source_images = to_cuda(data['images'][:, 1]), source_semantic = to_cuda(data['semantics'][:, 1]), source_keypoint=to_cuda(data['keypoint'][:, 1]), source_condition=to_cuda(data['conditions'][:, 1]),
                    sr_iters = self.opt.trainer.sr_iters if hasattr(self.opt.trainer, 'sr_iters') else self.opt.trainer.inversion.iterations,
                    )

                for k,v in _inverse_losses.items():
                    if k not in self.gen_losses:
                        self.gen_losses[k] = []
                    self.gen_losses[k].append(v.detach().cpu())

                # lr change back to normal
                self.reset_lr(self.opt_Warp, self.opt.warp_optimizer.lr)

            ws_scaling, ws_trans, alpha = self.net_Warp_module(target_semantic)
            ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
            ws_trans = ws_trans * self.ws_stdv.to(w_opt)

            planes = self.net_G_module.before_planes(self.clip(w_opt * ws_scaling + ws_trans), noise_mode='const')

            # rendering
            synth_dict = self.net_G_module.render_from_planes(
                self.add_bias(target_condition, intri_bias, trans_bias), 
                planes, 
                neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
            predict_images, predict_features = synth_dict['image'], synth_dict['image_feature']

            # perceputal loss
            percept_loss = self._percept_loss(predict_images, target_images, self.criteria['perceptual_refine_lr'], target_keypoints=target_keypoint)
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                sr_dict = self.net_G_module.sr(predict_images, predict_features, self.clip(w_opt * ws_scaling + ws_trans))
                sr_images = sr_dict['image']
                percept_loss += self._percept_loss(sr_images, target_images, self.criteria['perceptual_refine_sr'], target_keypoints=target_keypoint)
                
            # monotonic loss
            monotonic_loss = self.criteria['monotonic'](w_opt, planes)

            # tv loss
            TV_loss = self.criteria['TV'](w_opt, planes)

            # local loss
            local_loss = self._local_loss(predict_images, target_images, target_keypoint, self.criteria['local'])
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                local_loss += self._local_loss(sr_images, target_images, target_keypoint, self.criteria['local'])

            # pixel loss
            pixel_loss = self.pixel_loss(predict_images, target_images, self.criteria['pixel'])
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                pixel_loss += self.pixel_loss(sr_images, target_images, self.criteria['pixel'])

            #  id loss
            id_loss = self._local_loss(predict_images, target_images, target_keypoint, self.criteria['id'])
            if hasattr(self.opt.trainer, 'use_sr') and self.opt.trainer.use_sr and do_sr:
                id_loss += self._local_loss(sr_images, target_images, target_keypoint, self.criteria['id'])
        
            # alpha loss
            a_norm_loss = self.criteria['a_norm'](alpha)

            # sum up losses
            loss =  self.weights['perceptual_refine_lr'] * percept_loss + \
                    self.weights['monotonic'] * monotonic_loss + \
                    self.weights['TV'] * TV_loss + \
                    self.weights['local'] * local_loss + \
                    self.weights['pixel'] * pixel_loss + \
                    self.weights['a_norm'] * a_norm_loss + \
                    self.weights['id'] * id_loss
                    
            # grad backward
            self.opt_Warp.zero_grad()
            if self.train_G:
                self.opt_G.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_Warp_module.parameters(), 1)
            self.opt_Warp.step()
            if self.train_G:
                torch.nn.utils.clip_grad_norm_(self.net_G_module.parameters(), 1)
                self.opt_G.step()
            
                #record loss
            gen_losses['warp_perceptual'].append(percept_loss.detach().cpu())
            gen_losses['warp_monotonic'].append(monotonic_loss.detach().cpu())
            gen_losses['warp_TV'].append(TV_loss.detach().cpu())
            gen_losses['warp_local'].append(local_loss.detach().cpu())
            gen_losses['warp_pixel'].append(pixel_loss.detach().cpu())
            gen_losses['warp_a_norm'].append(a_norm_loss.detach().cpu())
            gen_losses['warp_id'].append(id_loss.detach().cpu())

        # accumulate loss
        for term, values in gen_losses.items():
            if k not in self.gen_losses:
                self.gen_losses[k] = []
            if isinstance(values, list):
                self.gen_losses[term] = sum(values) / len(values) if len(values) != 0 else torch.tensor(0)
            else:
                self.gen_losses[term] = values            

        # accumulate parameter
        accumulate(self.net_Warp_ema, self.net_Warp_module, self.accum_Warp)
        if self.train_G:
            accumulate(self.net_G_ema, self.net_G_module, self.accum_G)

    def sample_zs(self, batch_size = None, if_w_only = False):
        batch_size = self.opt.w_samples if batch_size is None else batch_size

        z_samples = to_cuda(torch.randn(
            batch_size, 
            self.net_G_module.z_dim
        ))

        camera_lookat_point = torch.tensor(
            self.net_G_module.rendering_kwargs['avg_camera_pivot'], 
        )

        cam2world_pose = to_cuda(LookAtPoseSampler.sample(
            horizontal_mean = 3.14 / 2, 
            vertical_mean = 3.14 / 2, 
            lookat_position = camera_lookat_point,
            radius=self.net_G_module.rendering_kwargs['avg_camera_radius'],
            batch_size=batch_size,
        ))
        
        focal_length = 4.2647
        intrinsics = to_cuda(torch.tensor([
            [focal_length, 0, 0.5],
            [0, focal_length, 0.5],
            [0, 0, 1]
        ]))

        c_samples = torch.cat([
                cam2world_pose.reshape(-1, 16),
                intrinsics.reshape(-1, 9).repeat(batch_size, 1),
            ],
            dim = 1)

        with torch.no_grad():
            w_samples = self.net_G_module.mapping(z_samples, c_samples)

        if if_w_only:
            return w_samples

        # w_avg = self.net_G_ema.mapping
        # torch.mean(
        #     w_samples[:, :1],
        #     dim = 0,
        #     keepdim = True
        # )
        w_avg = self.net_G_ema.backbone.mapping.w_avg.clone()

        w_std = torch.sum(
            (w_samples[:, :1] - w_avg) ** 2 / batch_size,
            dim = 0,
            keepdim = True
        ) ** 0.5

        return w_avg, w_std

    def _get_visualizations(self, data):

        B, D, _, H, W = data['images'].shape
        target_image_list = []
        sr_tr_refine_list = []
        lr_tr_refine_list = []
        sr_ema_refine_list = []
        lr_ema_refine_list = []
        sr_tr_ws_list = []
        lr_tr_ws_list = []
        sr_ema_ws_list = []
        lr_ema_ws_list = []        
        
        for sub_iter in range(D):
        
            # load data to device
            target_images = to_cuda(data['images'][:10, sub_iter,])
            target_semantic = to_cuda(data['semantics'][:10, sub_iter, ])
            target_condition = to_cuda(data['conditions'][:10, sub_iter, ])
            target_keypoint = to_cuda(data['keypoint'][:10, sub_iter, ])

            if sub_iter == 0:
                # Inversion ###########################
                self.net_Warp_ema.eval()
                
                # setup inversion parameters and optimizers
                opt_Ws, w_opt, w_std = self.inverse_setup(target_images.shape[0])

                # iteratively optimize
                w_opt, intri_bias, trans_bias, _ = self.inverse_optimize(
                    target_images, target_semantic, target_condition, target_keypoint, opt_Ws, w_opt, w_std,
                    use_ema=True,
                    sr_iters = self.opt.trainer.sr_iters if hasattr(self.opt.trainer, 'sr_iters') else self.opt.trainer.inversion.iterations)
            
            Hp, Wp = 128, 128

            # model forward
            with torch.no_grad():

                # source_semantic = to_cuda(data['semantics'][:10, 1, ])   
                # ws_scaling_s, ws_trans_s, alpha_s = self.net_Warp_ema(source_semantic)        

                ws_scaling, ws_trans, alpha = self.net_Warp_ema(target_semantic)
                ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
                ws_trans = ws_trans * self.ws_stdv.to(w_opt)

                # generation via trained net_G, w/ refine
                planes = self.net_G_module.before_planes(self.clip(w_opt * ws_scaling + ws_trans), noise_mode='const' )
                synth_dict = self.net_G_module.render_from_planes(
                    self.add_bias(target_condition, intri_bias, trans_bias), 
                    planes, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images, predict_features = synth_dict['image'], synth_dict['image_feature']
                sr_dict = self.net_G_module.sr(predict_images, predict_features, self.clip(w_opt * ws_scaling + ws_trans))
                sr_images = sr_dict['image']

                # collect images
                sr_images = F.interpolate(sr_images, size = (Hp, Wp), mode='area')
                lr_images = F.interpolate(predict_images, size = (Hp, Wp), mode='area')
                sr_tr_refine_list.append(sr_images.cpu())
                lr_tr_refine_list.append(lr_images.cpu())

                # generation via ema net_G, w/ refine
                planes = self.net_G_ema.before_planes(self.clip(w_opt * ws_scaling + ws_trans), noise_mode='const' )
                synth_dict = self.net_G_ema.render_from_planes(
                    self.add_bias(target_condition, intri_bias, trans_bias), 
                    planes, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images, predict_features = synth_dict['image'], synth_dict['image_feature']
                sr_dict = self.net_G_ema.sr(predict_images, predict_features, self.clip(w_opt * ws_scaling + ws_trans))
                sr_images = sr_dict['image']

                # collect images
                sr_images = F.interpolate(sr_images, size = (Hp, Wp), mode='area')
                lr_images = F.interpolate(predict_images, size = (Hp, Wp), mode='area')
                sr_ema_refine_list.append(sr_images.cpu())
                lr_ema_refine_list.append(lr_images.cpu())

                # generation via trained net_G, w/o refine
                planes = self.net_G_module.before_planes(self.clip(w_opt), noise_mode='const' )
                synth_dict = self.net_G_module.render_from_planes(
                    self.add_bias(target_condition, intri_bias, trans_bias),
                    planes, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images, predict_features = synth_dict['image'], synth_dict['image_feature']
                sr_dict = self.net_G_module.sr(predict_images, predict_features, self.clip(w_opt))
                sr_images = sr_dict['image']

                # collect images
                sr_images = F.interpolate(sr_images, size = (Hp, Wp), mode='area')
                lr_images = F.interpolate(predict_images, size = (Hp, Wp), mode='area')
                sr_tr_ws_list.append(sr_images.cpu())
                lr_tr_ws_list.append(lr_images.cpu())

                # generation via ema net_G, w/o refine
                planes = self.net_G_ema.before_planes(self.clip(w_opt), noise_mode='const' )
                synth_dict = self.net_G_ema.render_from_planes(
                    self.add_bias(target_condition, intri_bias, trans_bias),
                    planes, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images, predict_features = synth_dict['image'], synth_dict['image_feature']
                sr_dict = self.net_G_ema.sr(predict_images, predict_features, self.clip(w_opt))
                sr_images = sr_dict['image']

                # collect images
                sr_images = F.interpolate(sr_images, size = (Hp, Wp), mode='area')
                lr_images = F.interpolate(predict_images, size = (Hp, Wp), mode='area')
                sr_ema_ws_list.append(sr_images.cpu())
                lr_ema_ws_list.append(lr_images.cpu())

                target_images = F.interpolate(target_images, size = (Hp, Wp), mode='area')
                target_image_list.append(target_images.cpu()) 

        target_images_4_show = torch.cat(target_image_list, dim = 3)
        sr_tr_refine_4_show = torch.cat(sr_tr_refine_list, dim = 3)
        sr_tr_ws_4_shot = torch.cat(sr_tr_ws_list, dim = 3)
        sr_ema_refine_4_show = torch.cat(sr_ema_refine_list, dim = 3)
        sr_ema_ws_4_show = torch.cat(sr_ema_ws_list, dim = 3) 
        lr_tr_refine_4_show = torch.cat(lr_tr_refine_list, dim = 3)
        lr_ema_refine_4_show = torch.cat(lr_ema_refine_list, dim = 3)
        lr_tr_ws_4_show = torch.cat(lr_tr_ws_list, dim = 3)
        lr_ema_4_show = torch.cat(lr_ema_ws_list, dim = 3)

        compare_images_4_show = torch.cat([
            target_images_4_show, 
            sr_tr_refine_4_show, sr_tr_ws_4_shot, sr_ema_refine_4_show, sr_ema_ws_4_show,
            lr_tr_refine_4_show, lr_ema_refine_4_show, lr_tr_ws_4_show, lr_ema_4_show ], 
            dim = 2)
        compare_images_4_show = torch.cat(torch.chunk(compare_images_4_show, compare_images_4_show.size(0), 0), 2)

        return compare_images_4_show


    def test(self, data_loader, output_dir, current_iteration=-1, test_limit = 120):

        output_dir = os.path.join(
            self.opt.logdir, 'evaluation',
            'epoch_{:03}_iteration_{:07}'.format(self.current_epoch, current_iteration)
            )
        os.makedirs(output_dir, exist_ok=True)

        metrics = {
            'lpips':[],
            'psnr': [],
        }
        for it, data in enumerate(data_loader):
            if it >= test_limit:
                break
            _metrics = self._compute_metrics(data, it)
            for k,v in _metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        for term, values in metrics.items():
            metrics[term] = sum(values) / len(values)

        return metrics

    def _compute_metrics(self, data, current_iteration):
        metrics = {
            'lpips':[],
            'psnr':[],
        }

        B, D, _, H, W = data['images'].shape

        for sub_iter in range(D):

            # load data to device
            target_images = to_cuda(data['images'][:, sub_iter])
            target_semantic = to_cuda(data['semantics'][:, sub_iter])
            target_condition = to_cuda(data['conditions'][:, sub_iter])
            target_keypoint = to_cuda(data['keypoint'][:, sub_iter])

            if sub_iter == 0:
                # Inversion ###########################
                self.net_Warp_ema.eval()

                # setup inversion parameters and optimizers
                opt_Ws, w_opt, w_std = self.inverse_setup(B)

                # iteratively optimize
                w_opt, intri_bias, trans_bias, _ = self.inverse_optimize(
                    target_images, target_semantic, target_condition, target_keypoint, opt_Ws, w_opt, w_std,
                    use_ema=True,
                    sr_iters = self.opt.trainer.sr_iters if hasattr(self.opt.trainer, 'sr_iters') else self.opt.trainer.inversion.iterations)

            net_G = self.net_G_module #self.net_G_ema if isinstance(self.net_G_ema, torch.nn.Module) else self.net_G_module 

            # model forward
            with torch.no_grad():
                ws_scaling, ws_trans, alpha = self.net_Warp_ema(target_semantic)
                ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
                ws_trans = ws_trans * self.ws_stdv.to(w_opt)
                
                planes = self.net_G_module.before_planes(self.clip(w_opt * ws_scaling + ws_trans), noise_mode='const')
                synth_dict = net_G.render_from_planes(
                    self.add_bias(target_condition, intri_bias, trans_bias), 
                    planes, 
                    neural_rendering_resolution = self.opt.trainer.neural_rendering_resolution if hasattr(self.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images, predict_feature = synth_dict['image'], synth_dict['image_feature']
                sr_dict = net_G.sr(predict_images, predict_feature, self.clip(w_opt * ws_scaling + ws_trans))
                sr_images = sr_dict['image']

                if sr_images.shape[2] != H or sr_images.shape[3] != W:
                    sr_images = F.interpolate(sr_images, size = (H, W), mode='area')

                # metrics
                metrics['lpips'].append(self.lpips(sr_images, target_images).mean())
                metrics['psnr'].append(self.psnr(
                    (sr_images + 1) /2 * 255., 
                    (target_images + 1) /2 * 255,
                    ).item())

        # accumulate loss
        for term, values in metrics.items():
            metrics[term] = sum(values) / len(values)

        return metrics