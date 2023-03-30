import os
import cv2 
import argparse
import numpy as np

from tqdm import tqdm

import torch
from torch.nn import functional as F
from traitlets import default

from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import gen_model_optimizer_4_warping_n_inversion, set_random_seed, get_trainer_4_warping_n_inversion
from util.distributed import master_only_print as print
from config import Config

from time import time

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--cross_id_target', default=None, help = 'the target identity to render')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--multi_view', type=bool, default=False, help = 'whether to perform multi-view test')
    parser.add_argument('--cam_optim', type=bool, default=False, help = 'whether to optimize camera poses')

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--with_gt', type=int, default=True)

    parser.add_argument('--ws_per_frame', action='store_true')
    parser.add_argument('--ws_plus', action='store_true')
    parser.add_argument('--pti', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--rotate', action='store_true')

    parser.add_argument('--fps', action='store_true')

    args = parser.parse_args()
    return args

def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = results_dir+'.mp4' 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() 

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a dataset
    opt.data.cross_id = args.cross_id
    opt.data.cross_id_target = args.cross_id_target
    if args.multi_view:
        from data.multiface_video_dataset_inv_fix_target import MultifaceVideoDataset
        dataset = MultifaceVideoDataset(opt.data, is_inference=True)
        # opt.trainer.inversion.iterations = 300
    else:
        if args.cross_id_target is not None:
            assert args.cross_id
            from data.hdtf_video_dataset_inv_fix_target import HDTFVideoDataset
        else:
            from data.hdtf_video_dataset_inv import HDTFVideoDataset
        dataset = HDTFVideoDataset(opt.data, is_inference=True)

    # create a model
    net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_warp, opt_G, sch_G \
        = gen_model_optimizer_4_warping_n_inversion(opt)

    # change iterations

    trainer = get_trainer_4_warping_n_inversion(opt, net_Warp, net_Warp_ema, opt_Warp, sch_Warp, net_G, net_G_warp, opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)     

    trainer.net_Warp_ema.eval()
    trainer.net_G_ema.eval()

    output_dir = os.path.join(
        args.output_dir, 
        'epoch_{:05}_iteration_{:09}'.format(current_epoch, current_iteration)
        )
    os.makedirs(output_dir, exist_ok=True)



    for video_index in tqdm(range(dataset.__len__())):
        data = dataset.load_next_video()
        name = data['video_name']

        output_images, gt_images= [],[]
        for frame_index, idx in enumerate(tqdm(range(len(data['target_semantics'])))):

            if frame_index > 50 and args.debug: 
                break
            
            target_semantic = data['target_semantics'][frame_index][None].cuda()
            target_condition = data['target_conditions'][frame_index][None].cuda()
            target_image = data['target_image'][frame_index][None].cuda()
            target_keypoint = data['target_keypoint'][frame_index][None].cuda()

            if args.rotate: # if specified, used rotation camera poses
                from util.camera_utils import LookAtPoseSampler
                N = len(data['target_conditions'])
                pitch_range = 0.50
                yaw_range = 0.50
                frames_period = 100
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_index / frames_period),
                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_index / frames_period),
                    torch.tensor(trainer.net_G_ema.rendering_kwargs['avg_camera_pivot']), 
                    radius=trainer.net_G_ema.rendering_kwargs['avg_camera_radius'],
                    ).to(target_condition)
                target_condition[..., :16] = cam2world_pose.reshape(-1, 16)

            if (frame_index == 0 and not args.ws_per_frame) or args.ws_per_frame:

                source_semantic = data['source_semantics'][None].cuda()
                source_condition = data['source_conditions'][None].cuda()
                source_image = data['source_image'][None].cuda()
                source_keypoint = data['source_keypoint'][None].cuda()
                
                opt_Ws, w_opt, w_std = trainer.inverse_setup(1,)
                w_opt, intri_bias, trans_bias, _ = trainer.inverse_optimize(
                    source_image, source_semantic, source_condition, source_keypoint, opt_Ws, w_opt, w_std,
                    use_ema=True, 
                    sr_iters = opt.trainer.sr_iters if hasattr(opt.trainer, 'sr_iters') else opt.trainer.inversion.iterations,
                )     

                if args.ws_plus:
                    opt_Ws, w_opt, w_std = trainer.inverse_setup(1,  ws = w_opt)
                    w_opt, intri_bias, trans_bias, _ = trainer.inverse_optimize(
                        source_image, source_semantic, source_condition, source_keypoint, opt_Ws, w_opt, w_std,
                        use_ema=True, 
                        sr_iters = opt.trainer.sr_iters if hasattr(opt.trainer, 'sr_iters') else opt.trainer.inversion.iterations,
                        intri_bias = intri_bias,
                        trans_bias = trans_bias
                    )     
                if args.pti:
                    net_G, intri_bias, trans_bias, = trainer.net_G_optimize(source_image, source_semantic, source_condition, source_keypoint, opt_Ws, w_opt, w_std,
                        use_ema = True, 
                        sr_iters = opt.trainer.sr_iters if hasattr(opt.trainer, 'sr_iters') else opt.trainer.inversion.iterations,
                        intri_bias = intri_bias,
                        trans_bias = trans_bias,
                        max_iters = 50
                    )
                else:
                    net_G = trainer.net_G_ema

            if frame_index == 0 and args.cam_optim:
                # this ws should not be used !
                # take note that the grad of w_opt has been detached here!
                w_opt.requires_grad = False # just to make sure
                
                # _opt_Ws, _w_opt, _w_std = trainer.inverse_setup(1,)
                _, intri_bias, trans_bias, _ = trainer.inverse_optimize(
                    target_image, target_semantic, target_condition, target_keypoint, opt_Ws, w_opt, w_std,
                    use_ema=True, #if_optmize_translation = True,
                    sr_iters = opt.trainer.sr_iters if hasattr(opt.trainer, 'sr_iters') else opt.trainer.inversion.iterations,
                )   

            start = time()

            with torch.no_grad():
        
                ws_scaling, ws_trans, alpha = trainer.net_Warp_ema(target_semantic)
                ws_scaling = ws_scaling + 1 if ws_scaling is not None else 1
                ws_trans = ws_trans * trainer.ws_stdv.to(w_opt)

                # source_semantic = data['target_semantics'][0][None].cuda()
                # ws_scaling_s, ws_trans_s, alpha_s = trainer.net_Warp_ema(source_semantic)
                # import pdb; pdb.set_trace()
                planes = net_G.before_planes(w_opt * ws_scaling + ws_trans, noise_mode = 'const')
                output_dict = net_G.render_from_planes(
                    trainer.add_bias(target_condition, intri_bias, trans_bias), 
                    planes, 
                    neural_rendering_resolution = trainer.opt.trainer.neural_rendering_resolution if hasattr(trainer.opt.trainer, 'neural_rendering_resolution') else 64)
                predict_images_res, predict_feature_res = output_dict['image'], output_dict['image_feature']
  
                output_dict = net_G.sr(predict_images_res, predict_feature_res, w_opt * ws_scaling + ws_trans)

            end = time()

            if output_dict['image'].shape[-1] != args.image_size:
                output_dict['image'] = F.interpolate(output_dict['image'], size = (args.image_size, ) * 2, mode='area')
            

            if args.fps:
                fps = 1 / (end - start)
                text = 'FPS:{}'.format(int(fps))
                pred = output_dict['image'].cpu().clamp_(-1, 1)
                pred_numpy = (((pred[0].permute([1,2,0]) + 1) / 2) * 255).numpy().astype(np.uint8)
                pred_numpy = cv2.cvtColor(pred_numpy, cv2.COLOR_RGB2BGR)
                H, W = pred_numpy.shape[:2]
                orig = (int(0.55 * W), int(0.1 * H))
                fontScale= H // 256
                thickness= W // 256
                pred_numpy = cv2.putText(img = pred_numpy, text = text, org = orig, fontFace=cv2.FONT_HERSHEY_TRIPLEX, color = (0, 244, 0), fontScale=fontScale, thickness=thickness)
                # cv2.imwrite('wasted/fps.png', pred_numpy)
                pred_numpy = cv2.cvtColor(pred_numpy, cv2.COLOR_BGR2RGB)
                pred = (torch.from_numpy(pred_numpy) / 255 * 2 - 1).permute([2,0,1])[None, ...].to(output_dict['image'])
                output_images.append(pred)
            else:
                output_images.append(
                    output_dict['image'].cpu().clamp_(-1, 1)
                    )
            gt_images.append(
                data['target_image'][frame_index][None]
                )
        
        gen_images = torch.cat(output_images, 0)
        gt_images = torch.cat(gt_images, 0)

        if args.with_gt:
            write2video("{}/{}".format(output_dir, name), gt_images, gen_images)
        else:
            write2video("{}/{}".format(output_dir, name), gt_images)

        print("write results to video {}/{}".format(output_dir, name))

