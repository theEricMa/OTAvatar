import torch
import numpy as np

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1])
    zeros = torch.zeros([batch_size, 1])
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)

def compute_rotation_inv(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin( ), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def process_camera_inv(translation, Rs, focals): #crop_params):

    c_list = []

    N = len(translation)
    # for trans, R, crop_param in zip(translation,Rs, crop_params):
    for idx, (trans, R, focal) in enumerate(zip(translation, Rs, focals)):

        idx_prev = max(idx - 1, 0)
        idx_last = min(idx + 2, N - 1)

        trans = np.mean(translation[idx_prev: idx_last], axis = 0)
        R = np.mean(Rs[idx_prev: idx_last], axis = 0)

        # why
        trans[2] += -10
        c = -np.dot(R, trans)

        # # no why
        # c = trans

        pose = np.eye(4)
        pose[:3, :3] = R
        
        # why
        c *= 0.27
        c[1] += 0.015
        c[2] += 0.161
        # c[2] += 0.050  # 0.160

        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]

        # focal = 2985.29
        w = 1024#224
        h = 1024#224


        K =np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        # fix intrinsics
        K[0,0] = 2985.29/700 * focal / 1050
        K[1,1] = 2985.29/700 * focal / 1050
        K[0,2] = 1/2
        K[1,2] = 1/2     
        assert K[0,1] == 0
        assert K[2,2] == 1
        assert K[1,0] == 0
        assert K[2,0] == 0
        assert K[2,1] == 0  

        # fix_pose_orig
        pose = np.array(pose).copy()

        # why
        pose[:3, 3] = pose[:3, 3]/4.0 * 2.7
        # # no why
        # t_1 = np.array([-1.3651,  4.5466,  6.2646])
        # s_1 = np.array([-2.3178, -2.3715, -1.9653]) + 1
        # t_2 = np.array([-2.0536,  6.4069,  4.2269])
        # pose[:3, 3] = (pose[:3, 3] + t_1) * s_1 + t_2

        c = np.concatenate([pose.reshape(-1), K.reshape(-1)])
        c_list.append(c.astype(np.float32))          

    return c_list

def process_camera_inv_test(translation, Rs, focals): #crop_params):

    c_list = []

    N = len(translation)

    _translation = translation.copy()
    _translation[..., 2] += -10
    # _c = -torch.tensor(Rs[:1]).repeat(Rs.shape[0], 1, 1).bmm(torch.tensor(_translation[..., None]))[..., 0]
    # _c = -torch.tensor(Rs).mean(dim = 0, keepdim = True).repeat(Rs.shape[0], 1, 1).bmm(torch.tensor(_translation[..., None]))[..., 0]
    # _c = -torch.eye(3).unsqueeze(0).repeat(Rs.shape[0], 1, 1).bmm(torch.tensor(_translation[..., None]))[..., 0]
    _c = -torch.tensor(Rs).bmm(torch.tensor(_translation[..., None]))[..., 0]

    a = torch.tensor([0.27, 0.27, 0.65])

    tmp_c = _c.clone()
    tmp_c *= 0.27
    tmp_c[..., 1] += 0.015
    tmp_c[..., 2] += 0.161

    tmp_c_2 = a * _c - (a * _c.mean(dim = 0) - tmp_c.mean(dim = 0))
    translation = tmp_c_2.numpy()    

    # for trans, R, crop_param in zip(translation,Rs, crop_params):
    for idx, (trans, R, focal) in enumerate(zip(translation, Rs, focals)):

        idx_prev = max(idx - 2, 0)
        idx_last = min(idx + 3, N - 1)

        trans = np.mean(translation[idx_prev: idx_last], axis = 0)
        R = np.mean(Rs[idx_prev: idx_last], axis = 0)

        # # why
        # trans[2] += -10
        # c = -np.dot(R, trans)

        # no why
        c = trans

        pose = np.eye(4)
        pose[:3, :3] = R
        
        # why
        # c *= 0.27
        # c[1] += 0.015
        # c[2] += 0.161
        # # c[2] += 0.050  # 0.160

        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]

        # focal = 2985.29
        w = 1024#224
        h = 1024#224


        K =np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        # fix intrinsics
        K[0,0] = 2985.29/700 * focal / 1050
        K[1,1] = 2985.29/700 * focal / 1050
        K[0,2] = 1/2
        K[1,2] = 1/2     
        assert K[0,1] == 0
        assert K[2,2] == 1
        assert K[1,0] == 0
        assert K[2,0] == 0
        assert K[2,1] == 0  

        # fix_pose_orig
        pose = np.array(pose).copy()

        # why
        pose[:3, 3] = pose[:3, 3]/4.0 * 2.7
        # # no why
        # t_1 = np.array([-1.3651,  4.5466,  6.2646])
        # s_1 = np.array([-2.3178, -2.3715, -1.9653]) + 1
        # t_2 = np.array([-2.0536,  6.4069,  4.2269])
        # pose[:3, 3] = (pose[:3, 3] + t_1) * s_1 + t_2

        c = np.concatenate([pose.reshape(-1), K.reshape(-1)])
        c_list.append(c.astype(np.float32))          

    return c_list

    # translation_old = translation
    # translation_new = np.vstack(c_list)[:, :16].reshape(-1, 4, 4)[:, :3, 3]

    # translation_old = torch.from_numpy(translation_old).cuda()
    # translation_old_mean = torch.mean(translation_old, dim = 0, keepdim=True)
    # translation_old_std = torch.std(translation_old, dim = 0, keepdim=True)
    # translation_old = (translation_old - translation_old_mean) / translation_old_std

    # translation_new = torch.from_numpy(translation_new).cuda()
    # translation_new_mean = torch.mean(translation_new, dim = 0, keepdim=True)
    # translation_new_std = torch.std(translation_new, dim = 0, keepdim=True)
    # translation_new = (translation_new - translation_new_mean) / translation_new_std

    # t_1 = torch.nn.parameter.Parameter(torch.zeros([1, 3]).cuda(),)
    # s_1 = torch.nn.parameter.Parameter(torch.zeros([1, 3]).cuda(),)
    # t_2 = torch.nn.parameter.Parameter(torch.zeros([1, 3]).cuda(),)

    # optim = torch.optim.AdamW([t_1, s_1, t_2], lr = 0.0001, weight_decay=0.1)
    
    # best = {'loss': np.inf, 't_1': t_1.detach().clone(), 's_1': s_1.detach().clone(), 't_2': t_2.detach().clone()}

    # for i in range(10000000):
    #     loss = torch.nn.functional.l1_loss(translation_new, ((translation_old + t_1) * (s_1 + 1)) + t_2)
    #     loss.backward()
    #     optim.step()

    #     loss = loss.detach().cpu().numpy()

    #     if loss <= best['loss']:
    #         best = {'loss': loss, 't_1': t_1.detach().clone(), 's_1': s_1.detach().clone(), 't_2': t_2.detach().clone()}
    #         print(best)

    #     if i % 1000 == 0:
    #         print(loss)
    
    # w_1 = torch.nn.parameter.Parameter(torch.eye(3).unsqueeze(0).cuda(),)
    # t_1 =  torch.tensor([[0, 0, -10]]).cuda()
    # s_1= torch.nn.parameter.Parameter(torch.tensor([[0.27, 0.27, 0.27]]).cuda(),)
    # t_2 = torch.nn.parameter.Parameter(torch.tensor([[0, 0.030, 0.161]]).cuda(),)
    # s_2= torch.nn.parameter.Parameter(torch.tensor([[1/4 * 2.7, 1/4 * 2.7, 1/4 * 2.7]]).cuda(),)

    # optim = torch.optim.AdamW([w_1, t_1, s_1, t_2, s_2], lr = 0.0001, weight_decay=0.)
    # for i in range(10000000):
    #     translation_pred = (torch.bmm(w_1.repeat(10969, 1, 1), (translation_old + t_1).unsqueeze(-1)).squeeze(-1) * s_1 + t_2) * s_2
    #     loss = torch.nn.functional.l1_loss(translation_new, translation_pred)
    #     loss.backward()
    #     optim.step()

    #     if i % 100 == 0:
    #         print(loss.detach().cpu().numpy())

    

#----------------------------------------------------------------------------

def process_camera(translation, Rs, warpings, resolution = 256): #crop_params):
    warpings_new = []

    # for idx in range(len(warpings)):
    #     idx_prev = max(idx - 10, 0)
    #     idx_last = min(idx + 11, len(warpings) - 1)
    #     warping_avg= np.average(warpings[idx_prev: idx_last], axis = 0)

    #     warpings_new.append(warping)
    for idx in range(len(warpings)):
        # warping scaleing
        idx_prev = max(idx - 10, 0)
        idx_last = min(idx + 11, len(warpings))
        warping_avg= np.average(warpings[idx_prev: idx_last], axis = 0)
        
        # warpings_new.append(warping_avg)   
        warping = warpings[idx]
        warping[0,0] = warping_avg[0,0]
        warping[1,1] = warping_avg[1,1]

        # warping translation
        idx_prev = max(idx - 4, 0)
        idx_last = min(idx + 5, len(warpings))
        warping_avg= np.average(warpings[idx_prev: idx_last], axis = 0)

        warping[0,3] = warping_avg[0,3]
        warping[1,3] = warping_avg[1,3]

        warpings_new.append(warping)   

    warpings = warpings_new

    c_list = []
    # for trans, R, crop_param in zip(translation,Rs, crop_params):
    for trans, R, warping in zip(translation,Rs, warpings):

        # trans += warping[:3, -1] / 255

        trans[2] += -10
        c = -np.dot(R, trans)
        pose = np.eye(4)
        pose[:3, :3] = R


        c *= 0.27
        c[1] += 0.006
        c[2] += 0.161
        
        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]

        focal = 2985.29
        w = 1024#224
        h = 1024#224

        c = -np.dot(R, trans)

        K =np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0

        Rot = np.eye(3)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1        
        pose[:3, :3] = np.dot(pose[:3, :3], Rot)


        # w0, h0, s, t_x, t_y = crop_param

        # fix intrinsics
        K[0,0] = 2985.29/700 / (resolution / 256)
        K[1,1] = 2985.29/700  / (resolution / 256)
        K[0,2] = 1/2
        K[1,2] = 1/2     
        assert K[0,1] == 0
        assert K[2,2] == 1
        assert K[1,0] == 0
        assert K[2,0] == 0
        assert K[2,1] == 0  

        # # add scaling and translation
        # t_x_new = (w0 * s / 2 - 300 / 2 + float((t_x - w0 / 2) * s)).astype(np.int32)
        # t_y_new = (h0 * s / 2 - 300 / 2 + float((h0 / 2 - t_y) * s)).astype(np.int32)

        # K[0,0] /= s 
        # K[1,1] /= s
        # K[0,2] /= s; K[0,2] += t_x_new / (w0 * s) * 1/2 
        # K[1,2] /= s; K[1,2] += t_y_new / (h0 * s) * 1/2 


        # fix_pose_orig
        pose = np.array(pose).copy()
        location = pose[:3, 3] 
        radius = np.linalg.norm(location)
        pose[:3, 3] = pose[:3, 3]/radius * 2.7

        # # # implement talking face warpingz
        # warping[0,-1] = -1 * warping[0, -1] / warping[0,0] / 512
        # warping[1,-1] = -1 * warping[1, -1] / warping[1,1] / 512
        # warping[0,0] = 1 / warping[0,0] #1 / warping[0,0]
        # warping[1,1] = 1 / warping[1,1] #1 / warping[1,1]
        # pose = warping @ pose
        # # warping_ = np.eye(4); warping_[:3, :3] = warping[:3, :3]
        # # pose = warping_ @ pose

        c = np.concatenate([pose.reshape(-1), K.reshape(-1)])
        c_list.append(c.astype(np.float32))
        

    if len(c_list) > 1:
        # perform average
        c_list_new = []
        for idx in range(len(c_list)):
            idx_prev = max(idx - 5, 0)
            idx_last = min(idx + 6, len(c_list))
            c_new = np.average(np.vstack(c_list[idx_prev: idx_last]), axis = 0)
            c_list_new.append(c_new)        

        return c_list_new
    else:
        return c_list