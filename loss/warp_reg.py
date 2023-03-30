import torch
from models.warpper import make_coordinate_grid

def warp_TV_loss(ws, warp, net_G, reduction = 1):
    initial_coordinates = (torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1) / reduction # Front
    perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * net_G.rendering_kwargs['density_reg_p_dist']
    all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)

    warp = torch.cat(
        [
            warp[..., 0:2].permute(0, 3, 1, 2).unsqueeze(1),
            warp[..., 2:4].permute(0, 3, 1, 2).unsqueeze(1),
            warp[..., 4:6].permute(0, 3, 1, 2).unsqueeze(1),
        ],
        dim = 1
    )
    
    warp_samples = net_G.renderer.run_model(warp, None, all_coordinates, None, net_G.rendering_kwargs).permute(0, 2, 3, 1)
    
    std = torch.cat(
        [
            make_coordinate_grid(warp[:, 0]).permute(0, 3, 1, 2).unsqueeze(1),
            make_coordinate_grid(warp[:, 1]).permute(0, 3, 1, 2).unsqueeze(1),
            make_coordinate_grid(warp[:, 2]).permute(0, 3, 1, 2).unsqueeze(1),
        ],
        dim = 1
    )
    
    std_samples = net_G.renderer.run_model(std, None, all_coordinates, None, net_G.rendering_kwargs).permute(0, 2, 3, 1)
    
    warp_initial =  (warp_samples -  std_samples)[:, :warp_samples.shape[1] // 2]
    warp_perturbed =  (warp_samples -  std_samples)[:, warp_samples.shape[1] // 2:]
    warp_TV_loss = torch.nn.functional.l1_loss(warp_initial, warp_perturbed)
    return warp_TV_loss

def warp_all_loss(warp):

    warp = torch.cat(
        [
            warp[..., 0:2].permute(0, 3, 1, 2).unsqueeze(1),
            warp[..., 2:4].permute(0, 3, 1, 2).unsqueeze(1),
            warp[..., 4:6].permute(0, 3, 1, 2).unsqueeze(1),
        ],
        dim = 1
    )

    std = torch.cat(
        [
            make_coordinate_grid(warp[:, 0]).permute(0, 3, 1, 2).unsqueeze(1),
            make_coordinate_grid(warp[:, 1]).permute(0, 3, 1, 2).unsqueeze(1),
            make_coordinate_grid(warp[:, 2]).permute(0, 3, 1, 2).unsqueeze(1),
        ],
        dim = 1
    )

    warp_all_loss = torch.nn.functional.smooth_l1_loss(warp, std)
    return warp_all_loss