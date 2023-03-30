import torch
import torch.nn as nn
import pickle
import dnnlib
# import legacy

# discriminator_kwargs = {
#     'epilogue_kwargs': {
#         'mbstd_group_size': None,
#         'mbstd_num_features': 1,
#         'nonlinearity': 'lrelu'
#     },
#     'mapping_kwargs': {
#         'num_layers': 0,
#         'embed_features': 
#     }
# }


#         mapping_kwargs = dnnlib.EasyDict(
#             num_layers          = kwarg('mapping_layers',       0),
#             embed_features      = kwarg('mapping_fmaps',        None),
#             layer_features      = kwarg('mapping_fmaps',        None),
#             activation          = kwarg('nonlinearity',         'lrelu'),
#             lr_multiplier       = kwarg('mapping_lrmul',        0.1),

class GANLoss(nn.Module):
    def __init__(self, ckpt = 'pretrained/ffhqrebalanced512-64.pkl'):
        super().__init__()
        with open(ckpt, 'rb') as f:
            self.net_D = pickle.load(f)['D']


    def forward(self, image_dict, condition):
        self.net_D.eval()
        logits = self.net_D(image_dict, condition)
        loss = torch.nn.functional.softplus(-logits)
        return loss.mean(0)
