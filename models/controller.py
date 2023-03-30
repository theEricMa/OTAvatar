import torch
import torch.nn as nn

class MappingNet(torch.nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer):
        super().__init__()
        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0,bias=True)
        )        

        for i in range(layer):
            net = nn.Sequential(
                nonlinearity,
                nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0,dilation=3),
            )
            setattr(self, 'encoder'+str(i), net)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder'+ str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out),             
        return out
    
class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 512
        weight = self.weight + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

class VideoCodeBook(nn.Module):
    def __init__(self,        
        descriptor_nc,
        mapping_layers,
        mlp_layers,
        directions,
        if_use_pose = False,
        if_plus_scaling = False,
        if_short_cut = True,
        if_norm = False,
        ) -> None:

        super(VideoCodeBook, self).__init__()
        self.mapping_3DMM = MappingNet(
            coeff_nc=70 if if_use_pose else 64,
            descriptor_nc=descriptor_nc,
            layer=mapping_layers,
        )

        # add new mapping net 
        self.mapping_Refine = nn.Sequential(
            *[nn.Linear(descriptor_nc, descriptor_nc), nn.LeakyReLU(0.1)] * mlp_layers,
            nn.Linear(descriptor_nc, (28 + 2) * directions if if_plus_scaling else (14 + 1) * directions),
        )

        self.direction = Direction(directions)
        self.if_use_pose = if_use_pose
        self.if_pluse_scaling = if_plus_scaling
        self.if_short_cut = if_short_cut
        self.if_norm = if_norm

        if if_norm:
            self.norm = nn.BatchNorm1d(descriptor_nc)
            if if_plus_scaling:
                self.direction2 = Direction(directions)
                self.norm2 = nn.BatchNorm1d(descriptor_nc)
        else:
            self.norm = lambda x: x
            if if_plus_scaling:
                self.norm2 = lambda x: x

    def forward(self, driving):

        B = driving.shape[0]
        if not self.if_use_pose:
            driving = driving[:, :64]

        feat = self.mapping_3DMM(driving)[0].permute(0,2,1)
        if self.if_short_cut:
            feat = self.mapping_Refine[-2:](feat + self.mapping_Refine[:-2](feat))
        else:
            feat = self.mapping_Refine(feat)

        if not self.if_pluse_scaling:            
            feat = feat.view(B, 15, -1)
            feat = self.norm(feat[:, :1] + feat[:, 1:])
            
            directions = self.direction(feat.view(B * 14, -1)).view(B, 14, -1)
            return None, directions, feat

        else:
            feat = feat.view(B, 30, -1)

            feat1 = self.norm(feat[:, 0:1] + feat[:, 1:15])
            directions1 = self.direction(feat1.view(B * 14, -1)).view(B, 14, -1)

            feat2 = self.norm2(feat[:, 15:16] + feat[:, 16:30])
            directions2 = self.direction(feat2.view(B * 14, -1)).view(B, 14, -1)

            return directions1, directions2, feat