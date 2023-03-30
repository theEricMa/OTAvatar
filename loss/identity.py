import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from kornia.geometry.transform import get_tps_transform, warp_image_tps ,get_perspective_transform, warp_affine, warp_perspective
from kornia.geometry.homography import find_homography_dlt
from skimage import transform

def conv3x3(inplanes, outplanes, stride=1):
    """A simple wrapper for 3x3 convolution with padding.
    Args:
        inplanes (int): Channel number of inputs.
        outplanes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
    """
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block used in the ResNetArcFace architecture.
    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """
    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    """Improved residual block (IR Block) used in the ResNetArcFace architecture.
    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """
    expansion = 1  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block used in the ResNetArcFace architecture.
    Args:
        inplanes (int): Channel number of inputs.
        planes (int): Channel number of outputs.
        stride (int): Stride in convolution. Default: 1.
        downsample (nn.Module): The downsample module. Default: None.
    """
    expansion = 4  # output channel expansion ratio

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """The squeeze-and-excitation block (SEBlock) used in the IRBlock.
    Args:
        channel (int): Channel number of inputs.
        reduction (int): Channel reduction ration. Default: 16.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # pool to 1x1 without spatial information
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.PReLU(), nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetArcFace(nn.Module):
    """ArcFace with ResNet architectures.
    Ref: ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    Args:
        block (str): Block used in the ArcFace architecture.
        layers (tuple(int)): Block numbers in each layer.
        use_se (bool): Whether use the SEBlock (squeeze and excitation block). Default: True.
    """

    def __init__(self, block, layers, use_se=True):
        if block == 'IRBlock':
            block = IRBlock
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetArcFace, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

class IDLoss(nn.Module):
    def __init__(self, base_dir='./', ckpt_dict = None, ):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = ResNetArcFace(
            block = 'IRBlock',
            layers = [2, 2, 2, 2],
            use_se = False
        )

        ret = self.facenet.load_state_dict(
            torch.load(
                os.path.join(base_dir, 'pretrained', 'arcface_resnet18.pth'),
                map_location = torch.device('cpu'),
            ) if ckpt_dict is None else ckpt_dict,
        )

        print('loading id loss module: {}'.format(ret))
        for param in self.facenet.parameters():
                param.requires_grad = False
        self.facenet.eval()


        # torch.save(self.facenet.module.state_dict(), os.path.join(base_dir, 'pretrained', 'arcface_resnet18.pth'))
        self.arcface_src = torch.tensor([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ],)

        self._cached = {}

    def warp_image(self, input_image, source_keypoint, use_cache = False):
        # return F.interpolate(input_image, [128, 128])
        B, _, W, H = input_image.shape
        assert W == H
        target_keypoint = self.arcface_src[None, ...].repeat(B, 1, 1).to(input_image)
        source_keypoint = self.lms68_2_lms5(source_keypoint) 
        # source_keypoint = source_keypoint
        
        # import kornia as K; import cv2; import numpy as np; from PIL import Image
        # def draw_points(img_t: torch.Tensor, points: torch.Tensor) -> np.ndarray:
        #     """Utility function to draw a set of points in an image."""

        #     # cast image to numpy (HxWxC)
        #     img: np.ndarray = K.tensor_to_image(img_t)

        #     img_out: np.ndarray = img.copy()

        #     n = len(points)
        #     for i, pt in enumerate(points):
        #         x, y = int(pt[0]), int(pt[1])
        #         img_out = cv2.circle(
        #             img_out, (x, y), radius=1, color=(int(255 / n * i) , int(255 / n * i) , int(255 / n * i) ), thickness=5
        #         )
        #     return np.clip(img_out, 0, 255)

        # image_1 = draw_points(255 * torch.ones([3, 112, 122]), self.arcface_src)
        # Image.fromarray(image_1.astype(np.uint8)).save('wasted/std_points.png')
        # image_2 =  draw_points(255 * torch.ones([3, 64, 64]), source_keypoint[0])
        # Image.fromarray(image_2.astype(np.uint8)).save('wasted/new_points.png')
        # Image.fromarray(((input_image[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/input.png')
              
        # extract M matrix
        if use_cache:
            Ms = self._cached[H]
        else:
            M_list = list()
            for i in range(B):
                tform = transform.estimate_transform(
                    'similarity', 
                    source_keypoint[i].cpu().numpy(), 
                    target_keypoint[i].cpu().numpy())
                M = tform.params[:2, :]
                M_list.append(M)
            Ms = torch.stack([torch.from_numpy(M) for M in M_list], dim = 0).to(input_image)
            self._cached[H] = Ms.detach() # save the result of defferent resolutions seperately

        warped_image = warp_affine(input_image, Ms[:, :2], [128, 128])
        
        # debug
        # from PIL import Image; import numpy as np
        # Image.fromarray(((warped_image[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/warped_gt.png')
        # Image.fromarray(((input_image[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/warped_gt_orig.png')
        return warped_image

    def lms68_2_lms5(self, lms_68):
        lms5 =  torch.cat(
            [
                (lms_68[:, 36:36 + 1] + lms_68[:, 39:39 + 1]) / 2,
                (lms_68[:, 42:42 + 1] + lms_68[:, 45:45 + 1]) / 2,
                lms_68[:, 30:30 + 1],
                lms_68[:, 48:48 + 1],
                lms_68[:, 54:54 + 1],
            ],
            dim = 1
        )
        return lms5

    def color2grad(self, out):
        return (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])

    def forward(self, predict_image, gt_image, gt_keypoint, criterion = torch.nn.CosineEmbeddingLoss(), use_cache = False):
        B = predict_image.shape[0]
        gt_align = self.color2grad(self.warp_image(gt_image, gt_keypoint, use_cache = use_cache)).unsqueeze(1)
        pred_align = self.color2grad(self.warp_image(predict_image, gt_keypoint, use_cache = use_cache)).unsqueeze(1)
        try:
            loss = criterion(self.facenet(gt_align).detach(), self.facenet(pred_align))
        except: # CosineEmbedding loss
            loss = criterion(self.facenet(gt_align).detach(), self.facenet(pred_align), target = torch.ones([B]).to(predict_image))
        return loss