# OTAvatar : One-shot Talking Face Avatar with Controllable Tri-plane Rendering
## [Paper](https://arxiv.org/abs/2303.14662)  |  [Demo](https://youtu.be/qpIoMYFr7Aw)

## Update

April.4: The preprocessed dataset is released, please see the `Data preparation` section. Some missing files are also uploaded.

## Get started
### Environment Setup
```
git clone git@github.com:theEricMa/OTAvatar.git
cd OTAvatar
conda env create -f environment.yml
conda activate otavatar
```

### Pre-trained Models

Create `pretrained` folder under the root directory. 

Download and copy EG3D FFHQ model from offical [webpage](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d) to the `pretrained` directory. Choose the model with the name of `ffhqrebalanced512-64.pkl`.

Download [`arcface_resnet18.pth`](https://github.com/ronghuaiyang/arcface-pytorch) and save to the `pretrained` directory.

### Data preparation 
We upload the processed dataset in [Google Drive](https://drive.google.com/drive/folders/1ce9o_iB5v9oqmxop3Qn-pCXP60YdxURq?usp=share_link) and [Baidu Netdisk](https://pan.baidu.com/s/1R8j3pLqXsA4qRL7_eTrhQw?pwd=CBSR) (password: CBSR). 



Generally the processing scripts is a mixture of that in [PIRenderer](https://github.com/RenYurui/PIRender) and [ADNeRF](https://github.com/YudongGuo/AD-NeRF). We plan to further open a new repo to upload our revised preocessing script.

### Face Animation
Create the folder `result/otavatar`if it does not exist. Please the model (TODO) under this directory. Run,
```
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference_refine_1D_cam.py \
--config ./config/otavatar.yaml \
--name config/otavatar.yaml \
--no_resume \
--which_iter 2000 \
--image_size 512 \
--ws_plus \
--cross_id \
--cross_id_target WRA_EricCantor_000 \
--output_dir ./result/otavatar/evaluation/cross_ws_plus_WRA_EricCantor_000
```
To animate each identity given the motion from `WRA_EricCantor_000`.

Or simply run,
```
sh scripts/inference.sh
```

### Start Training
Run,
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12346 train_inversion.py \
--config ./config/otavatar.yaml \
--name otavatar
```

Or simply run,
```
sh scripts/train.sh
```

### Acknowledgement
We appreciate the model or code from [EG3D](https://github.com/NVlabs/eg3d), [PIRenderer](https://github.com/RenYurui/PIRender), [StyleHEAT](https://github.com/FeiiYin/StyleHEAT), [EG3D-projector](https://github.com/oneThousand1000/EG3D-projector).

### Citation
If you find this work helpful, please cite:
```
@article{ma2023otavatar,
  title={OTAvatar: One-shot Talking Face Avatar with Controllable Tri-plane Rendering},
  author={Ma, Zhiyuan and Zhu, Xiangyu and Qi, Guojun and Lei, Zhen and Zhang, Lei},
  journal={arXiv preprint arXiv:2303.14662},
  year={2023}
}
```

