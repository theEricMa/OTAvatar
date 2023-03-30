export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12346 train_inversion.py \
--config ./config/otavatar.yaml \
--name otavatar