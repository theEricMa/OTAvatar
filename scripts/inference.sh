export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference_refine_1D_cam.py \
--config ./config/config/otavatar.yaml \
--name config/otavatar.yaml \
--no_resume \
--which_iter 2000 \
--image_size 512 \
--ws_plus \
--cross_id \
--cross_id_target WRA_EricCantor_000 \
--output_dir ./result/otavatar/evaluation/cross_ws_plus_WRA_EricCantor_000
