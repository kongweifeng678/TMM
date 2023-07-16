# Distortion-Aware Self-Supervised Indoor $360 ^{\circ}$ Depth Estimation via Hybrid Projection Fusion and Structural Regularities


## Train Command
OPENCV_IO_ENABLE_OPENEXR=1 python train.py --train_path YOUR_TRAIN_TXT_PATH --test_path YOUR_VAL_TXT_PATH --dataset my_3d60 --model_name YOUR_SAVE_MODEL_NAME --batch_size 6 --num_epochs 100 --height 256 --width 512 --imagenet_pretrained --net UniFuse_weight --gpu_devices GPU_ID --fusion cee --using_disp2seg --using_normloss

## Test Command
python 3d60_test.py --spherical_weights --spherical_sampling --median_scale --dataset my_3d60 --data_path YOUR_TEST_TXT_PATH  --load_weights_dir YOUR_MODEL_WEIGHT_PATH
