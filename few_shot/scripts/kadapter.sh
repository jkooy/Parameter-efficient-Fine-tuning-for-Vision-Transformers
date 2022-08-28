dataset_dir=/data1/xh
output_dir=./kadapter_output
model_cfg=vit_base_patch32_224
random_seed=3

cd vision_benchmark

# CUDA_VISIBLE_DEVICES=2 python commands/kronecker_adaptation.py  --ds resources/datasets/gtsrb.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed
# CUDA_VISIBLE_DEVICES=3 python commands/kronecker_adaptation.py  --ds resources/datasets/fer2013.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed
# CUDA_VISIBLE_DEVICES=1 python commands/kronecker_adaptation.py  --ds resources/datasets/cifar100.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 
# CUDA_VISIBLE_DEVICES=1 python commands/kronecker_adaptation.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 
# CUDA_VISIBLE_DEVICES=1 python commands/kronecker_adaptation.py  --ds resources/datasets/cifar10.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed   


python commands/kronecker_adaptation.py  --ds resources/datasets/cifar10.yaml --model resources/model/vitb32_CLIP_par.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True






# output_dir=../finetune_output
# model_cfg=vitb32_CLIP
# random_seed=1

# cd vision_benchmark

# python commands/finetune.py  --ds resources/datasets/cifar100.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 