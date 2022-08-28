dataset_dir=./finetune_output
output_dir=./layernorm_output
model_cfg=vit_base_patch32_224
random_seed=3

cd vision_benchmark

# python commands/finetune.py  --ds resources/datasets/cifar100.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 
python commands/layernorm_tuning.py  --ds resources/datasets/fer2013.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5  TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 
# python commands/finetune.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 0  TRAIN.TWO_LR False DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 









# output_dir=../finetune_output
# model_cfg=vitb32_CLIP
# random_seed=1

# cd vision_benchmark

# python commands/finetune.py  --ds resources/datasets/cifar100.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 