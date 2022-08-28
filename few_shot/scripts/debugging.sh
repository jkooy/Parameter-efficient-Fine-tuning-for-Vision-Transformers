dataset_dir=/data1/xh
output_dir=./seed_output
model_cfg=vit_base_patch32_224
random_seed=0
cd vision_benchmark

# CUDA_VISIBLE_DEVICES=2 python commands/debugging.py  --ds resources/datasets/fer2013.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5  DATASET.RANDOM_SEED_SAMPLING $random_seed DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 

# GPU=3 python commands/seed_debugging.py  --ds resources/datasets/cifar10.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5  DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_3
# python commands/linear_probe.py  --ds resources/datasets/voc2007classification.yaml --model resources/model/vitb32_CLIP.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log 

# python commands/linear_probe.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local  DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log 

# python commands/linear_probe.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local  DATASET.RANDOM_SEED_SAMPLING $random_seed DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed



python commands/debugging.py --ds resources/datasets/country211.yaml --model resources/model/vitb32_CLIP_par.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True MODEL.CLIP_FP32 True
# python commands/finetune.py  --ds resources/datasets/$datasetp.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True MODEL.CLIP_FP32 True