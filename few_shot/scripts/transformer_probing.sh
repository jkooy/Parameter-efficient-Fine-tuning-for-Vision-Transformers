dataset_dir=./finetune_output
output_dir=./transformer_probe_output
model_cfg=vit_base_patch32_224
random_seed=3
cd vision_benchmark

CUDA_VISIBLE_DEVICES=3 python commands/transformer_probe.py  --ds resources/datasets/fer2013.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $dataset_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 

# python commands/linear_probe.py  --ds resources/datasets/voc2007classification.yaml --model resources/model/vitb32_CLIP.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log 

# python commands/linear_probe.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local  DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log 

# python commands/linear_probe.py  --ds resources/datasets/caltech101.yaml --model resources/model/vit_base_patch32_224.yaml --classifier linear --save-feature True --target local  DATASET.RANDOM_SEED_SAMPLING $random_seed DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $output_dir/$model_cfg/datasets OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed 

