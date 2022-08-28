dataset_dir=/data1/xh
output_dir=./test
model_cfg=vitb32_CLIP
random_seed=0
# dataset=hateful-memes
# dataset=voc2007classification
# dataset=caltech101
# dataset=ping-whiskey-plus
# dataset=resisc45-clip
# dataset=cifar100
# dataset=imagenet-1k
# dataset=country211

cd vision_benchmark

# for dataset in cifar10
for dataset in cifar10 cifar100 dtd eurosat-clip fer2013 fgvc-aircraft-2013b food101 gtsrb flower102 oxford-iiit-pets ping-attack-on-titan-plus rendered-sst2 resisc45-clip stanfordcar country211 kitti-distance mnist patchcamelyon ping-whiskey-plus caltech101 hateful-memes voc2007classification imagenet-1k
do  
    for random_seed in 0
    # for random_seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=3 python commands/lora_clip.py  --ds resources/datasets/$dataset.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True MODEL.CLIP_FP32 True
    done
done