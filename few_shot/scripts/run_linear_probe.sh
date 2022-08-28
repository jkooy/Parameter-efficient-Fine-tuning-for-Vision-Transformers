dataset_dir=/data1/xh
output_dir=./linearprobe_output3
model_cfg=vitb32_CLIP
# random_seed=0
merge_encoder_and_proj=False
# dataset=hateful-memes
# dataset=voc2007classification
# dataset=caltech101
# dataset=voc2007classification
# dataset=ping-attack-on-titan-plus
# dataset=cifar10
# dataset=imagenet-1k
# dataset=ping-whiskey-plus
# dataset=country211

cd vision_benchmark

# for dataset in sun397 stl10
# for dataset in imagenet-1k
# for dataset in cifar100 dtd
for dataset in imagenet-1k cifar100 dtd eurosat-clip fer2013 fgvc-aircraft-2013b food101 gtsrb flower102 oxford-iiit-pets ping-attack-on-titan-plus rendered-sst2 resisc45-clip stanfordcar country211 kitti-distance mnist patchcamelyon ping-whiskey-plus caltech101 hateful-memes voc2007classification
do  
    for random_seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=1 python commands/linear_probe.py  --ds resources/datasets/$dataset.yaml --model resources/model/vitb32_CLIP.yaml --classifier linear --save-feature True --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/$random_seed  DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.FREEZE_IMAGE_BACKBONE True MODEL.CLIP_FP32 True TRAIN.INIT_HEAD_WITH_TEXT_ENCODER False TRAIN.MERGE_ENCODER_AND_HEAD_PROJ $merge_encoder_and_proj
    done
done