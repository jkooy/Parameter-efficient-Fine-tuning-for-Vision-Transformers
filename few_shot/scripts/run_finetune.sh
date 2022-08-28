dataset_dir=/data1/xh
output_dir=./finetune_output
model_cfg=vitb32_CLIP
random_seed=0
# datasetp=hateful-memes
# dataset=ping-whiskey-plus
# datasetp=resisc45-clip
# datasetp=cifar10
# datasetp=caltech101

cd vision_benchmark

# for datasetp in cifar10
# for datasetp in sun397 stl10
# for datasetp in cifar100 dtd
for datasetp in cifar10 cifar100 dtd eurosat-clip fer2013 fgvc-aircraft-2013b food101 gtsrb flower102 oxford-iiit-pets ping-attack-on-titan-plus rendered-sst2 resisc45-clip stanfordcar country211 kitti-distance mnist patchcamelyon ping-whiskey-plus caltech101 hateful-memes voc2007classification
do 
    # for random_seed in 0
    for random_seed in 0 1 2
    do
        CUDA_VISIBLE_DEVICES=2 python commands/finetune.py  --ds resources/datasets/$datasetp.yaml --model resources/model/vitb32_CLIP.yaml --target local DATASET.NUM_SAMPLES_PER_CLASS 5 DATASET.RANDOM_SEED_SAMPLING $random_seed TRAIN.TWO_LR False DATASET.ROOT $dataset_dir/ OUTPUT_DIR $output_dir/$model_cfg/log_random_$random_seed TRAIN.INIT_HEAD_WITH_TEXT_ENCODER True MODEL.CLIP_FP32 True
    done
done