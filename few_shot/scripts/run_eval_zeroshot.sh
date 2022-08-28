
cd vision_benchmark

# vb_zero_shot_eval --ds resources/datasets/cifar10.yaml --model resources/model/vitb32_CLIP.yaml --target local

python commands/zeroshot_eval.py --ds resources/datasets/fer2013.yaml --model resources/model/vitb32_CLIP.yaml --target local # ping-whiskey-plus
# python commands/zeroshot_eval.py --ds resources/datasets/cifar10.yaml --model resources/model/clip_swin.yaml --target local