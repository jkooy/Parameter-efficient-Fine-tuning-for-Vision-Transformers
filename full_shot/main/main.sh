
'''
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb32_CLIP_100.yaml  --data_dir /home/anonymous/dataset --output_dir output/lincls/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt 

#transformer probing
# python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml  --data_dir /home/anonymous/dataset --output_dir output/lincls/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune TransformerProbe --no-search False --lr-range 1e-05


# Finetune
# python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_300.yaml --data_dir /home/anonymous/dataset --output_dir output/lincls/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5
# CUDA_VISIBLE_DEVICES=2 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False 



# amlt run ml_datasets_adapter_vit_100_lr.yaml adapter_experiment_vitpython tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_test.yaml --data_dir /home/anonymous/dataset --output_dir all_datasets_tuning_vit_adapter/lr --model_ckpt /home/anonymous/mnist_pytorch/ViT-B-32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True
'''

### adapter
python tools/eval_local.py --ds experiments/eval/dataset/imagenet-1k.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /data1/xh --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True

'''
# CUDA_VISIBLE_DEVICES=2 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True --LoRA True --LoRAFix True
# CUDA_VISIBLE_DEVICES=2 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True --ladapter True
# CUDA_VISIBLE_DEVICES=0 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True --ladapter loradropadapter

# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True --ladapter True


### attention
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm Attention


### bias
CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm bias


### attention cswin
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm cswin
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml  --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm cswin

### attention position bias
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml  --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm PositionBias
# CUDA_VISIBLE_DEVICES=1 python tools/eval_local.py --ds experiments/eval/dataset/voc2007classification.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml  --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm PositionBias


### evaluate
# CUDA_VISIBLE_DEVICES=0 python tools/eval_metric.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_evaluate.yaml --data_dir /home/anonymous/dataset --output_dir output/evaluate --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5
# CUDA_VISIBLE_DEVICES=0 python tools/eval_metric.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_evaluate.yaml --data_dir /home/anonymous/dataset --output_dir output/evaluate --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5


'''



# CUDA_VISIBLE_DEVICES=2 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb32_CLIP_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm True --adapter True
# CUDA_VISIBLE_DEVICES=3 python tools/eval_local.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm bias




'''
# measure intrinsic dimension
# CUDA_VISIBLE_DEVICES=0 python tools/eval_intrinsic_dimension.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/evaluate --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --dintrinsic 300



## direct training
# CUDA_VISIBLE_DEVICES=0 python tools/eval_direct.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm Attention
# CUDA_VISIBLE_DEVICES=0 python tools/eval_direct.py --ds experiments/eval/dataset/cifar100.yaml --model experiments/eval/model/vitb16_supervised_100_adapter.yaml --data_dir /home/anonymous/dataset --output_dir output/sup/epoch0300/transfer --model_ckpt /home/anonymous/mnist_pytorch/ViT-B:32.pt --finetune True --no-search False --lr-range 1e-5 --layernorm MLP
'''