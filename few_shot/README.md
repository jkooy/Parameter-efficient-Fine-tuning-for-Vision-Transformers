# Parameter-efficient Fine-tuning for Vision Transformers

This repository contains the code for few-shot experiments. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Running
The implementations for all methods can be found in the ./vision_benchmark repo
To run the method(s) in the paper, all bash files can be found in the ./scripts/ repo 

For example, for Adapter-tuning using CLIP pretrained ViT, run commands:
```train
bash scripts/adapter_clip.sh
```

## Results

The outputs will be saved at [output_dir] specified in the bash files.

Run python vision_benchmark/read_results.py to read them.
