import numpy as np
import os
import json
import glob

import matplotlib.pyplot as plt

# python Vision-Benchmark-IC/vision_benchmark/read_results.py

datasets_checked=['caltech-101',
 'cifar-10',
 'cifar-100',
 'country211',
 'dtd',
 'eurosat_clip',
 'fer-2013',
 'fgvc-aircraft-2013b-variants102',
 'food-101',
 'gtsrb',
 'hateful-memes',
 'kitti-distance',
 'mnist',
 'oxford-flower-102',
 'oxford-iiit-pets',
 'patch-camelyon',
#  'ping-attack-on-titan-plus',
#  'ping-whiskey-plus',
 'rendered-sst2',
 'resisc45_clip',
 'stanford-cars',
 'voc-2007-classification',
 'imagenet-1k'
 ]

two_lr = ['', 'two_lr']




def read_json(log_path, dataset_name='', file_prefix=''):
    
    # print(dataset_names)

    
    datasets, accs, num_para = [], [], []

    # Returns a list of names in list files.
    log_path = os.path.join(log_path, dataset_name)
    # print(log_path)
    file_filter = file_prefix + f'*.txt'
    txt_path = os.path.join(log_path, file_filter)

    # print(txt_path)
    files = glob.glob(txt_path, recursive = True)
    # print(files)


    if 'finetuning' in file_prefix and 'two_lr' not in 'finetuning':
        files = [f for f in files if 'two_lr' not in 'finetuning']
    # print(files)

    for file in files:
        
        data = ''

        # multiple dict-like string in the file
        try: 
            Lines = open(file, 'r').readlines()   
            texts = open(file, 'r').read() 
            # print(texts)
            data =  Lines[-1].strip()
            # print(data)
            data = data.split(' ')[-1].replace('%', '')
            
            accs.append( float(data) )

            # parameter_data = texts.strip().split('trainable params: ')[-1].split('M')[0]
            parameter_data = texts.strip().split('trainable params: ')[-1].split('M')[0]

            num_para.append(parameter_data)
        except:

            # print(f"Failed at {file}")
            continue
    
    return accs, num_para



# finetuning evaluation
def extract_finetune_results(proj_path, dataset_name, num_samples_per_class, rs):
    training_mode = ['finetuning']  # ['finetuning', 'linear_probe']
    # training_mode = ['linear_probe']
    accs = np.zeros([len(training_mode), len(num_samples_per_class)])
    for j in range(len(training_mode)):
        for i in range(len(num_samples_per_class)):
            file_prefix = training_mode[j] + '_' +  num_samples_per_class[i] + '_' 

            clip_results, num_para = read_json(proj_path, dataset_name, file_prefix)
            # print(dataset_name)
            # print(clip_results)
            # print(num_para)
            print(num_para[-1])
            accs[j, i] = np.mean(clip_results)
            # print(f'{dataset_name}, samples {num_samples_per_class[i]}, {clip_results[-1]}, {rs} ')
            # print(clip_results[-1])

    return accs
    
# proj_path = '../vision_benchmark/compacter_clip_output/vitb32_CLIP'
# proj_path = '../vision_benchmark/adapter_clip_output/vitb32_CLIP'
# proj_path = '../vision_benchmark/adapter_clip_output2/vitb32_CLIP'
# proj_path = '../vision_benchmark/adapter_clip_output3_no_layernorm/vitb32_CLIP'
# proj_path = '../vision_benchmark/adapter_clip_output4/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output5_trycompacter_replace_adapter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output5_trycompacter_replace_adapter_tuned/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output6_trycompacter_layer_norm_before/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output6_trycompacter_phm4/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output7_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output8_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output9_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output10_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output11_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output12_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output13_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output16_trycompacter/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output17_trycompacter_self_write_linear/vitb32_CLIP'
# proj_path = '../vision_benchmark/compacter_clip_output19_use_matvec_product'
proj_path = '../vision_benchmark/compacter_clip_output19_use_matvec_product_search'
# proj_path = '../vision_benchmark/lora_clip_output_6_real_lora_with_new_linear/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output_no_delta_try3/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output_2/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output4/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output5/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output6/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output_lora_with_activation/vitb32_CLIP'
# proj_path = '../vision_benchmark/kadapt_clip_output_lora_with_no_activation/vitb32_CLIP'
# proj_path = '../vision_benchmark/linearprobe_output/vitb32_CLIP'
# proj_path = '../vision_benchmark/linearprobe_output2/vitb32_CLIP'
# proj_path = '../vision_benchmark/finetune_output/vitb32_CLIP'


num_samples_per_class = ['5'] # ['5', '20', '50', 'full']

# random_seeds =  ['log_random_0', 'log_random_1', 'log_random_2'] # , 'random_3_sgd','random_4_sgd'
random_seeds =  ['log_random_0']
# random_seeds =  ['0']

accs_per_dataset_rs = []
for rs in random_seeds:
    proj_path_rs = os.path.join(proj_path, rs)
    accs_per_dataset = []
    for dataset_name in datasets_checked:
        accs = extract_finetune_results(proj_path_rs, dataset_name, num_samples_per_class, rs)
        accs_per_dataset.append(accs)
    accs_per_dataset_rs.append(accs_per_dataset)

