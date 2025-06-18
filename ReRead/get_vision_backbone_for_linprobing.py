
import torch
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_model', type=str)
args, unknown = parser.parse_known_args()

checkpoint = torch.load(args.path_to_model)#, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

filtered_params = {name: param for name, param in state_dict.items() if 'vision_encoder' in name}

if 'mae' in args.path_to_model.lower():
    replace_name = 'model.vision_encoder.model.'
    file_suffix = '_bscan_multimae.pth'
else:
    replace_name = 'model.vision_encoder.'
    file_suffix = '_bscan.pth'
    
updated_params = OrderedDict({name.replace(replace_name, ''): param for name, param in filtered_params.items()})

model_version = args.path_to_model.split('/')[1]

torch.save(updated_params, '../linear_probing/_weights/' + args.path_to_model.rsplit('/', 1)[-1].split('.')[0] + '_' + model_version + file_suffix)



