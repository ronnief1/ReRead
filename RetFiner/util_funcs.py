import numpy as np
from transforms import to_min_max_norm_tensor, to_255_norm_tensor
from functools import partial
import torch



def image_loading(scan):
    full_scan = np.load('path/to/scan')#, allow_pickle=True, mmap_mode='r')

    image = to_min_max_norm_tensor(full_scan)

    return image

def image_loading_255(scan):
    full_scan = np.load('path/to/scan')#, allow_pickle=True, mmap_mode='r')

    image = to_255_norm_tensor(full_scan)

    return image

DOMAIN_CONF = {
    'bscan': {
        'channels': 1,
        'stride_level': 1,
        'input_adapter': partial(PatchedInputAdapter, num_channels=1),
        'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    }
}

def param_groups_lrd(
    model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75, num_layers=None
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if num_layers is None:
        #import pdb; pdb.set_trace()
        num_layers = len(model.multimae.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


def load_vision_backbone(vision_weights_path):

    if 'retfound' in vision_weights_path.lower():
        print('Loading RETFound weights')
        from vit import vit_large_patch16
        vision_encoder = vit_large_patch16(
            img_size=224,
            num_classes=2,
            drop_path_rate=0.1,
            global_pool=False,
        )

        print('loading weights from ', vision_weights_path)
        checkpoint = torch.load(vision_weights_path, map_location="cpu")

        checkpoint_model = checkpoint["model"]

        state_dict = vision_encoder.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = vision_encoder.load_state_dict(checkpoint_model, strict=False)
        vision_encoder.get_vision_features = vision_encoder.forward_features


    elif 'visionfm' in vision_weights_path.lower():
        print('Loading VisionFM weights')
        from VisionFM_main.models.vision_transformer import VisionTransformer
        vision_encoder = VisionTransformer(return_all_tokens=True, qkv_bias=True)
        print('loading weights from ', vision_weights_path)
        state_dict = torch.load(vision_weights_path, map_location='cpu')['teacher']
        for key in list(state_dict.keys()):
            if 'backbone.' in key:
                state_dict[key.replace('backbone.', '')] = state_dict.pop(key)

        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        vision_encoder.get_vision_features = vision_encoder.forward

    # elif 'clip' in vision_weights_path:
    #     from transformers import CLIPModel
    #     vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model

    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your vision model loading code to util_funcs.py in the function load_vision_backbone.")

    return vision_encoder

def get_model_name(model_weights):
    if 'retfound' in model_weights:
        return 'RETFound'
    elif 'visionfm' in model_weights:
        return 'VisionFM'
    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your vision model loading code to util_funcs.py in the function load_vision_backbone.")


def get_data_loaders(model_weights, args):
    from pandas import read_csv
    from ImageCaptionDataset import ImageCaptionDataset
    from torch.utils.data import DataLoader
    from transforms import collate_fn

    train_data = read_csv('path/to/data', index_col=0).reset_index(drop=True)
    val_data = read_csv('path/to/data', index_col=0).reset_index(drop=True)


    if 'retfound' in model_weights:
        from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
        transform = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Copy the single channel to 3 channels
            Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet means
                               std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ImageCaptionDataset(train_data, transform, load_255=True)
        val_dataset = ImageCaptionDataset(val_data, transform, load_255=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    elif 'visionfm' in model_weights:
        from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC

        transform = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # Copy the single channel to 3 channels
            # Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet means
            #                    std=[0.229, 0.224, 0.225])
            Normalize(mean = [0.21091926, 0.21091926, 0.21091919],
                std = [0.17598894, 0.17598891, 0.17598893])
        ])
        train_dataset = ImageCaptionDataset(train_data, transform, load_255=True)
        val_dataset = ImageCaptionDataset(val_data, transform, load_255=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=10) #Define your own dataloader
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=10) #Define your own dataloader

    else:
        raise ValueError("Undefined vision backbone. Don't know how to load. Add your data loader code to util_funcs.py in the function get_data_loaders.")

    return train_loader, val_loader

