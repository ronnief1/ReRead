from copy import deepcopy
import json
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import datetime
import numpy as np
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.backends.cudnn as cudnn
from torchvision import models

from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from torchvision.models import vit_b_16, vit_b_32, vit_l_16

from src.retfound.vit import vit_large_patch16, vit_base_patch16

from src.utils.dataset import build_dataset
# from src.retfound.pos_embed import interpolate_pos_embed
import src.retfound.lr_decay as lrd
import src.retfound.misc as misc
from src.utils.train_eval import *
from src.mim_oct.models.vit.ViT import Attention, VisionTransformer
import warnings
from functools import partial

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Retinal image classification experiments", add_help=False
    )

    # Model parameters
    # parser.add_argument(
    #     "--model",
    #     default="vit_large_patch16",
    #     type=str,
    #     metavar="MODEL",
    #     help="Name of model to train",
    # )
    parser.add_argument(
        "--input_size",
        default=224,
        type=int,
        help="Images input size",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-8,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default: 0.1). If 0 -> CrossEntropyLoss.",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints).",
    )

    # * Supervised training params
    parser.add_argument(
        "--linear_probing",
        default="false",
        type=str,
        help="Set to True for not training the encoder weights.",
    )
    parser.add_argument(
        "--label_efficiency_exp",
        default=False,
        type=bool,
        help="Set to True for label efficiency experiments.",
    )
    parser.add_argument(
        "--init_weights",
        default="_weights/RETFound_oct_weights.pth",
        type=str,
        help="Pre-trained weights to initialise the model with. Default: RetFound.",
    )
    parser.add_argument(
        "--resume", default="", help="Resume from checkpoint - for supervised training."
    )
    parser.add_argument("--task", default="", type=str, help="name of the experiment")
    parser.add_argument(
        "--global_pool",
        default=True,
        type=bool,
        help="By default we're using global pooling instead of the CLS token for classifier input.",
    )
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     default="__results/retfound_exp/",
    #     help="path where to save, empty for no saving",
    # )

    # * Dataset parameters
    parser.add_argument(
        "--data_root",
        default="/path/to/data",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--data_set", default="", type=str, help="dataset folder name")
    parser.add_argument(
        "--nb_classes", default=2, type=int, help="Number of the classification types. 0 for automatic counting."
    )
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # * Training parameters
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus). 0 for automatic calculation.",
    )
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument(
        "--eval",
        default="false",
        type=str,
        help="Set to True for only running the evaluation on the test set.",
    )
    parser.add_argument(
        "--early_stopping_epochs",
        default=200,
        type=int,
        help="Parameter to control how many epochs to wait for the validation loss to improve before stopping..",
    )
    parser.add_argument(
        "--early_stopping_delta",
        default=0.01,
        type=float,
        help="Parameter to specify the minimum change in the validation loss required to consider it an improvement.",
    )
    parser.add_argument(
        "--early_stopping_delta_two",
        default=0.01,
        type=float,
        help="Parameter to specify the minimum change in the validation loss required to consider it an improvement.",
    )
    parser.add_argument(
        "--early_start_from",
        default=0,
        type=int,
        help="Parameter to specify the epoch to start early stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
    )
    # parser.add_argument(
    #     "--rot",
    #     action="store_true",
    # )
    parser.add_argument(
        "--all-tokens",
        action="store_true",
    )
    parser.add_argument(
        "--input-modality",
        default="bscan",
    )
    # parser.add_argument(
    #     "--scaler",
    #     default="z-score",
    #     type=str,
    #     help="Scaler to use for the input images. Options: z-score, min-max, none.",
    # )
    parser.add_argument(
        "--version",
        required=True,
    )
    parser.add_argument(
        '--get-embeddings',
        type=str,
        default=None,
        choices=['test', 'all'],
    )
    parser.add_argument(
        "--dont-override-args",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        '--val_metric',
        default='loss',
        type=str,
    )
    parser.add_argument(
        '--val_metric_two',
        default='bacc',
        type=str,
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
    )
    parser.add_argument(
        '--fill',
        default=None,
        type=int,
    )
    parser.add_argument(
        '--no_affine',
        action='store_true',
    )
    parser.add_argument(
        '--no_minmax',
        action='store_true',
    )
    parser.add_argument(
        '--ft_weights',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--representation_method', #should be one of 'patch_features', 'cls_token', 'concat' or 'all_tokens'
        default=None,
        type=str,
    )

    return parser


def main(args):
    val_stats_names = [
        "epoch", "loss", "bacc", "auroc", "ap", "f1", "mcc",
    ]

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    cudnn.benchmark = True

    dataset_train = None
    dataset_val = None
    if not args.eval:
        if args.get_embeddings is not None:
            augment_train = False
            shuffle = False
        else:
            augment_train = True
            shuffle = True
        dataset_train = build_dataset(subset="train", args=args, augment=augment_train)
        print(dataset_train.class_to_idx)
        train_loader = DataLoader(
            dataset_train,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f"Number of training samples: {len(dataset_train)}")

        dataset_val = build_dataset(subset="val", args=args, augment=False)
        valid_loader = DataLoader(
            dataset_val,
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f"Number of validation samples: {len(dataset_val)}")
    else:
        train_loader = None
        valid_loader = None

    if 'cross_train' not in args.data_set.lower():
        dataset_test = build_dataset(subset="test", args=args, augment=False)
        if args.get_embeddings == 'all':
            # Join all data in a single dataloader
            if not args.eval and dataset_train is not None and dataset_val is not None:
                dataset_test = ConcatDataset(
                    [dataset_train, dataset_val, dataset_test]
                )
        test_loader = DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        print(f"Number of test samples: {len(dataset_test)}")
    else:
        test_loader = None

    if args.init_weights == "vit_b_32":
        model = vit_b_32()
        model.load_state_dict(torch.load("./_weights/vit_b_32-d86f8d99.pth"))
        print("Loaded ImageNet weights for ViT-B/32")
        model.heads.head = nn.Linear(768, args.nb_classes)

    elif args.init_weights == "vit_b_16":
        model = vit_b_16()
        model.load_state_dict(torch.load("./_weights/vit_b_16-c867db91.pth"))
        print("Loaded ImageNet weights for ViT-B/16")
        model.heads.head = nn.Linear(768, args.nb_classes)

    elif args.init_weights == 'vit_l_16':
        model = vit_l_16()
        model.load_state_dict(torch.load('./_weights/vit_l_16-852ce7e3.pth'))
        print("Loaded ImageNet weights for ViT-L/16")
        model.heads.head = nn.Linear(1024, args.nb_classes)

    elif args.init_weights.lower() == 'clip':
        from transformers import CLIPModel, CLIPConfig
        print('Using CLIP')
        config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch16")
        config.vision_config.linear_probing = True
        config.vision_config.feat_concat = True

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", config=config).vision_model

        if args.ft_weights:
            state_dict = torch.load(args.ft_weights, map_location='cpu')
            print('Using ft weights from', args.ft_weights)
            model.load_state_dict(state_dict, strict=False)

        model.head = nn.Linear(in_features=1536, out_features=args.nb_classes)
        trunc_normal_(model.head.weight, std=2e-5)

    elif 'vfm_oct' in args.init_weights.lower():
        #VisionFM implementation
        from src.VisionFM_main.models.vision_transformer import VisionTransformer
        if args.ft_weights:

            print('Loading VisionFM finetuned weights from', args.ft_weights)

            state_dict = torch.load(args.ft_weights, map_location='cpu')
            model = VisionTransformer(feat_concat=True, num_classes=args.nb_classes, qkv_bias=True, use_norm=True)# norm is commented out in source code. try also adding own norm

            for key in list(state_dict.keys()):
                if 'backbone.' in key:
                    state_dict[key.replace('backbone.', '')] = state_dict.pop(key)

            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)

        else:
            print("Loading VisionFM weights...")
            model_disp_name = "VisionFM"
            weights_fn = './_weights/VFM_OCT_weights.pth'
            state_dict = torch.load(weights_fn)['teacher']
            for key in list(state_dict.keys()):
                if 'backbone.' in key:
                    state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
            model = VisionTransformer(feat_concat=True, num_classes=args.nb_classes, qkv_bias=True)
            msg = model.load_state_dict(state_dict, strict=False)

        in_dim = 1536 #if args.representation_method == 'concat' else 768

        model.head = nn.Linear(in_features=in_dim, out_features=args.nb_classes)
        trunc_normal_(model.head.weight, std=2e-5)

       #RETFound timm implementation
#             if args.ft_weights is None:
#                 state_dict = torch.load(args.init_weights)['teacher']
#             else:
#                 state_dict = torch.load(args.ft_weights)
# #            print(' > Loaded from', Path(weights_fn).name)
#             for key in list(state_dict.keys()):
#                 if 'backbone.' in key:
#                     state_dict[key.replace('backbone.', '')] = state_dict.pop(key)
#                 # if args.init_weights is not None and 'ln.' in key:
#                 #     state_dict[key.replace('ln.', 'norm.')] = state_dict.pop(key)
#             model = vit_base_patch16(
#                 img_size=args.input_size,
#                 num_classes=args.nb_classes,
#                 drop_path_rate=args.drop_path,
#                 representation_method=args.representation_method,
#                 use_ln=True
#             )
#             msg = model.load_state_dict(state_dict, strict=False)
#             print(msg)
#             if args.representation_method == 'concat':
#                     model.head = nn.Linear(in_features=1536, out_features=args.nb_classes)
#             trunc_normal_(model.head.weight, std=2e-5)
        # assert set(msg.missing_keys) == {
        #     "head.weight",
        #     "head.bias",
        # }

    elif args.init_weights in ['vit_l_16_21k', 'vit_b_16_21k']:
        # model = timm.create_model('hf_hub:timm/vit_large_patch16_224.orig_in21k', pretrained=True)
        # load from ./_weights/timm--vit_base_patch16_224.orig_in21k.bin
        if args.init_weights == 'vit_l_16_21k':
            model_fun = vit_large_patch16
            state_dict_fn = './_weights/timm--vit_large_patch16_224.orig_in21k.bin'
            model_name = 'ViT-L'
        else:
            model_fun = vit_base_patch16
            state_dict_fn = './_weights/timm--vit_base_patch16_224.orig_in21k.bin'
            model_name = 'ViT-B'
        print(f'Loading 21k pre-trained weights for {model_name}/16...')
        model = model_fun(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        state_dict = torch.load(state_dict_fn)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            import re
            for line in str(e).split('\n'):
                line = line.strip()
                if line.startswith('Missing key'):
                    # Get all keys in double quotes
                    missing_keys = re.findall(r'"(.*?)"', line)
                    for key in missing_keys:
                        if 'head' not in key and 'fc_norm' not in key:
                            raise ValueError(e)
                    print("\tMissing keys:", missing_keys)
                elif line.startswith('Unexpected key'):
                    unexpected_keys = re.findall(r'"(.*?)"', line)
                    print("\tUnexpected keys:", unexpected_keys)
                    for key in unexpected_keys:
                        if 'norm' not in key:
                            raise ValueError(e)
            model.load_state_dict(state_dict, strict=False)
        print('> Loaded')
        if model_name == 'ViT-L':
            embed_dim = 1024
        else:
            embed_dim = 768
        model.head = nn.Linear(embed_dim, args.nb_classes)

    elif args.init_weights == "dinov2_vitb14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        print("Loaded ImageNet weights for DINOv2 ViT-B/14")
        model.head = nn.Linear(768, args.nb_classes)

    elif args.init_weights == 'dinov2_vitl14':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        print("Loaded ImageNet weights for DINOv2 ViT-L/14")
        model.head = nn.Linear(1024, args.nb_classes)

    elif "convnext" in args.init_weights:
        if args.init_weights == "convnext_base":
            model = models.convnext_base(pretrained=True)
            model.classifier[2] = nn.Linear(1024, args.nb_classes)
        elif args.init_weights == "convnext_small":
            model = models.convnext_small(pretrained=True)
            model.classifier[2] = nn.Linear(768, args.nb_classes)
        else:
            raise ValueError("Unknown ConvNext model.")


    else:
        model = vit_large_patch16(
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            representation_method=args.representation_method,
        )

        # load pre-trained weights if path provided
        if args.ft_weights:
            checkpoint = torch.load(args.ft_weights, map_location="cpu")
            print("Load pre-trained checkpoint from: %s" % args.ft_weights)

            checkpoint_model = checkpoint
            del checkpoint_model['head.bias']
            del checkpoint_model['head.weight']
            del checkpoint_model['proj.bias']
            del checkpoint_model['proj.weight']

            state_dict = model.state_dict()
            msg = model.load_state_dict(checkpoint_model, strict=False)

                    # manually initialize fc layer
            if args.representation_method == 'concat':
                model.head = nn.Linear(in_features=2048, out_features=args.nb_classes)
            trunc_normal_(model.head.weight, std=2e-5)

        else:
            checkpoint = torch.load(args.init_weights, map_location="cpu")

            print("Load pre-trained checkpoint from: %s" % args.init_weights)
            # >> RETFound weights
            if ("retfound" in str(args.init_weights).lower()):
                checkpoint_model = checkpoint["model"]
            else:
                checkpoint_model = checkpoint

            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            # interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)

            if not args.eval and args.get_embeddings is None:
                if args.global_pool:
                    assert set(msg.missing_keys) == {
                        "head.weight",
                        "head.bias",
                        "fc_norm.weight",
                        "fc_norm.bias",
                    }
                else:
                    assert set(msg.missing_keys) == {"head.weight", "head.bias"}

                if args.representation_method == 'concat':
                    model.head = nn.Linear(in_features=2048, out_features=args.nb_classes)
                # manually initialize fc layer
                trunc_normal_(model.head.weight, std=2e-5)

    if args.linear_probing:
        print('Freezing encoder layers for linear probing')
        # freeze encoder layers for linear probing
        for name, param in model.named_parameters():
            if "head." not in name: # and "norm." not in name:
                param.requires_grad = False
            else:
                print('Letting ', name, ' train.')

    else:
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)

    # print model info
    n_parameters = sum(p.numel() for p in model.parameters())
    n_tr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model))
    print("number of params (N): %.2e" % (n_parameters))
    print("number of params (N):", n_parameters)
    print("number of trainable params (M): %.2e" % (n_tr_parameters))
    print("number of trainable params (M):", n_tr_parameters)

    # Save args to output_dir
    if args.output_dir:
        with open(f"{args.output_dir}/{args.task}/args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    if args.dry_run:
        print("Dry run. Exiting.")
        sys.exit(0)

    if args.get_embeddings is not None:
        if args.eval and args.get_embeddings == 'test':
            # Evaluate on the best checkpoint
            args.resume = f"{args.output_dir}/{args.task}/checkpoint-best-model.pth"
            misc.load_model(args=args, model=model, optimizer=None)
            save_path = Path(f"{args.output_dir}/embeddings/ft_test")
        elif not args.eval and args.get_embeddings == 'test':
            save_path = Path(f"{args.output_dir}/embeddings/test")
        else:
            save_path = Path(f"{args.output_dir}/embeddings/all")
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            model.forward_features
        except AttributeError:
            model.heads = nn.Sequential(nn.Identity())
        test_stats = evaluate(
            model, test_loader, "Best", device, args.nb_classes, mode="Test",
            get_embeddings=True, save_path=save_path
        )
        exit(0)
    elif args.save_predictions:
        print("Getting predictions for the best checkpoint")
        args.resume = f"{args.output_dir}/{args.task}/checkpoint-best-model.pth"
        misc.load_model(args=args, model=model, optimizer=None)
        save_path = args.output_dir
        test_stats = evaluate(
            model, test_loader, "Best", device, args.nb_classes, mode="Test",
            save_predictions=True, save_path=save_path
        )
        exit(0)

    if args.full_finetune or args.linear_probing:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(
            model,
            args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=args.layer_decay,
            num_layers=None,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    if not args.eval:
        if args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # if args.val_metric != 'loss':
        #     greater_is_better = True
        # else:
        #     greater_is_better = False

        greater_is_better = args.val_metric != 'loss'
        greater_is_better_two = args.val_metric_two != 'loss'

        # Initialize early stopping object
        early_stopping = EarlyStopping(
            patience=args.early_stopping_epochs,
            delta=args.early_stopping_delta,
            delta_two=args.early_stopping_delta_two,
            greater_is_better=greater_is_better,
            greater_is_better_two=greater_is_better_two,
            start_from=args.early_start_from,
        )

        print(f"Start training for {args.epochs} epochs")
        print("Early stopping:")
        print(f"  patience: {args.early_stopping_epochs}")
        print(f"  start_from: {args.early_start_from}")
        print(f"  delta: {args.early_stopping_delta}")
        print(f"  metric: {args.val_metric}")
        print(f"  greater_is_better: {greater_is_better}")
        print(f"  delta_two: {args.early_stopping_delta}")
        print(f"  metric_two: {args.val_metric_two}")
        print(f"  greater_is_better_two: {greater_is_better_two}")
        start_time = time.time()
        train_stats_all, val_stats_all = [], []
        best_model = argparse.Namespace()
        assert train_loader is not None
        assert valid_loader is not None
        for epoch in range(args.start_epoch, args.epochs):
            try:
                train_stats = train_1_epoch(
                    model,
                    criterion,
                    train_loader,
                    optimizer,
                    device,
                    epoch,
                    args=args,
                )
            except ValueError as e:
                print('Early stopping')
                print(e)
                break

            train_stats_all.append(train_stats)

            val_stats = evaluate(
                model, valid_loader, epoch, device, args.nb_classes,
                mode="Valid", args=args
            )
            val_stats_all.append(val_stats)

            # if the validation loss has improved, save checkpoint
            # Check if early stopping criterion is met
            assert val_stats is not None
            idx = val_stats_names.index(args.val_metric)
            idx_two = val_stats_names.index(args.val_metric_two)
            is_best = early_stopping(val_stats[idx], val_stats[idx_two], epoch)
            if early_stopping.early_stop:
                print(f"Early stopping @ epoch {epoch}")
                break
            else:
                if is_best and args.output_dir:
                    # Save in memory to avoid writing to disk all the time
                    # NOTE: If you only plan to keep the best performing model (according to the acquired validation loss), don’t forget that best_model_state = model.state_dict() returns a reference to the state and not its copy! You must serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) otherwise your best best_model_state will keep getting updated by the subsequent training iterations. As a result, the final model state will be the state of the overfitted model.
                    best_model= argparse.Namespace(
                        model=deepcopy(model.state_dict()),
                        optimizer=deepcopy(optimizer.state_dict()),
                        epoch=epoch,
                    )
                    # misc.save_model(args, epoch, model, optimizer)
                    print(f"New best model ({early_stopping.best_value}, {early_stopping.best_value_two}) @ epoch {epoch}")

        misc.save_model(args, epoch=best_model.epoch, model=best_model.model, optimizer=best_model.optimizer)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Save evaluation results
        pd.DataFrame(
            data=train_stats_all, columns=["Epoch", "Loss", "BAcc", "F1-score"]
        ).to_csv(f"{args.output_dir}/{args.task}/train_eval.csv", index=False)

        pd.DataFrame(
            data=val_stats_all,
            columns=["Epoch", "Loss", "BAcc", "AUROC", "AP", "F1-score", "MCC"],
        ).to_csv(f"{args.output_dir}/{args.task}/valid_eval.csv", index=False)

    if test_loader is not None:
        # Evaluate on the best checkpoint
        args.resume = f"{args.output_dir}/{args.task}/checkpoint-best-model.pth"
        misc.load_model(args=args, model=model, optimizer=optimizer)
        test_stats = evaluate(
            model, test_loader, "Best", device, args.nb_classes, mode="Test", save_predictions=True, save_path=args.output_dir
        )
        pd.DataFrame(
            data=[test_stats],
            columns=["Epoch", "Loss", "BAcc", "AUROC", "AP", "F1-score", "MCC"],
        ).to_csv(f"{args.output_dir}/{args.task}/test_eval.csv", index=False)



def override_args(args):
    print("!"*80)
    print("WARNING: Overriding args. Some of the parsed args will be changed!")

    # Check model-dataset compatibility
    exit_p = False
    if "oct_slo" in args.data_set.lower():  # bscan+slo datasets
        if not (
            "bscan-slo" in args.init_weights.lower()
            or "rgb-depth" in args.init_weights.lower()
        ):
            exit_p = True
    elif "slo" in args.data_set.lower():  # slo datasets
        if "_bscan_" in args.init_weights.lower():
            exit_p = True
    else:  # bscan datasets
        if "_slo_" in args.init_weights.lower():
            exit_p = True
    if exit_p:
        print("\nERROR: incompatible model and dataset. Skipping.")
        print(args.init_weights, args.data_set)
        print("!"*80)
        time.sleep(3)
        exit(0)
    print("!"*80)
    time.sleep(3)

    if "_bscan_" in args.init_weights:
        args.input_modality = "bscan"
    elif "_slo_" in args.init_weights:
        args.input_modality = "slo"
    elif "_bscan-slo" in args.init_weights:
    # also for "_bscan-slo-bscanlayermap_"
        if "oct_slo" in args.data_set.lower():
            args.input_modality = "bscan-slo"
        elif "slo" in args.data_set.lower():
            args.input_modality = "slo"
        else:
            args.input_modality = "bscan"
    elif "_rgb_" in args.init_weights:
        args.input_modality = "rgb"
    elif "_rgb-depth-semseg_" in args.init_weights:
        if "oct_slo" in args.data_set.lower():
            args.input_modality = "rgb-depth"
        else:
            args.input_modality = "rgb"

    elif "retfound" in args.init_weights.lower():
        args.input_size = 224
        if args.linear_probing:
            args.lr = 1e-5
        else:
            if 'octdl' in args.data_set.lower():
                args.lr = 1e-2
                if args.save_predictions:
                    args.lr = 1e-4
            else:
                args.lr = 1e-4
        args.weight_decay = 1e-2
        args.full_finetune = False
        args.all_tokens = False
        # args.scaler = 'z-score'

    else:
        args.input_size = 224
        args.lr = 1e-5
        args.weight_decay = 1e-2
        args.full_finetune = True
        args.all_tokens = False
        # args.scaler = 'z-score'

    # IMPORTANT: DO NOT USE ALL TOKENS EVER!
    args.all_tokens = False

    # if 'oct_slo' in args.data_set.lower():
    #     args.rot = 'channel'

    # if 'kermany' in args.data_set.lower():
    #     args.rot = 'none'

    if args.linear_probing:
        print("Linear probing")
        args.full_finetune = False
        # NOTE: Works way better with a higher learning rate like the
        # following.
        args.lr = 1e-3
    return args



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    args.linear_probing = (args.linear_probing.lower() == "true")
    args.eval = (args.eval.lower() == "true")

    # IMPORTANT: Change args depending on the model
    if not args.dont_override_args:
        args = override_args(args)

    hostname = socket.gethostname()
    if hostname == "hemingway":
        print(f"Running on {hostname}")
        args.data_root = "/path/to/data"
        args.batch_size = 2

    args.data_path = args.data_root + args.data_set

    train_data_path = args.data_path + "/train"
    num_samples = 0
    no_nb_classes = args.nb_classes == 0
    for class_dir in Path(train_data_path).iterdir():
        if no_nb_classes and class_dir.is_dir():
            args.nb_classes += 1
        for img in class_dir.iterdir():
            num_samples += 1
    print(f"Number of classes: {args.nb_classes}")
    print(f"Number of samples: {num_samples}")
    # Batch size is closest power of 2 to 1/10 of the dataset
    if args.batch_size == 0:
        min_batch_size = 64
        # if 'kermany' in args.data_set.lower():
        #     min_batch_size = 128
        args.batch_size = min(min_batch_size, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f"Batch size: {args.batch_size}")

    args.output_dir = "__results/"
    args.output_dir += f"{args.version}/"
    args.output_dir += f"{args.seed}/"
    if args.linear_probing:
        args.output_dir += "linear/"
    else:
        args.output_dir += "finetune/"
    args.output_dir += f"{args.data_set}/"
    if "retfound" in args.init_weights.lower():
        if args.ft_weights:
            import re
            new_model_name = re.search('best-model_(.*).pth', args.ft_weights).group(1)
            args.output_dir += new_model_name
        else:
            args.output_dir += "RETFound_baseline"

    elif "mimoct" in args.init_weights.lower():
        if args.ft_weights:
            import re
            new_model_name = re.search('best-model_(.*).pth', args.ft_weights).group(1)
            args.output_dir += new_model_name
        else:
            args.output_dir += "MIMOCT_baseline"
    elif "vfm_oct" in args.init_weights.lower():
        if args.ft_weights:
            import re
            new_model_name = re.search('best-model_(.*).pth', args.ft_weights).group(1)
            args.output_dir += new_model_name
        else:
            args.output_dir += "VisionFM_baseline"
    elif args.init_weights == "vit_b_16":
        args.output_dir += "ViT_B_16"
    elif args.init_weights == 'vit_l_16':
        args.output_dir += 'ViT_L_16'
    elif args.init_weights == 'vit_l_16_21k':
        args.output_dir += 'ViT_L_16_21k'
    elif args.init_weights == 'vit_b_16_21k':
        args.output_dir += 'ViT_B_16_21k'
    elif args.init_weights == "dinov2_vitb14":
        args.output_dir += "DINOv2_ViT_B_14"
    elif args.init_weights == 'dinov2_vitl14':
        args.output_dir += 'DINOv2_ViT_L_14'
    else:
        if args.ft_weights is not None and 'vision-language' in args.ft_weights:
            args.output_dir += "_ft--vision-language"
        if args.ft_weights is not None and 'best-model' in args.ft_weights:
            import re
            new_model_name = re.search('best-model_(.*).pth', args.ft_weights).group(1)
            args.output_dir += new_model_name
        if args.ft_weights is None:
            args.output_dir += "_baseline"
        args.output_dir += f"_{args.input_size}"
    # learning rate string always in scientific notation
    lr_str = "_{:.0e}".format(args.lr)
    args.output_dir += lr_str
    if args.full_finetune:
        args.output_dir += "_fullfinetune"
    else:
        args.output_dir += "_finetune"
    if args.all_tokens:
        args.output_dir += "_alltokens"

    if args.representation_method is not None:
        args.output_dir += "_" + args.representation_method


    # Print configuration
    args_dict = vars(args)
    # Order alphabetically
    args_dict = dict(sorted(args_dict.items()))
    print(json.dumps(args_dict, indent=2))

    if args.output_dir:
        print(f"> Saving to {args.output_dir}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{args.output_dir}/{args.task}").mkdir(parents=True, exist_ok=True)

    if ((Path(args.output_dir) / 'test_eval.csv').exists()
        and (args.get_embeddings is None)
        and not args.overwrite
        and not args.save_predictions
    ):
        print("Experiment already run. Exiting.")
        sys.exit(0)

    if ((Path(args.output_dir) / 'predictions.npz').exists()
        and args.save_predictions
        and not args.overwrite
    ):
        print("Predictions already saved. Exiting.")
        sys.exit(0)

    main(args)
