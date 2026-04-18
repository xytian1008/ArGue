import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
from clip import clip
from fast_pytorch_kmeans import KMeans

import datasets.oxford_pets
import datasets.fgvc_aircraft
import datasets.eurosat
import datasets.caltech101
import datasets.oxford_flowers
import datasets.food101
import datasets.ucf101
import datasets.dtd
import datasets.sun397
import datasets.imagenet
import datasets.stanford_cars

import datasets.imagenetv2
import datasets.imagenet_sketch
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.argue
from trainers.losses import transpose



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.INCLUDE_ALL_CLASSES = False
    cfg.DATASET.ATTR = []
    cfg.DATASET.BIAS = [["background"] for idx in range(100)]

    cfg.DATALOADER.SELECTION = True
    cfg.TRAINER.ARGUE = CN()
    cfg.TRAINER.ARGUE.PREC = "amp"  # fp16, fp32, amp
    cfg.DATASET.OUTPUT_DIR = "/workspace/ArGue/temp.txt"

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)


    return cfg

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    model = load_clip_to_cpu(cfg)
    model.to('cuda')
    if cfg.TRAINER.ARGUE.PREC == "fp32" or cfg.TRAINER.ARGUE.PREC == "amp":
        model.float()
    dtype = model.dtype

    sub_attrs = []

    n_cls = len(cfg.DATASET.ATTR)
    for i in range(n_cls):
        attrs = cfg.DATASET.ATTR[i]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in attrs]).to('cuda')
        text_features = model.encode_text(tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        kmeans = KMeans(n_clusters=2, mode='cosine', verbose=1, init_method = 'fixed')
        cluster_labels = kmeans.fit_predict(text_features)
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            label = label.item()
            if label not in clusters.keys():
                clusters[label] = [attrs[idx]]
            else:
                clusters[label].append(attrs[idx])

        cfg.DATASET.CUR_IDX = i
        dm = DataManager(cfg)
        batch = next(iter(dm.train_loader_x))
        input = batch["img"]
        if isinstance(input, list):
            input = [inp.to('cuda', non_blocking=True) for inp in input]
        else:
            input = input.to('cuda', non_blocking=True)
        image_features = model.encode_image(input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        temp_attrs = []
        for key, cluster in clusters.items():
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in cluster]).to('cuda')
            cluster_features = model.encode_text(tokenized_prompts)
            cluster_features = cluster_features / cluster_features.norm(dim=-1, keepdim=True)
            logit = (model.logit_scale.exp() * cluster_features @ transpose(image_features)).mean(dim = -1)
            temp_attrs.append(cluster[logit.argmax()])
        sub_attrs.append(temp_attrs)
    with open(cfg.DATASET.OUTPUT_DIR, "w") as output:
        output.write(str(sub_attrs))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)