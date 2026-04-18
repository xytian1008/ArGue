import os.path as osp
from glob import glob
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .losses import contrastive_loss

_tokenizer = _Tokenizer()


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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_encoder_model, all_classnames):
        super().__init__()
        n_cls = len(all_classnames) if cfg.DATASET.INCLUDE_ALL_CLASSES else len(classnames)
        n_ctx = cfg.TRAINER.LASP.N_CTX
        ctx_init = cfg.TRAINER.LASP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        all_classnames = [name.replace("_", " ") for name in all_classnames]

        if cfg.DATASET.INCLUDE_ALL_CLASSES:
            # Preserve class order
            classes_delta = [name for name in all_classnames if name not in classnames]
            print(f'Number of extra class names: {len(classes_delta)}')
            classnames += classes_delta
            print(f'Number of class names after: {len(classnames)}')
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        ##### ---- LASP ----- ######
        if cfg.TRAINER.LASP.ENABLE:
            self.construct_references_lasp(cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.all_classnames = all_classnames

        if cfg.TRAINER.LASP.ENABLE_CORRECTION:
            self.w = nn.Parameter(torch.zeros(1, ctx_dim, device=embedding.device, dtype=dtype), requires_grad=self.cfg.TRAINER.LASP.TRAIN_W)


    def construct_references_lasp(self, cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype, n_ctx):
        print('Initializing LASP prompts...')
        template_prompts = cfg.TRAINER.LASP.LASP_PROMPTS
        all_classnames = [name.replace("_", " ") for name in all_classnames]
        print(f'Num classes used for LASP: {len(all_classnames)}')

        all_class_text_features = []
        for c_init in template_prompts:
            prompts = [c_init.format(name) for name in all_classnames]
            tokenized_prompts_all_c = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            text_encoder_model.cuda()
            with torch.no_grad():
                embedding_all_cls = clip_model.token_embedding(tokenized_prompts_all_c).cuda().type(dtype)
                class_text_features = text_encoder_model(embedding_all_cls, tokenized_prompts_all_c).type(dtype)
                all_class_text_features.append(class_text_features)
            
            self.register_buffer("class_text_features", torch.stack(all_class_text_features, dim=0))

        prompts = [prompt_prefix + " " + name + "." for name in all_classnames]
        tokenized_prompts_all_c_ = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_all_c_).type(dtype)

        self.register_buffer("token_prefix_all", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_all", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts_all = tokenized_prompts_all_c
        self.tokenized_prompts_all_c_ = tokenized_prompts_all_c_
        self.n_cls_all = len(prompts)
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, all=False):
        if not all:
            prefix = self.token_prefix
            suffix = self.token_suffix
            n_cls = self.n_cls
        else:
            prefix = self.token_prefix_all
            suffix = self.token_suffix_all
            n_cls = len(self.all_classnames)
        ctx = self.ctx # (n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
        
        return prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_classnames):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = nn.DataParallel(TextEncoder(clip_model))
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, self.text_encoder, all_classnames)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.loss = contrastive_loss

    def forward_text_to_text(self, image_features=None):
        with torch.no_grad():
            class_text_features = self.prompt_learner.class_text_features
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        if torch.rand(1).item() < 0.5:
            noise = 0.05 * torch.randn_like(class_text_features)
            class_text_features.add_(noise)

        prompts = self.prompt_learner(all=True)
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_all_c_)

        if self.cfg.TRAINER.LASP.ENABLE_CORRECTION:
            w = self.prompt_learner.w
            text_features = text_features + w

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0)

        label = torch.arange(self.prompt_learner.n_cls_all, device=class_text_features.device, dtype=torch.long).unsqueeze(0).expand(class_text_features.size(0), -1)
        return self.loss(text_features, class_text_features, label, t=self.logit_scale)[0]

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if self.cfg.TRAINER.LASP.ENABLE_CORRECTION:
            w = self.prompt_learner.w
            text_features = text_features + w
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        loss, logits = self.loss(image_features, text_features, label, t=self.logit_scale)

        if self.prompt_learner.training:
            if self.cfg.TRAINER.LASP.ENABLE:
                loss += self.cfg.TRAINER.LASP.LASP_LOSS_WEIGHT * self.forward_text_to_text(image_features)

        if self.prompt_learner.training:
            return loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class LASP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LASP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        all_classnames = self.dm.dataset.all_classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.LASP.PREC == "fp32" or cfg.TRAINER.LASP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, all_classnames)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        if cfg.TRAINER.LASP.FINETUNE_VIT_LN:
            print('Re-enabling LN...')
            for name, param in self.model.named_parameters():
                if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                    param.requires_grad_(True)  
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        if cfg.TRAINER.LASP.FINETUNE_VIT_LN:
            group1, group2 = [], []
            for name, param in self.model.named_parameters():
                if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                    group1.append(param)
                else:
                    group2.append(param)

            param_groups = [
                {
                    "params": group1,
                    "lr": cfg.OPTIM.LR * 0.1
                },
                {
                    "params": group2
                },
            ]
            self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        else:
            self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LASP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.LASP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if isinstance(input, list):
            input = [inp.to(self.device, non_blocking=True) for inp in input]
        else:
            input = input.to(self.device, non_blocking=True)
        label = label.to(self.device)

        if self.cfg.DATALOADER.K_TRANSFORMS > 1:
            input = torch.cat(input)
            label = label.repeat(self.cfg.DATALOADER.K_TRANSFORMS)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                print('Model not found at "{}", retrying to find one automatically...'.format(model_path))
                model_path = glob(f'{directory}/{name}/model-best.pth.tar-*')[0]
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            ignore_list = ['token_prefix', 'token_suffix', 'token_prefix_all', 'token_suffix_all', 'class_text_features']
            ignore_list += [f'prompt_learner.{il}' for il in ignore_list]

            for k in ignore_list:
                state_dict.pop(k, None)

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            w_weights = None
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in self._models[name].state_dict():
                    # if k == 'w':
                    #     w_weights = v
                    if v.size() == self._models[name].state_dict()[k].size():
                        new_state_dict[k] = v
                    else:
                        print(k, v.shape, self._models[name].state_dict()[k].size())
            print(f'Num of preserved keys: {len(new_state_dict)}')
            print(f'Keys: {new_state_dict.keys()}')
            #new_state_dict = {}
            self._models[name].load_state_dict(new_state_dict, strict=False)
        return w_weights
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        return list(results.values())[0]