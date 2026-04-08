import copy
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import time
import torchsort
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
import sys

_tokenizer = _Tokenizer()

def save_matrix(matrix):

    numpy_matrix = matrix.numpy()

    df = pd.DataFrame(numpy_matrix)

    df.to_csv("matrix.csv", index=False)

    print("save matrix successfully")

def save_parameter_names(model):
    param_names = [name for name, _ in model.named_parameters()]
    with open("clip_parameter_names.txt", "w") as f:
        for name in param_names:
            f.write(name + "\n")

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:

        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if not zero_shot_model:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": cfg.TRAINER.PLKR.PROMPT_DEPTH_VISION,
            "language_depth": cfg.TRAINER.PLKR.PROMPT_DEPTH_TEXT,
            "vision_ctx": cfg.TRAINER.PLKR.N_CTX_VISION,
            "language_ctx": cfg.TRAINER.PLKR.N_CTX_TEXT,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:

        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model

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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        assert cfg.TRAINER.PLKR.PROMPT_DEPTH_TEXT >= 1, (
            "In Independent VL prompting, Language prompt depth should be >=1"
            "\nPlease use VPT trainer if you want to learn only vision "
            "branch"
        )
        n_ctx = cfg.TRAINER.PLKR.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PLKR.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:

            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx

            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():

                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

            prompt_prefix = ctx_init
        else:

            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(
            f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PLKR.N_CTX_VISION}"
        )
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )

        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            self.ZS_image_encoder = clip_model_temp_image.visual

            all_teacher_features = []

            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.class_feature_dict = {}

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:

            fixed_embeddings = (
                self.prompt_learner.fixed_embeddings
            )

            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(
                dim=-1, keepdim=True
            )

            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(
                    image.type(self.dtype)
                )

                for i, lbl in enumerate(label):
                    if lbl.item() not in self.class_feature_dict:
                        self.class_feature_dict[lbl.item()] = []
                    self.class_feature_dict[lbl.item()].append(zero_shot_features[i])

                prototype_features = torch.zeros_like(zero_shot_features)

                for i, lbl in enumerate(label):
                    class_features = self.class_feature_dict[lbl.item()]
                    prototype = torch.mean(torch.stack(class_features), dim=0)
                    prototype_features[i] = prototype

                prototype_features = prototype_features / prototype_features.norm(
                    dim=-1, keepdim=True
                )

            return (
                F.cross_entropy(logits, label),
                text_features,
                fixed_embeddings,
                prototype_features,
                image_features,
            )
        else:
            return logits

def generate_mixup(features):
    num_samples, num_features = features.shape
    num_combinations = num_samples * (num_samples - 1) // 2

    new_features = torch.zeros(
        num_samples + num_combinations, num_features, device=features.device
    )

    new_features[:num_samples] = features

    count = num_samples
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            _lambda = torch.rand(
                1, device=features.device
            )
            pij = _lambda * features[i] + (1 - _lambda) * features[j]
            new_features[count] = pij
            count += 1

    return new_features

def efficient_mixup_double(features1, features2, max_added_samples=None):

    assert features1.device == features2.device, "两个feature矩阵必须在同一个设备上"
    assert features1.shape == features2.shape, "两个feature矩阵的形状必须相同"

    device = features1.device
    num_samples, num_features = features1.shape
    num_combinations = num_samples * (num_samples - 1) // 2

    if max_added_samples is None or max_added_samples > num_combinations:
        max_added_samples = num_combinations

    lambdas = torch.full((max_added_samples, 1), 0.5, device=device)

    new_features1 = torch.zeros(
        num_samples + max_added_samples, num_features, device=device
    )
    new_features1[:num_samples] = features1

    new_features2 = torch.zeros(
        num_samples + max_added_samples, num_features, device=device
    )
    new_features2[:num_samples] = features2

    count = 0
    for i in range(num_samples):
        n_pairs = num_samples - (i + 1)
        pairs_to_add = min(n_pairs, max_added_samples - count)

        new_features1[num_samples + count : num_samples + count + pairs_to_add] = (
            lambdas[count : count + pairs_to_add] * features1[i]
            + (1 - lambdas[count : count + pairs_to_add])
            * features1[i + 1 : i + 1 + pairs_to_add]
        )

        new_features2[num_samples + count : num_samples + count + pairs_to_add] = (
            lambdas[count : count + pairs_to_add] * features2[i]
            + (1 - lambdas[count : count + pairs_to_add])
            * features2[i + 1 : i + 1 + pairs_to_add]
        )

        count += pairs_to_add
        if count >= max_added_samples:
            break

    return new_features1, new_features2

def compute_edge_L1_changes_and_mean(features1, features2):

    distance_matrix1 = torch.cdist(features1, features1, p=1)

    distance_matrix2 = torch.cdist(features2, features2, p=1)

    edge_changes = torch.abs(distance_matrix1 - distance_matrix2)

    upper_triangle_elements = edge_changes.triu(diagonal=1)

    non_zero_indices = torch.nonzero(upper_triangle_elements)

    non_zero_values = upper_triangle_elements[
        non_zero_indices[:, 0], non_zero_indices[:, 1]
    ]

    mean_of_non_zero_changes = torch.mean(non_zero_values)

    return mean_of_non_zero_changes

def compute_edge_changes_and_mean(features1, features2):

    distance_matrix1 = 1.0 - torch.mm(features1, features1.t())

    distance_matrix2 = 1.0 - torch.mm(features2, features2.t())

    edge_changes = torch.abs(distance_matrix1 - distance_matrix2)

    upper_triangle_elements = edge_changes.triu(diagonal=1)

    mean_of_changes = torch.mean(upper_triangle_elements)

    return mean_of_changes

def spearman_correlation_loss(x, y):
    n = x.size(0)

    rank_x = x
    rank_y = y

    d = rank_x - rank_y
    d_squared = d**2

    rho = 1 - (6 * d_squared.sum(dim=0) / (n * (n**2 - 1)))

    loss = 1 - rho.mean()

    return loss

def compute_distance_matrix(features, p=2):
    dist_matrix = torch.cdist(features, features, p=p)
    return dist_matrix

def neighborhood_consistency_loss(ft_new, ft_old, p=2):

    dist_matrix_new = compute_distance_matrix(ft_new, p)
    dist_matrix_old = compute_distance_matrix(ft_old, p)

    soft_sorted_indices_new = torchsort.soft_rank(
        dist_matrix_new, regularization_strength=0.0001
    )
    soft_sorted_indices_old = torchsort.soft_rank(
        dist_matrix_old, regularization_strength=0.0001
    )

    end_loss = spearman_correlation_loss(
        soft_sorted_indices_new, soft_sorted_indices_old
    )

    return end_loss

def euclidean_distance(tensor_a, tensor_b):

    diff = tensor_a - tensor_b

    distances = torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True))

    average_distance = torch.mean(distances)
    return average_distance

def cosine_similarity(tensor_a, tensor_b):

    norm_a = torch.linalg.norm(tensor_a, dim=1, keepdims=True)
    norm_b = torch.linalg.norm(tensor_b, dim=1, keepdims=True)

    dot_product = torch.sum(tensor_a * tensor_b, dim=1, keepdims=True)

    cosine_similarity = dot_product / (norm_a * norm_b)

    cosine_distance = torch.abs(1 - cosine_similarity)

    average_cosine_distance = torch.mean(cosine_distance)

    return average_cosine_distance

@TRAINER_REGISTRY.register()
class PLKR(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLKR.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PLKR.PREC == "fp32" or cfg.TRAINER.PLKR.PREC == "amp":

            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:

                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.PLKR.PREC == "amp" else None
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        print("LOSS_Type:", self.cfg.TRAINER.PLKR.LOSS_TYPE)
        print("Mixup:", self.cfg.TRAINER.PLKR.MIXUP)

        print(
            "TEXT_Vertex_LOSS_WEIGHT:",
            self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT,
        )
        print("TEXT_Edge_LOSS_WEIGHT:", self.cfg.TRAINER.PLKR.TEXT_EDGE_LOSS_WEIGHT)
        print(
            "IMAGE_Vertex_LOSS_WEIGHT:",
            self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT,
        )
        print(
            "IMAGE_Edge_LOSS_WEIGHT:", self.cfg.TRAINER.PLKR.IMAGE_EDGE_LOSS_WEIGHT
        )

        print(
            "TEXT_Neighborhood_LOSS_WEIGHT:",
            self.cfg.TRAINER.PLKR.TEXT_NEIGHBORHOOD_LOSS_WEIGHT,
        )
        print(
            "IMAGE_Neighborhood_LOSS_WEIGHT:",
            self.cfg.TRAINER.PLKR.IMAGE_NEIGHBORHOOD_LOSS_WEIGHT,
        )

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PLKR.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            (
                loss_ce,
                normalized_text_features,
                zs_clip_text_embeddings,
                zs_image_embedd,
                image_ft,
            ) = model(image, label)

            if self.cfg.TRAINER.PLKR.MIXUP == True:
                zs_image_embedd_cuda = zs_image_embedd.cuda()
                zs_clip_text_embeddings_cuda = zs_clip_text_embeddings.cuda()

                image_ft_mixup, zs_image_embedd_mixup = efficient_mixup_double(
                    image_ft, zs_image_embedd_cuda, max_added_samples=image_ft.shape[0]
                )

                (
                    normalized_text_features_mixup,
                    zs_clip_text_embeddings_mixup,
                ) = efficient_mixup_double(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    max_added_samples=normalized_text_features.shape[0],
                )

            else:
                zs_image_embedd_cuda = zs_image_embedd.cuda()
                zs_clip_text_embeddings_cuda = zs_clip_text_embeddings.cuda()

            if self.cfg.TRAINER.PLKR.LOSS_TYPE == "Edge":

                loss_text_vertex_distance = F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    reduction="mean",
                )

                loss_text_edge_variation = compute_edge_L1_changes_and_mean(
                    normalized_text_features.float(),
                    zs_clip_text_embeddings_cuda.float(),
                )

                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                loss_image_edge_variation = compute_edge_L1_changes_and_mean(
                    image_ft.float(), zs_image_embedd_cuda.float()
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_text_edge_variation
                    * self.cfg.TRAINER.PLKR.TEXT_EDGE_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                    + loss_image_edge_variation
                    * self.cfg.TRAINER.PLKR.IMAGE_EDGE_LOSS_WEIGHT
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "ALL_Edge":
                loss_text_vertex_distance = F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    reduction="mean",
                )
                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                features_merged = torch.cat([image_ft, normalized_text_features], dim=0)

                features_merged_zs = torch.cat(
                    [zs_image_embedd_cuda, zs_clip_text_embeddings_cuda], dim=0
                )

                Edge_loss = compute_edge_L1_changes_and_mean(
                    features_merged.float(), features_merged_zs.float()
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                    + Edge_loss
                    * self.cfg.TRAINER.PLKR.IMAGE_NEIGHBORHOOD_LOSS_WEIGHT
                    / 100
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "Neighborhood":
                loss_text_vertex_distance = F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    reduction="mean",
                )
                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                image_neighborhood_loss = neighborhood_consistency_loss(
                    image_ft.float(), zs_image_embedd_cuda.float()
                )

                text_neighborhood_loss = neighborhood_consistency_loss(
                    normalized_text_features.float(),
                    zs_clip_text_embeddings_cuda.float(),
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + text_neighborhood_loss
                    * self.cfg.TRAINER.PLKR.TEXT_NEIGHBORHOOD_LOSS_WEIGHT
                    / 100
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                    + image_neighborhood_loss
                    * self.cfg.TRAINER.PLKR.IMAGE_NEIGHBORHOOD_LOSS_WEIGHT
                    / 100
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "ALL_Neighborhood":
                loss_text_vertex_distance = F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    reduction="mean",
                )
                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                features_merged = torch.cat([image_ft, normalized_text_features], dim=0)

                features_merged_zs = torch.cat(
                    [zs_image_embedd_cuda, zs_clip_text_embeddings_cuda], dim=0
                )

                neighborhood_loss = neighborhood_consistency_loss(
                    features_merged.float(),
                    features_merged_zs.float(),
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                    + neighborhood_loss
                    * self.cfg.TRAINER.PLKR.ALL_NEIGHBORHOOD_LOSS_WEIGHT
                    / 50
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "ICI_TCI":
                loss_text_vertex_distance = F.l1_loss(
                    normalized_text_features,
                    zs_clip_text_embeddings_cuda,
                    reduction="mean",
                )
                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "COS_ICI_TCI":
                loss_text_vertex_distance = cosine_similarity(
                    normalized_text_features, zs_clip_text_embeddings_cuda
                )
                loss_image_vertex_distance = cosine_similarity(
                    image_ft, zs_image_embedd_cuda
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "EUC_ICI_TCI":
                loss_text_vertex_distance = euclidean_distance(
                    normalized_text_features, zs_clip_text_embeddings_cuda
                )
                loss_image_vertex_distance = euclidean_distance(
                    image_ft, zs_image_embedd_cuda
                )

                L_Graph = (
                    loss_text_vertex_distance
                    * self.cfg.TRAINER.PLKR.TEXT_VERTEX_LOSS_WEIGHT
                    + loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                )

            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "ICI":
                loss_image_vertex_distance = F.l1_loss(
                    image_ft, zs_image_embedd_cuda, reduction="mean"
                )

                L_Graph = (
                    loss_image_vertex_distance
                    * self.cfg.TRAINER.PLKR.IMAGE_VERTEX_LOSS_WEIGHT
                )
            elif self.cfg.TRAINER.PLKR.LOSS_TYPE == "NO":
                L_Graph = 0

            loss = loss_ce + L_Graph

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            self.step_counter = self.step_counter + 1

        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):

        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):

        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = modified_dict[key] + dict1[key]
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )

            self._models[name].load_state_dict(state_dict, strict=False)
