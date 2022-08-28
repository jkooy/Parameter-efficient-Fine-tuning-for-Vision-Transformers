"""
Linear classifier implemented with Pytorch Linear class
"""
import time
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .feature import FeatureData, get_model
from ..optim import build_optimizer
from ..evaluation.metric import get_metric

from ..evaluation.clip_vit import *
from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import clip_zeroshot_evaluator, construct_dataloader
from .feature import extract_text_features
import gc

import pdb


from functools import partial
from itertools import repeat

import logging
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

import collections.abc as container_abcs


MULTILABEL_DATASETS = {"voc-2007-classification","chestx-ray8"}




# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 res_score=False):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.res_score = res_score

    def forward(self, x, prev=None):
        B, N, C = x.shape
        qkv = self.qkv(x) \
                  .reshape(B, N, 3, self.num_heads, C // self.num_heads) \
                  .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale

        if prev is not None and self.res_score:
            attn_score = attn_score + prev

        if self.res_score:
            prev = attn_score

        attn = F.softmax(attn_score, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, prev


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pre_norm=True,
                 res_score=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, res_score=res_score
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        self.pre_norm = pre_norm
        self.res_score = res_score

    def forward(self, x, add_adapter=None, prev=None):
        if self.pre_norm:
            attn, prev = self.attn(self.norm1(x), prev)
            x = x + self.drop_path(attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            attn, prev = self.attn(x, prev)
            x = self.norm1(x + self.drop_path(attn))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        
        
        return x, prev


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_chans, img_size[0], img_size[1])
                )[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)





class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 use_cls_tocken=True,
                 norm_embed=False,
                 pre_norm=True,
                 res_score=False,
                 init='trunc_norm'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans,
                embed_dim=embed_dim
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim
            )

        self.norm_embed = norm_layer(embed_dim) if norm_embed else None
        num_patches = self.patch_embed.num_patches

        if use_cls_tocken:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches+1, embed_dim)
            )
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, pre_norm=pre_norm,
                res_score=res_score
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) if pre_norm else None

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    print(k, v.size(), model_dict[k].size())
                    if 'pos_embed' == k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new**2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.norm_embed:
            x = self.norm_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        prev = None
        for id, blk in enumerate(self.blocks):
            x, prev = blk(x, prev)

        if self.norm:
            x = self.norm(x)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


      
def get_cls_model(config, **kwargs):
    vit_spec = config.MODEL.SPEC
    if config.MODEL.CLIP_FP32:
        import clip_vlp as clip
    else:
        import clip
    if config.MODEL.NAME in clip.available_models(): 
        vit, _ = clip.load(config.MODEL.NAME, jit=False)
        vit.forward = vit.encode_image
        logging.info(f'Using CLIP pretrained model {config.MODEL.NAME}, input size {vit.visual.input_resolution}')
    else:
        vit = VisionTransformer(
            img_size=config.TRAIN.IMAGE_SIZE[0],
            patch_size=vit_spec.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=vit_spec.EMBED_DIM,
            qkv_bias=vit_spec.QKV_BIAS,
            depth=vit_spec.DEPTH,
            num_heads=vit_spec.NUM_HEADS,
            mlp_ratio=vit_spec.MLP_RATIO,
            drop_rate=vit_spec.DROP_RATE,
            attn_drop_rate=vit_spec.ATTN_DROP_RATE,
            drop_path_rate=vit_spec.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_cls_tocken=vit_spec.USE_CLS_TOKEN,
            norm_embed=getattr(vit_spec, 'NORM_EMBED', False),
            pre_norm=getattr(vit_spec, 'PRE_NORM', True),
            res_score=getattr(vit_spec, 'RES_SCORE', False),
            init=getattr(vit_spec, 'INIT', 'trunc_norm')
        )

        if config.MODEL.INIT_WEIGHTS:
            vit.init_weights(
                config.MODEL.PRETRAINED,
                config.MODEL.PRETRAINED_LAYERS,
                config.VERBOSE
            )

    return vit








class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda):
        super(Classifier, self).__init__()
        if config.MODEL.CLIP_FP32:
            import clip_vlp as clip
        else:
            import clip

        feature_type="image"
        model = get_model(config, feature_type=feature_type)
        if config.MODEL.NAME in clip.available_models():
            # self.backbone = get_cls_model(config)
            self.backbone = get_model(config, feature_type=feature_type)
            # self.backbone.float()
            if config.MODEL.NAME.startswith('vit_'):
                self.backbone.head = self.backbone.head_dist = None

            for name, param in self.backbone.named_parameters():
                if name.startswith('text') or name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final') \
                    or name.startswith('positional_embedding') or name.startswith('logit_scale'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                # if name.startswith('text') or name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                    param.requires_grad = False
                    # print(name, param.requires_grad)

                if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                    # freeze for supervised ViT under linear probing settings
                    if config.MODEL.NAME.startswith('vit_'):
                        param.requires_grad = False

                    if name.startswith('visual.conv1') or name.startswith('visual.ln_pre') or name.startswith('visual.transformer') or name.startswith('visual'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                        param.requires_grad = False

            input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
            self.optim = None
            self.l2_lambda = l2_lambda
            self.channel_bn = torch.nn.BatchNorm1d(
                input_dim,
                affine=False,
            )
            self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

            if config.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER:
                if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
                    tokenizer = SimpleTokenizer()
                elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
                    tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
                else:
                    tokenizer = None

                zeroshot_weights = extract_text_features(config, tokenizer, model=self.backbone, return_numpy=False)
                self.layers[0].weight.data = zeroshot_weights.T.to(self.layers[0].weight.dtype).to(self.layers[0].weight.device).contiguous()
                self.layers[0].bias.data.fill_(0.0)

            if config.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ and self.backbone.visual.proj is not None:
                encoder_proj = self.backbone.visual.proj
                head_proj = self.layers[0].weight.data
                head_bias = self.layers[0].bias.data
                self.backbone.visual.proj = None
                encoder_ic, encoder_oc = encoder_proj.shape
                self.channel_bn = torch.nn.BatchNorm1d(
                    encoder_ic,
                    affine=False,
                )
                self.layers = torch.nn.Sequential(torch.nn.Linear(encoder_ic, output_dim))
                self.layers[0].weight.data = head_proj @ encoder_proj.T.to(head_proj.dtype).to(head_proj.device)
                self.layers[0].bias.data = head_bias

            if config.TRAIN.INIT_HEAD_WITH_LOGIT_SCALE:
                self.layers[0].weight.data = self.layers[0].weight.data * self.backbone.logit_scale.exp().to(self.layers[0].weight.dtype).to(self.layers[0].weight.device)

            self.normalize_visual_output = config.TRAIN.NORMALIZE_VISUAL_FEATURE
        else:
            self.backbone = get_cls_model(config)
            logging.info(f'{self.backbone}, backbone')
            logging.info(f'{model}, backbone2')

            if os.path.exists("vit.pth"):
                backbone = torch.load('vit.pth')
            else:
                torch.save(model.state_dict(), "vit.pth") 
                backbone = torch.load('vit.pth')
            tprobing_model_dict = self.backbone.state_dict()
            state_dict = {k:v for k,v in backbone.items() if k in tprobing_model_dict.keys()}
            tprobing_model_dict.update(state_dict)
            self.backbone.load_state_dict(tprobing_model_dict)
            logging.info(f'{model}, model')
            logging.info(f'{self.backbone}, model1')
            # for param in model.parameters():
            #     logging.info(f'{torch.sum(param)}, model1')
            # for params in self.backbone.parameters():
            #     logging.info(f'{torch.sum(params)}, model2')

            for param in self.backbone.parameters():
                param.requires_grad = False


            input_dim, output_dim = config.MODEL.SPEC.VISION.EMBED_DIM, config.DATASET.NUM_CLASSES
            self.optim = None
            self.l2_lambda = l2_lambda
            self.channel_bn = torch.nn.BatchNorm1d(
                input_dim,
                affine=False,
            )
            self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

    def forward(self, img):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)
        
        if self.normalize_visual_output:
            outputs = F.normalize(outputs)
        # pdb.set_trace()
        outputs = self.layers(outputs)
        return outputs


def hyperparameter_sweep(train_dataloader, val_dataloader, config):
    logging.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=97).tolist()
    # l2_lambda_list = np.logspace(-3, 3, num=97).tolist()
    l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=7))]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]

        # best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        try:
            best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        except:
            best_score_ = 0.0
            continue       

        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            config.TRAIN.WD = l2_lambda_list[left]
            
            # best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            try:
                best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            except:
                best_score_ = 0.0
                continue

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time() - start))
    return l2_lambda_list[peak_idx], peak_score


def train_task(train_dataloader, test_dataloader, config, sweep_run=False):
    best_acc1 = 0

    model = Classifier(config, 0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable params: {pytorch_total_params / 1000000}M.')


    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    if config.DATASET.DATASET in MULTILABEL_DATASETS:
        criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        if not config.TRAIN.EMULATE_ZERO_SHOT:
            train_one(train_dataloader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1 = validate(test_dataloader, model, criterion, epoch, config)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

    logging.info(f'=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}')

    # TODO: check this again, force gc to address oom issue.
    del model, criterion, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    if sweep_run and config.TRAIN.SEARCH_RESULT_ON_LAST_EPOCH:
        return acc1
    return best_acc1



def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    end = time.time()
    for _,  batch in enumerate(train_loader):

        images, target = batch[:2]

        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)

        if images.shape[0] == 1: continue # TODO: check this fix on batch left is size-1
        if target.shape[-1] == 1: target = target[:,0]
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)

        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100. * metric(labels, logits)
    except:
        metric_result = 0.
    logging.info(f'[Epoch {epoch}] Train: {metric_name} {metric_result:.3f}')


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, config):
    batch_time = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.eval()
    end = time.time()
    for batch in val_loader:
        images, target = batch[:2]

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)
        target = target.cuda(config.GPUS[0], non_blocking=True)
        if target.shape[-1] == 1: target = target[:,0]

        # compute output
        output = model(images)
        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100. * metric(labels, logits)
    except:
        metric_result = 0.
    logging.info(f'[Epoch {epoch}] Val: {metric_name} {metric_result:.3f}')

    return metric_result


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.TRAIN.LR
    for milestone in config.TRAIN.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def hyperparameter_sweep_lr(train_dataloader, val_dataloader, config):
    logging.info("=> Start hyperparameter tuning.")
    start = time.time()
    learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = hyperparameter_sweep(train_dataloader, val_dataloader, config)
        logging.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logging.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time() - start))
    logging.info('=> Finished hyperparameter tuning.')
    return best_lr, best_l2_lambda


def merge_trainval_loader(train_loader, val_loader):
    # TODO: DataLoader from feature.py get_dataloader()
    trainset, valset = train_loader.dataset, val_loader.dataset
    fullset = trainset.dataset
    assert trainset.dataset is valset.dataset
    assert len(fullset) == len(trainset) + len(valset)

    trainval_loader = torch.utils.data.DataLoader(
        fullset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        sampler=None,
        drop_last=False,
    )
    return trainval_loader


def seed_debug(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config):
    # best_lr = lr
    # best_l2_lambda = l2
    
    best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_dataloader, val_dataloader, config)

    logging.info("=> The final classifier is on training ...")
    logging.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()

    if config.DATASET.DATASET == 'patch-camelyon' and config.DATASET.NUM_SAMPLES_PER_CLASS == 10000:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logging.info(f'Used the subset to train the model, regenerating the full set for final run.')
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = -1
        config.freeze()
        logging.info(f'Old: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.')
        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)
        logging.info(f'Generated: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.')

    if config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN:
        trainval_dataloader = merge_trainval_loader(train_dataloader, val_dataloader)
        logging.info(f'Using the full trainval set to train final model. len(dataset)={len(trainval_dataloader.dataset)}')
    else:
        trainval_dataloader = train_dataloader
        logging.info(f'Using the train set only to train final model. len(dataset)={len(trainval_dataloader.dataset)}')
    train_task(trainval_dataloader, test_dataloader, config)


