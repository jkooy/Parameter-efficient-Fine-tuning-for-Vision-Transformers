from collections import OrderedDict
from typing import Callable, Tuple, Union
import logging

import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from timm.models.layers import trunc_normal_

from utils.comm import comm
from utils.comm import gather_tensors


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm2d, LayerNorm)):
            if comm.is_main_process():
                logging.info('=> init {} gamma to 1'.format(m))
                logging.info('=> init {} beta to 0'.format(m))
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from tranc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3)
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask)
                for _ in range(layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 pool_type: str = 'default',
                 skip_cls: bool = False):
        super().__init__()
        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        self.sequence_length = (input_resolution // patch_size) ** 2 + 1

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            if (not self.skip_cls):
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length, stride=self.sequence_length, groups=width)
            else:
                self.conv_pool = nn.Conv1d(width, width, self.sequence_length-1, stride=self.sequence_length, groups=width)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                self.sequence_length, width
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if (self.pool_type == 'average'):
            if self.skip_cls:
                x = x[:,1:,:]
            x = torch.mean(x,dim=1)
        elif (self.pool_type == 'linear'):
            if self.skip_cls:
                x = x[:,1:,:]
            x = x.permute(0,2,1)
            x = self.conv_pool(x)
            x = x.permute(0,2,1).squeeze()
        else:
            x = x[:, 0, :]

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_style: str = 'clip',
                 gather_tensors: bool = False,
                 eot_token: int = None,
                 tokenizer_style: str = 'clip',
                 pool_type: str = 'default',
                 skip_cls: bool = False
                 ):
        super().__init__()

        self.pool_type = pool_type
        self.skip_cls = skip_cls
        self.tokenizer_style = tokenizer_style
        self.eot_token = eot_token
        self.context_length = context_length
        self.transformer_style = transformer_style
        self.transformer_width = transformer_width
        self.vocab_size = vocab_size
        self.gather_tensors = gather_tensors

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                pool_type = self.pool_type,
                skip_cls=self.skip_cls
            )

        if self.transformer_style == 'clip':
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
            )
            trunc_normal_(self.positional_embedding, std=.02)
        elif self.transformer_style.startswith('hf_') or self.transformer_style.startswith('hfc_'):
            logging.info('=> Using HuggingFace model {}'.format(self.transformer_style))
            if (not self.transformer_style.startswith('hfc_')):
                self.transformer = AutoModel.from_pretrained(self.transformer_style[3:])
            else:
                self.transformer = AutoModel.from_pretrained(self.transformer_style[4:])
            transformer_width = self.transformer(torch.zeros((1,1)).type(torch.LongTensor))['last_hidden_state'].shape[2]
            self.transformer_width = transformer_width
        
        if self.transformer_style.startswith('hfc_'):
            self.transformer.init_weights()

        self.conv_pool = None
        if (self.pool_type == 'linear'):
            self.conv_pool = nn.Conv1d(self.transformer_width, self.transformer_width, self.context_length, stride=self.context_length, groups=self.transformer_width)

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            if comm.is_main_process():
                logging.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                if comm.is_main_process():
                    logging.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
            'logit_scale'
        }

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, norm=True):
        x = self.visual(image.type(self.dtype))

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
    # 'text' is not the raw text, it is the tokens.

        if self.transformer_style == 'clip':
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        elif self.transformer_style.startswith('hf_'):
            x = self.transformer(text)['last_hidden_state']

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if (self.pool_type == 'default'):
            if self.tokenizer_style == 'clip':
                x = x[
                    torch.arange(x.shape[0]),
                    text.argmax(dim=-1)
                ]
            elif (self.tokenizer_style.startswith('hf_') or self.tokenizer_style.startswith('hfc_')):
                x = x[
                    torch.arange(x.shape[0]),
                    (text == self.eot_token).nonzero(as_tuple=True)[0][0]
                ]
        elif (self.pool_type == 'linear'):
            x = x.permute(0,2,1)
            x = self.conv_pool(x)
            x = x.permute(0,2,1).squeeze()
        else:
            x = x[
                    torch.arange(x.shape[0]),
                    :
                ]
            x = torch.mean(x,dim=1)
            
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        if self.gather_tensors:
            features_image_all = gather_tensors(features_image)
            features_text_all = gather_tensors(features_text)
            logits_image_text = T * features_image_all @ features_text_all.t()
        else:
            logits_image_text = T * features_image @ features_text.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_image_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def get_cls_model(config, vocab_size = None, eot_token = None, **kwargs):
    embed_dim = config.MODEL.SPEC.EMBED_DIM

    image_resolution = config.TRAIN.IMAGE_SIZE[0]
    spec_vision = config.MODEL.SPEC.VISION
    vision_width = spec_vision.WIDTH

    if (spec_vision.MODEL == 'vit'):
        vision_layers = spec_vision.LAYERS
        vision_patch_size = spec_vision.PATCH_SIZE
    else:
        vision_layers = tuple(spec_vision.LAYERS)
        vision_patch_size = None

    spec_text = config.MODEL.SPEC.TEXT
    context_length = spec_text.CONTEXT_LENGTH

    # Allow information from the tokenizer class,
    # which is contained in the dataloader, to pass
    # into our model definition
    if (vocab_size == None):
        vocab_size = spec_text.VOCAB_SIZE
    else:
        vocab_size = vocab_size
    
    transformer_width = spec_text.WIDTH
    transformer_heads = spec_text.HEADS
    transformer_layers = spec_text.LAYERS
    transformer_style = spec_text.STYLE
    tokenizer_style = spec_text.TOKENIZER

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads,
        transformer_layers, transformer_style,
        gather_tensors=getattr(config.MODEL.SPEC, 'GATHER_TENSORS', False),
        eot_token=eot_token, tokenizer_style = tokenizer_style,
        pool_type=getattr(config.MODEL.SPEC, 'POOL_TYPE', 'default'),
        skip_cls=getattr(config.MODEL.SPEC, 'SKIP_CLS', False),
    )

    return model
