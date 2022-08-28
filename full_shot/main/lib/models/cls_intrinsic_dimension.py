from functools import partial
from itertools import repeat
from torch._six import container_abcs

import logging
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_


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



class FastFoodWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, device=0):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastFood transform
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(FastFoodWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastfood_params = {}

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        V = nn.Parameter(torch.zeros((intrinsic_dimension)).to(device))
        self.register_parameter("V", V)
        v_size = (intrinsic_dimension,)

        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:

                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(device)
                )

                # Generate fastfood parameters
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(DD, device)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

        for name, base, localname in self.name_base_localname:
            delattr(base, localname)

    def forward(self, x):
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)

            # Fastfood transform te replace dence P
            ray = fastfood_torched(self.V, DD, self.fastfood_params[name]).view(
                init_shape
            )

            param = self.initial_value[name] + ray

            setattr(base, localname, param)

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x


def fast_walsh_hadamard_torched(x, axis=0, normalize=False):
    """
    Performs fast Walsh Hadamard transform
    :param x:
    :param axis:
    :param normalize:
    :return:
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), (
        "For a vector of shape %s, axis must be in [0, %d] but it is %d"
        % (orig_shape, len(orig_shape) - 1, axis)
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(np.prod(orig_shape[axis + 1 :]))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor).to(device)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL)).to(device)
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_().to(device)
    GG.requires_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return [BB, Pi, GG, divisor, LL]


def fastfood_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    if not param_list:

        BB, Pi, GG, divisor, LL = fastfood_vars(DD, device=device)

    else:

        BB, Pi, GG, divisor, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(BB, dd_pad)
    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, GG)

    # (HGPiHBX)
    mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)

    ret = torch.div(mul_5[:DD], divisor * np.sqrt(float(DD) / LL))

    return ret



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



# Single Adapter
class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="relu",
        init_bert_weights=True,
        add_layer_norm_before=True,
        add_layer_norm_after=False,
        residual_before_ln=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x):  # , residual_input=None):
        residual_input = x
        down = self.adapter_down(x)
        up = self.adapter_up(down)

        output = up

        # apply residual connection before layer norm if configured in this way
        if self.residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

            
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
        return x


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
                 res_score=False,
                 dintrinsic = 100):
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

        #add adapter
        self.adapter = Adapter(dim,
                                down_sample=64,
                                non_linearity="relu",
                                init_bert_weights=True,
                                add_layer_norm_before=True,
                                add_layer_norm_after=False,
                                residual_before_ln=True,
                                )

        logging.info(f"intrinsic dimension {dintrinsic}.")
        self.intrinsic_attn = FastFoodWrap(Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, res_score=res_score
        ), intrinsic_dimension=dintrinsic)
        self.intrinsic_mlp = FastFoodWrap(Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        ), intrinsic_dimension=dintrinsic)
        self.intrinsic_adapter = FastFoodWrap(Adapter(dim,
                                down_sample=64,
                                non_linearity="relu",
                                init_bert_weights=True,
                                add_layer_norm_before=True,
                                add_layer_norm_after=False,
                                residual_before_ln=True,
                                ), intrinsic_dimension=dintrinsic)

    def forward(self, x, prev = None, measure_idim = None):
        if measure_idim == "attention":
            if self.pre_norm:
                attn = self.intrinsic_attn(self.norm1(x))
                x = x + self.drop_path(attn)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                attn = self.intrinsic_attn(x)
                x = self.norm1(x + self.drop_path(attn))
                x = self.norm2(x + self.drop_path(self.mlp(x)))
        elif measure_idim == "adapter":
            if self.pre_norm:
                attn = self.attn(self.norm1(x))
                x = x + self.drop_path(attn)
                x = x + self.intrinsic_adapter(self.drop_path(self.mlp(self.norm2(x))))[0]
            else:
                attn = self.attn(x)
                x = self.norm1(x + self.drop_path(attn))
                x = self.norm2(x + self.intrinsic_adapter(self.drop_path(self.mlp(x))))[0]
        elif measure_idim == "mlp":
            if self.pre_norm:
                attn = self.attn(self.norm1(x))
                x = x + self.drop_path(attn)
                x = x + self.drop_path(self.intrinsic_mlp(self.norm2(x)))
            else:
                attn = self.attn(x)
                x = self.norm1(x + self.drop_path(attn))
                x = self.norm2(x + self.drop_path(self.intrinsic_mlp(x)))
        else:
            if self.pre_norm:
                attn = self.attn(self.norm1(x), prev)
                x = x + self.drop_path(attn)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                attn = self.attn(x, prev)
                x = self.norm1(x + self.drop_path(attn))
                x = self.norm2(x + self.drop_path(self.mlp(x)))

        return x


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
                 init='trunc_norm',
                 dintrinsic = 100,
                 layerType = "mlp",
                 layernum = 100):
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
                res_score=res_score,
                dintrinsic = dintrinsic
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

        self.layerType = layerType
        self.layernum = layernum

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
        layerType = self.layerType

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
            # if id ==11:
            # if id ==0:
            if id ==self.layernum:
                x = blk(x, prev, layerType)
            else:
                x = blk(x, prev, None)

            # if id ==0:
            #     # x = blk(x, prev, "adapter")
            #     # x = blk(x, prev, "attention")
            #     x = blk(x, prev, "mlp")
            # else:
            #     x = blk(x, prev, None)

        if self.norm:
            x = self.norm(x)

        if self.cls_token is not None:
            x = x[:, 0]
        else:
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def get_cls_model(config, dintrinsic, layerType, layernum, **kwargs):
    vit_spec = config.MODEL.SPEC
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
        init=getattr(vit_spec, 'INIT', 'trunc_norm'),
        dintrinsic = dintrinsic,
        layerType = layerType,
        layernum = layernum
    )

    if config.MODEL.INIT_WEIGHTS:
        vit.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE
        )

    return vit
