# the CNN + positional encoding

from torchvision.models import resnet50, resnet101, vgg16, vgg16_bn, resnet18, resnet34
import torch
from torch import nn
import math
from transformers import RobertaModel, RobertaTokenizerFast, BertModel, BertTokenizerFast
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
from utils import NestedTensor
import torch.nn.functional as F


def build_backbone(args):
    #device = args.device
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.num_channels, args.hidden_dim, args.backbone_checkpoint, args.freeze_image_encoder, args.img_back_pretrained, args.backbone_type)
    for name, parameter in backbone.named_parameters():
        if not train_backbone:
            parameter.requires_grad_(False)
    input_proj = nn.Conv2d(backbone.num_channels, args.hidden_dim, kernel_size=1)
    cnn_model = Joiner(backbone, position_embedding, input_proj)
    cnn_model.num_channels = backbone.num_channels
    text_model = TextBackbone(text_encoder_type=args.text_encoder_type, freeze_text_encoder=args.freeze_text_encoder, d_model=args.hidden_dim, path_chansey=args.path_chansey, device=args.device)

    model = Concat(cnn_model, text_model)
    model.config = text_model.config

    return model


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Backbone(nn.Module):
    def __init__(self, num_channels=2048, hidden_dim=256, backbone_checkpoint='', freeze_image_encoder=False, pretrained=False, net_type='vgg'):
        super().__init__()
        self.num_channels = num_channels
        if net_type == 'vgg':
            self.backbone = vgg16(pretrained=pretrained)
            del self.backbone.classifier

        if 'resnet' in net_type:
            if net_type == 'resnet18':
                self.backbone = resnet18(pretrained=pretrained)
            elif net_type == 'resnet34':
                self.backbone = resnet34(pretrained=pretrained)
            elif net_type == 'resnet50':
                self.backbone = resnet50(pretrained=pretrained)
            elif net_type == 'resnet101':
                self.backbone = resnet101(pretrained=pretrained)
            else:
                print("ERROR")
            del self.backbone.fc
            
        #self.backbone = resnet50(pretrained=True)#, norm_layer=FrozenBatchNorm2d)
        #self.backbone = resnet101(pretrained=True)#, norm_layer=FrozenBatchNorm2d)
        #del self.backbone.fc
        #self.backbone = vgg16()
        #del self.backbone.classifier

        # create conversion layer
        if net_type == 'resnet50' or net_type == 'resnet101':
            self.conv = nn.Conv2d(2048, self.num_channels, 1)
        else:
            self.conv = nn.Conv2d(512, self.num_channels, 1)

        if backbone_checkpoint:
            checkpoint = torch.load(backbone_checkpoint, map_location=torch.device('cpu'))
            new_check = OrderedDict()
            for key, value in checkpoint["model_state_dict"].items():
                if 'backbone' in key:
                    new_check[key.replace("backbone.", "")] = value
            self.backbone.load_state_dict(new_check, strict=False)
            print("loaded checkpoint: ", backbone_checkpoint)

        # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #return_layers = {"layer4": "0"}
        return_layers = {"avgpool": "0"}
        self.body = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
        
        if freeze_image_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, inputs):
        """
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to hidden_dim feature planes for the transformer
        h = self.conv(x)

        return h
        """
        xs = self.body(inputs.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(inputs.mask[None].float(), size=x.shape[-2:]).bool()[0]
            x = self.conv(x)
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, input_proj):
        super().__init__(backbone, position_embedding, input_proj)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        src, mask = out[-1].decompose()
        src = self[2](src)
        return src, mask, pos


# Neu:
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TextBackbone(nn.Module):
    def __init__(self, text_encoder_type, freeze_text_encoder, d_model, path_chansey, device):
        super().__init__()
        if text_encoder_type == 'roberta-base':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(path_chansey + "Data/Pretrain_files/" + text_encoder_type, local_files_only=True)
            self.text_encoder = RobertaModel.from_pretrained(path_chansey + "Data/Pretrain_files/" + text_encoder_type, local_files_only=True)
        elif  'biobert' in text_encoder_type:
            self.tokenizer = BertTokenizerFast.from_pretrained('giacomomiolo/biobert_reupload')
            self.text_encoder = BertModel.from_pretrained('giacomomiolo/biobert_reupload')
        elif text_encoder_type == 'robbert':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')
            self.text_encoder = RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base')
        elif 'bertje' in text_encoder_type:
            self.tokenizer = BertTokenizerFast.from_pretrained('GroNLP/bert-base-dutch-cased')
            self.text_encoder = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')

        #self.tokenizer = BertTokenizerFast.from_pretrained(text_encoder_type)
        #self.text_encoder = BertModel.from_pretrained(text_encoder_type)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.config = self.text_encoder.config
        self.expander_dropout = 0.1
        self.resizer = FeatureResizer(
            input_feat_size=self.config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.device = torch.device(device)

    def forward(self, text):
        # Encode the text
        tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(self.device)
        encoded_text = self.text_encoder(**tokenized)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.resizer(text_memory)

        text_backbone_memory = {
            "text_attention_mask": text_attention_mask,
            "text_memory_resized": text_memory_resized,
            "tokenized": tokenized,
            "encoded_text": encoded_text
        }

        return text_backbone_memory


class Concat(nn.Sequential):
    def __init__(self, cnn,  text_model):
        super().__init__(cnn, text_model)
        self.config = text_model.config
        self.d_model = text_model.d_model


    def forward(self, image, text=None):
        src, mask, pos = self[0](image)
        pos_embed = pos[-1]

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        device = src.device

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        if text:
            text_backbone_memory = self[1](text)

            # Concat on the sequence dimension
            src = torch.cat([src, text_backbone_memory["text_memory_resized"]], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_backbone_memory["text_attention_mask"]], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_backbone_memory["text_memory_resized"])], dim=0)

            return src, mask, pos_embed, text_backbone_memory

        return src, mask, pos_embed, None

