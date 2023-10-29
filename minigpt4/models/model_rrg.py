import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry

from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer


@registry.register_model("model_rrg")
class RRGModel(nn.Module):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/modelrrg.yaml",
    }
    
    def __init__(
        self,
        vit_model='ViT-B/32',
        llama_model="",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        vit_path = '/scratch/slurm-user3/changsun/data/ct-for-detect-seg/checkpoint_15000.pt',
    ): 
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        
        print('Loading VIT')
        
        ###FOR CheXZero
        self.visual_encoder = self.init_CheXzero_encoder(vit_path)
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')
        
        
        print('Loading LLAMA')
        
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        if self.low_resource:
            print("Low_resource activated")
            
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
        ###Parameter set to True
        '''
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        '''
        print('Loading LLAMA Done')
        
        self.llama_proj_chz = nn.Linear(768, self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
    
    def forward(self, samples):
        samples 
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "ViT-B/32") #"eva_clip_g"
        llama_model = cfg.get("llama_model")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", True)
        device_8bit = cfg.get("device_8bit", 0)
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            llama_model=llama_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
        
        return model
