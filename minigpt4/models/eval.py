import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from minigpt4.models.mini_gpt4_ita_frozen import MiniGPT4ItaFrozen 
from minigpt4.models.mini_gpt4_ita import MiniGPT4Ita


from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

pretrained_checkpoint = 'checkpoint_10.pth'
model = MiniGPT4Ita

model.load_state_dict(torch.load(pretrained_checkpoint))


