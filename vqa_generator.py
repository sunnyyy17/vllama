import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
#from rouge import Rouge
#from rouge import Rouge 

from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader

from vllama.common.config import Config
from vllama.common.dist_utils import get_rank
from vllama.common.registry import registry
from vllama.datasets.datasets.ct_datasets import brainMRIDataset

prompt = '<Img><ImageHere></Img> Could you describe the contents of this image for me?'

max_new_tokens=300
num_beams=1
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=1
temperature=1.0
max_length=2000

tokenizer = LlamaTokenizer.from_pretrained('/data/changsun/models/PMC-LLaMA')
model = LlamaForCausalLM.from_pretrained('/data/changsun/models/PMC-LLaMA')

question_list = [
    
]
with torch.no_grad()
with torch.no_grad():
    for idx, item in enumerate(tqdm(test_dataloader)):
        
        image = item[0]
        text = item[1]
        image = item[0]
        
        #input_image = vis_processor(image).to(device)
        image_emb, atts_img, _ = model.encode_img(image)
        input_emb, _ = model.prompt_wrap(image_emb, atts_img, prompt)
        
        #print('input_emb.shape', input_emb.shape)
        outputs = model.llama_model.generate(
            inputs_embeds=input_emb, 
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        
        
        print('==================================')
        print('Candidate: ', output_text)
        print('Ground Truth: ', text)
        print('==================================')

    
    

    