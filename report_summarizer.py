import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import nltk
import csv
import argparse
import json
import random
import numpy as np

from torch.utils.data import DataLoader

from vllama.common.config import Config
from vllama.common.dist_utils import get_rank
from vllama.common.registry import registry
from vllama.datasets.datasets.ct_datasets import rectalMRIDataset
from transformers import StoppingCriteria, StoppingCriteriaList, LlamaForCausalLM, LlamaTokenizer

from vllama.models.vllamaita_frozen import vllamaItaFrozen 
from vllama.models.vllamaita import vllamaIta
#from torchmetrics.text.rouge import ROUGEScore
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)






txt_path = '/scratch/slurm-user3/changsun/data/rectal_MRI_label/202301_MRI_impression_final.json'
#txt_path = '/data/changsun/data/MRI/rectal/202301_MRI_impression_final.json'
#dataset = rectalMRIDataset(img_path, txt_path, None, False)
#test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = 'cuda'
tokenizer = LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B', 
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map={'': device})

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
#prompt = '<Img><ImageHere></Img> Could you describe the contents of this image for me?'

###DECODING STRATEGY
max_new_tokens=100
num_beams=1
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=1
temperature=1.0
max_length=200
stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

with open(txt_path, 'r') as rfile:
    report_data = json.load(rfile)


prompt = "From the following report that comes after the word REPORT:, make me a set of questions and answers of the pathological findings. REPORT: "
new_report = {}

for patient_id, report in report_data.items():
    
    input_text = prompt + report
    #input_image = vis_processor(image).to(device)
    #image_emb, atts_img, _ = model.encode_img(image)
    #input_emb, _ = model.prompt_wrap(image_emb, atts_img, prompt)
    
    #print('input_emb.shape', input_emb.shape)
    input_tokens = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=True)
    
    input_ids = input_tokens.input_ids
    input_ids = input_ids.to(device)
    input_length = input_ids.size(1)
    
    outputs = model.generate(
        input_ids = input_ids,
        max_new_tokens=max_new_tokens,
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

    output_text = tokenizer.decode(output_token[input_length:], add_special_tokens=False)
    
    print('output_text: ', output_text)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    
    #print()
    new_report[patient_id] = output_text
    
    print('==================================')
    print('GT Report: ', report)
    print('Candidate: ', output_text)
    print('==================================')
        
    with open('new_mri_report.json', 'w') as file:
        json.dump(new_report, file, indent=4)
    

    