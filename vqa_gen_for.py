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






txt_path = '/scratch/slurm-user3/changsun/vllama/formatted_rectal_mri_report_real.json'#'/scratch/slurm-user3/changsun/data/rectal_MRI_label/202301_MRI_impression_final.json'
#txt_path = '/data/changsun/data/MRI/rectal/202301_MRI_impression_final.json'
#dataset = rectalMRIDataset(img_path, txt_path, None, False)
#test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = 'cuda:0'
tokenizer = LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B', load_in_8bit=True, torch_dtype=torch.float16, device_map='auto')

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
max_new_tokens=300
num_beams=5
min_length=1
top_p=0.9
repetition_penalty=1.2
length_penalty=1
temperature=1.0
max_length=200
stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device), torch.tensor([660, 29901]).to(device), torch.tensor([3319, 29901]).to(device)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

with open(txt_path, 'r') as rfile:
    report_data = json.load(rfile)


instruction = "Like the above examples, make me a list of questions preceded by the word Question: and the corresponding list of answers preceded by the word Answer: regarding the following report. REPORT: "
fs_prompt = ["REPORT: The distance of the lowest tumor margin from the anal verge is approximately 8 cm, and the tumor is partially peritonealized. It is partially encircling in its circumferential location, measuring 7 cm longitudinally. In T-staging, it is classified as CT3/CT4a. Tethering of the peritoneal reflection indicates circumferential resection margin involvement. The anal sphincter is unaffected. Mesorectal lymph nodes, larger than 8 mm with irregular borders, are present. Extramamesorectal lymph nodes are also present, and there is extramural venous invasion.",
"Questions: 'What is the distance of the lowest tumor margin from the anal verge?', 'What is the size of the tumor in its circumferential location?', 'How is the tumor classified in T-staging?', 'What does tethering of the peritoneal reflection indicate?', 'Is the anal sphincter affected?', 'What can you tell me about the lymph nodes in the paragraph?', 'Is there any mention of extramural venous invasion?'",
"Answers: 'The distance of the lowest tumor margin from the anal verge is approximately 8 cm.', 'The tumor in its circumferential location measures 7 cm longitudinally.', 'In T-staging, it is classified as CT3/CT4a.', 'Tethering of the peritoneal reflection indicates circumferential resection margin involvement.', 'The anal sphincter is unaffected.', 'Mesorectal lymph nodes, larger than 8 mm with irregular borders, are present. Extramamesorectal lymph nodes are also present.', 'Yes, there is extramural venous invasion.'",
"REPORT: The distance from the rectal cancer to the anal verge measures approximately 3.4 cm. The tumor does not relate to the peritoneum. It is partially encircling in its circumferential location and measures about 1.3 cm longitudinally. It is classified as CT2 in T-staging. The circumferential resection margin is negative, and the anal sphincter is suspicious for involvement (internal sphincter involvement) due to its close proximity to the tumor. Mesorectal lymph nodes are absent, as are extramesorectal lymph nodes and extramural venous invasion (EMVI).",
"Questions: 'What is the distance of the lowest tumor margin from the anal verge?', 'Is there any relationship between the tumor and the peritoneum?', 'What is the size of the tumor in its circumferential location?', 'How is the tumor classified in T-staging?', 'Is there any signs of circumferential resection margin?', 'Is the anal sphincter affected?', 'What can you tell me about the lymph nodes in the paragraph?', 'Is there any mention of extramural venous invasion?'",
"Answers: 'The distance of the lowest tumor margin from the anal verge is approximately 3.4 cm.', 'The tumor has no relationship with the peritoneum.', 'The tumor is partially encircling in its circumferential location and measures 1.3 cm longitudinally.', 'The tumor is classified as CT2 in T-staging, 'There is no involvement of the circumferential resection margin.', 'The anal sphincter is suspicious for (internal sphincter involvement) due to its close proximity to the tumor.', 'Mesorectal lymph nodes and extramesorectal lymph nodes are absent.', 'There is no extramural venous invasion.'",
"REPORT: The lowest tumor margin from the anal verge is 6.5 cm, and it has a partial relationship with the peritoneum. It is partially encircling in its circumferential location and measures 5.4 cm longitudinally. The tumor is classified as T2 on the basis of increased size. There is no involvement of the circumferential resection margin, the anal sphincter remains unaffected, and mesorectal lymph nodes, although present, are decreased in size. Extramesorectal lymph nodes and extramural venous invasion are absent.",
"Questions: 'What is the distance of the lowest tumor margin from the anal verge?', 'Is there any relationship between the tumor and the peritoneum?', 'How is the tumor described in terms of circumferential location and size?', 'What is the tumor classified as in T-staging?', 'Is there any involvement of the circumferential resection margin?', 'Is the anal sphincter affected by the tumor?', 'What is the status of mesorectal lymph nodes?', 'Are extramesorectal lymph nodes present?', 'Is there any extramural venous invasion?'",
"Answers: 'The distance of the lowest tumor margin from the anal verge is 6.5 cm.', 'The tumor has a partial relationship with the peritoneum.', 'The tumor is partially encircling in its circumferential location and measures 5.4 cm longitudinally.', 'The tumor is classified as T2 due to increased size.', 'There is no involvement of the circumferential resection margin.', 'The anal sphincter remains unaffected by the tumor.', 'Mesorectal lymph nodes are present but decreased in size.', 'Extramesorectal lymph nodes are absent.', 'There is no extramural venous invasion.'",
"REPORT: The lowest tumor margin is situated approximately 9 cm from the anal verge, and there is no relationship between the tumor and the peritoneum. The tumor is partially encircling (3-6 o'clock) in its circumferential location and measures about 1.3 cm longitudinally. It is classified as CT2 in T-staging, indicating a thickness less than 5mm. There is no involvement of the circumferential resection margin and the anal sphincter remains unaffected. Mesorectal lymph nodes, larger than 8mm with irregular borders, are absent, as are extramesorectal lymph nodes, and extramural venous invasion (EMVI) is also absent. ",
"Questions: 'What is the distance of the lowest tumor margin from the anal verge?', 'Is there any relationship between the tumor and the peritoneum?', 'How is the tumor described in terms of circumferential location and size?', 'What is the tumor classified as in T-staging?', 'Is there any involvement of the circumferential resection margin?', 'Is the anal sphincter affected by the tumor?', 'What is the status of mesorectal lymph nodes?', 'Are extramesorectal lymph nodes present?', 'Is there any extramural venous invasion?'",
"Answers: 'The distance of the lowest tumor margin from the anal verge is 9 cm.', 'The tumor has no relationship with the peritoneum.', 'The tumor is partially encircling (3-6 o'clock) in its circumferential location and measures 1.3 cm longitudinally.', 'The tumor is classified as CT2 in T-staging, indicating a thickness less than 5mm.', 'There is no involvement of the circumferential resection margin.', 'The anal sphincter remains unaffected by the tumor.', 'Mesorectal lymph nodes larger than 8mm with irregular borders, are absent.', 'Extramesorectal lymph nodes are absent.', 'There is no extramural venous invasion.'",
"REPORT: Distance of the lowest tumor margin from the anal verge is about 8.1 cm. The tumor is not related to the peritoneum. It is partially encircling in its circumferential location, measuring about 4.1 cm longitudinally. T-staging is CT3 (<5mm) > CT2. The circumferential resection margin is involved, in a 12 o'clock orientation, indicating a positive margin. The anal sphincter is unaffected. There are mesorectal lymph nodes present, and extramesorectal lymph nodes, as well as extramural venous invasion (EMVI) are also present.",
"Questions: 'What is the distance of the lowest tumor margin from the anal verge?', 'Is there any relationship between the tumor and the peritoneum?', 'What is the size of the tumor in its circumferential location?', 'How is the tumor classified in T-staging?', 'Is there any signs of circumferential resection margin?', 'Is the anal sphincter affected?', 'What can you tell me about the lymph nodes in the paragraph?', 'Is there any mention of extramural venous invasion?'",
"Answers: 'The distance of the lowest tumor margin from the anal verge is 8.1 cm.', 'The tumor has no relationship with the peritoneum.', 'The tumor is partially encircling in its circumferential location and measures 4.1 cm longitudinally.', 'The tumor is classified as CT3.', 'The circumferential resection margin is positive, in a 12 o'clock orientation.', 'The anal sphincter remains unaffected by the tumor.', 'Mesorectal lymph nodes and extramesorectal lymph nodes are both present.', 'Extramural venous invasion (EMVI) is present.'"
]


fs_prompt_str = " ".join(fs_prompt)

new_report = {}

for patient_id, report in report_data.items():
    
    input_text = fs_prompt_str + instruction + report
    #input_image = vis_processor(image).to(device)
    #image_emb, atts_img, _ = model.encode_img(image)
    #input_emb, _ = model.prompt_wrap(image_emb, atts_img, prompt)
    #print('input_text: ', input_text)
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
    
    #print('output_text: ', output_text)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    
    #print()
    new_report[patient_id] = output_text
    
    print('==================================')
    print('Candidate: ', output_text)
    print('GT Report: ', report)
    print('==================================')
    
    
with open('formatted_mri_report_real_vqa.json', 'w') as file:
    json.dump(new_report, file, indent=4)
    

    