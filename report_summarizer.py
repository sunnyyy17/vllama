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
num_beams=10
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=1
temperature=1.0
max_length=200
stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device), torch.tensor([660, 29901]).to(device), torch.tensor([3319, 29901]).to(device)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

'''
input_tokens = tokenizer.decode(
        stop_words_ids[0],
        add_special_tokens=False)

print(input_tokens)

input_tokens = tokenizer.decode(
        stop_words_ids[1],
        add_special_tokens=False)

print(input_tokens)
'''
with open(txt_path, 'r') as rfile:
    report_data = json.load(rfile)


prompt = "Every clinical report that comes after Q: is summarized as shown after A: . Summarize the last clinical report Q based on the above examples: "

fs_prompt = ["Q: * f/u since 2021-10-20 rectal mri 1> distance of the lowest tumor margin from the anal verge : about 7 cm 2> tumor relationship to the peritoneum ; none 3> circumferential tumor location ; partially encircling 4> longitudinal tumor size: about 1.4cm 5> t-staging ; ct3 ( > 5mm), 6> circumferential resection margin involvement; ; negative 7> anal sphincter involvement. ; absent 8> mesorectal lymph node (> 8mm, irregular border) ; present, decreased in size 9> extramesorectal lymph node ; none 10> extramural venous invasion (emvi) ; absent.",
"A: The lowest tumor margin is situated approximately 7 cm from the anal verge, and there is no relationship between the tumor and the peritoneum. The tumor is partially encircling in its circumferential location and measures about 1.4 cm longitudinally. It is classified as CT3 in T-staging, indicating a thickness greater than 5mm. There is no involvement of the circumferential resection margin and the anal sphincter remains unaffected. Mesorectal lymph nodes, larger than 8mm with irregular borders, are present but have decreased in size. There are no extramesorectal lymph nodes, and extramural venous invasion (EMVI) is also absent.",
"Q: - known rectal cancer, middle rectum without perirectal fat infiltration(ct3). : interval decrease in size - a few tiny perirectal and sigmoid mesenteric lymph nodes. ; r/o reactive hyperplasia rather than metastasis. : no significant interval change. - no abnormal fluid collection in significant amount, pelvis. - enlarged prostate.",
"A: The patient, known to have rectal cancer located in the middle rectum, shows no perirectal fat infiltration, classifying it as CT3. There has been an interval decrease in the size of the tumor. A few tiny perirectal and sigmoid mesenteric lymph nodes are noted; however, these are more suggestive of reactive hyperplasia rather than metastasis, as there has been no significant interval change. There is no abnormal fluid collection in any significant amount within the pelvis. Additionally, the patient has an enlarged prostate.",
"Q:"]

'''
a 3.4 cm size mass with calcification between left posterior aspet of the bladder and rectum.",
"A: The imaging reveals a mass measuring 3.4 cm with calcification situated between the left posterior aspect of the bladder and the rectum.",
"Q: f/u for mucinous adenocarcinoma ; decrease in size of heterogeneous enhancing mass (9 x 5cm on coronal image) around the both levator ani muscles, right perirectal and both ischioanal fossa, extending to subcutaneous layer of both posterior perineum - about 2.4cm sized lesion in right side seminal vesicle (mark) ; t2- low si ; d/dx. fibrosis vs. metastasis.",
"A: Follow-up for mucinous adenocarcinoma shows a decrease in the size of a heterogeneous enhancing mass, previously measuring 9 x 5 cm, located around both levator ani muscles, extending into the right perirectal and both ischioanal fossa, and reaching the subcutaneous layer of both posterior perineum. There is also a 2.4 cm sized lesion in the right seminal vesicle. The mass exhibits low signal intensity on T2-weighted images. Differential diagnosis includes fibrosis versus metastasis.",
"Q: 5x5x 10cm sized irregular bulky lobulated mucinous mass, perirectal, ischiorectal, perianal area. ; involving the bilateral levator ani muscles -> probably known mucinous adenocarcinoma. no significant lymphadenopathy, abdomen limited evaluation of rectum due to collapsed state.",
"A: There is a bulky, irregular, lobulated mucinous mass measuring 5x5x10 cm in the perirectal, ischiorectal, and perianal area. It involves the bilateral levator ani muscles and is likely a known mucinous adenocarcinoma. No significant lymphadenopathy is noted, and there is a limited evaluation of the rectum due to its collapsed state.",
"Q: * incomplete study due to limited mr scan range * compared to 2020-8-25 mri - decrease in size of previous noted t2 hyperintensity tubular lesion, posterior part of anus ; 2.4 cm -> 1cm > d/dx mucinous adeno carcinoma (r/o t3 cannot be excluded) vs. benign cystic lesion with significant fibrosis () - no evidence of mesorectal lymph nodes - no evidence of extra mesorectal lymph nodes - multiple variable sized myomas, uterus.",
"A: Compared to previous MRI, there is a decrease in the size of the previously noted T2 hyperintense tubular lesion in the posterior part of the anus, measuring now 1 cm from an initial 2.4 cm. Differential diagnosis includes mucinous adenocarcinoma versus a benign cystic lesion with significant fibrosis. There is no evidence of mesorectal or extra mesorectal lymph nodes, and multiple variable-sized myomas are noted in the uterus.",
"Q: nonspecific rectum. - no visible specific focal lesion on this mr. - no evidence of any abnormal lymphadenopathy in significant size, abdomen. - no abnormal fluid collection in significant amount, abdomen.",
"A: The MR imaging of the rectum is nonspecific with no visible specific focal lesions. There is no evidence of any abnormal lymphadenopathy of significant size in the abdomen, nor any abnormal fluid collection of significant amount.",
"Q: 1. distance of the lowest tumor margin from the anal verge: about 10 cm 2. tumor relationship to the peritoneum ; none 3. circumferential tumor location ; partially encircling 4. longitudinal tumor size: about 2cm 5. t-staging: ct2 6. circumferential resection margin involvement; ; negative 7. anal sphincter involvement ; absent 8. mesorectal lymph node (> 8mm, irregular border) ; absent 9. extramesorectal lymph node ; none 10. extramural venous invasion (emvi) ; absent.",
"A: The imaging findings include a tumor margin located approximately 10 cm from the anal verge, with the tumor partially encircling the circumference and measuring about 2 cm longitudinally. The tumor is staged as CT2 with no involvement of the circumferential resection margin or anal sphincter. Mesorectal and extramesorectal lymph nodes are absent, as is extramural venous invasion.",
"Q: * clinical inforamation : anastomotic stricture on sigmoidoscopy * fu since 2016-12-05 ct - s/p ar for rectosigmoid colon cancer (2016-1-26) - abrupt luminal narrowing at the anastomosis site ; stricture, suggested - no evidence of gross recurrence or metastasis - blind pouch at 11 o'clock direction of just above anastmotic site, rectum. ; probable, inflammatory tract. -> n.c. - soft tissue density at presacral area around bowel anastomotic site and fatty strand along the sigmoid mesocolon. ; probable postop change. -> n.c. - no significant lymphadenopathy in the scanned abdomen. - no change of small cystic lesion, lt. adnexa. ; r/o benign cystic tumor -> n.c.",
"A: Imaging following sigmoidoscopy for anastomotic stricture shows abrupt luminal narrowing at the anastomosis site suggestive of a stricture, but no evidence of gross recurrence or metastasis. A blind pouch is noted at the 11 o'clock direction just above the anastomotic site in the rectum, likely an inflammatory tract. Soft tissue density is present at the presacral area around the bowel anastomotic site, with fatty strands along the sigmoid mesocolon, likely postoperative changes. No significant lymphadenopathy is observed in the scanned abdomen, and there is no change in the small cystic lesion in the left adnexa.",
"Q: con) grossly decrease in size of mass since 2019-11-11 1. distance of the lowest tumor margin from the anal verge: about 5 cm 2. tumor relationship to the peritoneum ; partial peritonealization 3. circumferential tumor location ; completely encircling the lumen 4. longitudinal tumor size: about 11 cm 5. t-staging: ct4a (r/o t4b : r/o seminal vesicle) 6. circumferential resection margin involvement ; positive 7. anal sphincter involvement. ; absent 8. mesorectal lymph node (> 8mm, irregular border); present 9. extramesorectal lymph node ; present (right internal iliac, rt. external iliac) 10. extramural venous invasion (emvi) ; present.",
"A: The lowest tumor margin is approximately 5 cm from the anal verge. The tumor partially peritonealizes and completely encircles the lumen, measuring about 11 cm longitudinally. It is staged as CT4a, with a positive circumferential resection margin. The anal sphincter is not involved. Mesorectal lymph node involvement is present with nodes greater than 8mm and irregular borders, as well as the presence of extramesorectal lymph nodes in the right internal and external iliac regions. Extramural venous invasion is also present.",
"Q: 1. a 4.8 cm extent irregular wall thickening and heterogeneous enhancement of sigmoid colon with r/o pericolic infiltration. -> probable colonic malignancy (t3). 2. several borderline enlarged enhancing lns in superior rectal chain, r/o metastatic lns (n2). 3. no ascites.",
"A: Imaging shows a 4.8 cm extent of irregular wall thickening and heterogeneous enhancement of the sigmoid colon, suggestive of probable colonic malignancy staged as T3. Several borderline enlarged enhancing lymph nodes are noted in the superior rectal chain, raising the possibility of metastatic lymph nodes (N2). No ascites are observed.",
"Q: s/p ccrt for rectal cancer -interval decreased extent of enhancing wall thickening and perirectal fat infiltration at distal rectum (4.6cm-> 2.1cm), extending to anorectal junction -no evidence of crm involvement or emvi -tiny lns at perirectal and right internal iliac area ; proable benign.",
"A: Post chemoradiotherapy for rectal cancer, there is a decreased extent of enhancing wall thickening and perirectal fat infiltration at the distal rectum, now measuring 2.1 cm from an initial 4.6 cm, extending to the anorectal junction. There is no evidence of CRM involvement or EMVI. Tiny lymph nodes are noted at the perirectal and right internal iliac area, likely benign.",
"Q: * f/u since 2019-09-23 ct * c.i: left hepatectomy d/t ihd stone 1. distance of the lowest tumor margin from the anal verge : about 6.6 cm 2. tumor relationship to the peritoneum ; none 3. circumferential tumor location ; partially encircling 4. longitudinal tumor size: about 2.4 cm 5. t-staging ;ct2 >> ct3 6. circumferential resection margin involvement; ; negative 7. anal sphincter involvement. ; absent 8. mesorectal lymph node (> 8mm, irregular border) ; present 9. extramesorectal lymph node ; none 10. extramural venous invasion (emvi) ; absent.",
"A: Since the last CT, there has been a change in the tumor staging from CT2 to CT3. The lowest tumor margin is about 6.6 cm from the anal verge. The tumor partially encircles the circumference and measures about 2.4 cm longitudinally. There is no involvement of the circumferential resection margin or anal sphincter. Mesorectal lymph node involvement is present, with no extramesorectal lymph nodes observed. Extramural venous invasion is absent.",
"Q: - about 14cm sized solid and cystic mass with enhancing wall at the lt. ischiorectal fossa.",
"A: A solid and cystic mass measuring about 14 cm with an enhancing wall is observed in the left ischiorectal fossa.",
"Q: - multiple nabothian cysts, cervix of uterus. - about 2cm myoma, uterus. - r/o less than 1cm, batholin's glandular cyst, vagina. - no definite evidence of recto viginal fistula.",
"A: Imaging reveals multiple nabothian cysts in the cervix of the uterus, a 2 cm myoma in the uterus, and a probable Bartholin's glandular cyst in the vagina less than 1 cm in size. There is no definite evidence of a rectovaginal fistula.",
"Q: 1. distance of the lowest tumor margin from the anal verge: 6 cm 2. tumor relationship to the peritoneum ; none 3. circumferential tumor location ; partially encircling 4. longitudinal tumor size 2 cm 5. t-staging: r/o t3 6. circumferential resection margin involvement; ; negative 7. anal sphincter involvement. ; absent 8. mesorectal lymph node (> 8mm, irregular border) ; present 9. extramesorectal lymph node ; none 10. extramural venous invasion (emvi) ; equivocal.",
"A: The lowest tumor margin is 6 cm from the anal verge. The tumor partially encircles the circumference and measures 2 cm longitudinally. It is staged as possibly T3, with no involvement of the circumferential resection margin or anal sphincter. Mesorectal lymph node involvement is present, with no extramesorectal lymph nodes observed. Extramural venous invasion is equivocal",
"Q:"
]
'''

new_report = {}

fs_str = " ".join(fs_prompt)

for patient_id, report in report_data.items():
    
    input_text = prompt + fs_str + report
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
    
    with open('new_mri_report.json', 'w') as file:
        json.dump(new_report, file, indent=4)
