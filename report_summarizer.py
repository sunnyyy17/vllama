import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import nltk
import csv
import argparse
import json

from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.ct_datasets import rectalMRIDataset
from transformers import StoppingCriteria, StoppingCriteriaList

from minigpt4.models.mini_gpt4_ita_frozen import MiniGPT4ItaFrozen 
from minigpt4.models.mini_gpt4_ita import MiniGPT4Ita
#from torchmetrics.text.rouge import ROUGEScore
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cudnn.benchmark = False
    cudnn.deterministic = True

pretrained_checkpoint = 'checkpoint_10.pth'

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

txt_path = '/data/changsun/data/MRI/rectal/202301_MRI_impression_final.json'
#dataset = rectalMRIDataset(img_path, txt_path, None, False)
#test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = 'cuda:1'


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def get_context_emb(self, text, img_list):
        #prompt = conv.get_prompt()
        #prompt =
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

#prompt = '<Img><ImageHere></Img> Could you describe the contents of this image for me?'

max_new_tokens=300
num_beams=1
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=1
temperature=1.0
max_length=2000
stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

with open(txt_path, 'r') as rfile:
    report_data = json.load(rfile)


prompt = "Make this following MRI report more concise by only containing clinically important information: "
new_report = {}
with torch.no_grad():
    for patient_id, report in report_data.items():
    
        input_text = prompt + report
        #input_image = vis_processor(image).to(device)
        #image_emb, atts_img, _ = model.encode_img(image)
        #input_emb, _ = model.prompt_wrap(image_emb, atts_img, prompt)
        
        #print('input_emb.shape', input_emb.shape)
        input_tokens = model.llama_tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True)
        
        #print(input_tokens)
        input_tokens.to(device)
        outputs = model.llama_model.generate(
            input_ids = input_tokens.input_ids,
            attention_mask = input_tokens.attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=False,
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
        
        #print()
        new_report[patient_id] = output_text
        
        print('==================================')
        print('GT Report: ', report)
        print('Candidate: ', output_text)
        print('==================================')
        
    with open('new_mri_report.json', 'w') as file:
        json.dump(new_report, file, indent=4)
    

    