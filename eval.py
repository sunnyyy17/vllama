import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import nltk
import csv
import argparse
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from rouge import Rouge 
from tqdm import tqdm
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets.datasets.ct_datasets import rectalMRIDataset
from transformers import StoppingCriteria, StoppingCriteriaList

from minigpt4.models.mini_gpt4_ita_frozen import MiniGPT4ItaFrozen 
from minigpt4.models.mini_gpt4_ita import MiniGPT4Ita
from torchmetrics.text.rouge import ROUGEScore
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

#pretrained_checkpoint = 'checkpoint_10.pth'

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

img_path = '/data/changsun/data/MRI/brain/brain_MRI_volume/'
txt_path = '/data/changsun/data/MRI/brain/mri_3d_report.csv'
dataset = brainMRIDataset(img_path, txt_path, None, False)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

prompt = '<Img><ImageHere></Img> Could you describe the contents of this image for me?'

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
total_bleu_four = 0
total_rouge_l_prec = 0
total_rouge_l_rec = 0
total_rouge_l_f = 0

rouge = ROUGEScore(rouge_keys='rougeL')
'''
rouge = Rouge(metrics=['rouge-l'],
              length_limit=100,
              length_limit_type='words',
              apply_avg=False,
              apply_best=False,
              alpha=0.5, # Alpha for F1-score
              stemming=True)
'''
with torch.no_grad():
    for idx, item in enumerate(tqdm(test_dataloader)):
        image = item[0]
        text = item[1]
        
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

        bleu_score = sentence_bleu(text, output_text, weights=(0.25, 0.25, 0.25, 0.25))
        rouge_score = rouge(output_text, text)
        #print(rouge_score)
        total_bleu_four += bleu_score
        total_rouge_l_prec += rouge_score['rougeL_precision']
        total_rouge_l_rec += rouge_score['rougeL_recall']
        total_rouge_l_f += rouge_score['rougeL_fmeasure']
    

    avg_bleu_four = total_bleu_four / len(test_dataloader)
    avg_rouge_l_prec = total_rouge_l_prec / len(test_dataloader)
    avg_rouge_l_rec = total_rouge_l_rec / len(test_dataloader)
    avg_rouge_l_f = total_rouge_l_f / len(test_dataloader)

    print('BLEU-4', avg_bleu_four, 'ROUGE-L', avg_rouge_l_prec, avg_rouge_l_rec, avg_rouge_l_f)

#with open('eval_metric.csv', 'w') as csvfile:


    
    

    