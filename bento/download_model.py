import logging
import sys
import os
import argparse
#from omegaconf import OmegaConf
import torch
import bentoml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vllama.common.config import Config
from vllama.common.registry import registry
from vllama.models.vllamaita import vllamaIta
from custom_pipeline import ReportGenerationPipeline
#from vqmodel_wrapper import VQModelDecoderWrapper, VQModelEncoderWrapper
#from custom_pipeline import InstructionTextGenerationPipeline

from transformers.pipelines import SUPPORTED_TASKS
from transformers import StoppingCriteria, StoppingCriteriaList
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print("start")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Bento Model Generation")
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

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

print("initialize model start")

# mri-rrg
TASK_NAME = "mri-rrg"
TASK_DEFINITION = {
    "impl": ReportGenerationPipeline,
    "tf": (),
    "pt": (),
    "default": {},
    "type": "rrg",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

device = 'cuda'
stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

rrg = ReportGenerationPipeline(
    model=model,
    task="mri-rrg",
    max_new_tokens=300,
    stopping_criteria=stopping_criteria, 
    num_beams=1,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1,
    temperature=1.0,
)

logging.basicConfig(level=logging.DEBUG)

print("save model start")
bentoml.transformers.save_model(
    "mri-rrg",
    pipeline=rrg,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)