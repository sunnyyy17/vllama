import logging
import os
#from omegaconf import OmegaConf
import torch
import bentoml
from vllama.common.config import Config
from vllama.models.vllamaita import vllamaIta
from custom_pipeline import ReportGenerationPipeline
#from vqmodel_wrapper import VQModelDecoderWrapper, VQModelEncoderWrapper
#from custom_pipeline import InstructionTextGenerationPipeline

from transformers.pipelines import SUPPORTED_TASKS
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print("start")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


rrg = ReportGenerationPipeline(
    model=model,
    task="mri-rrg",
)

logging.basicConfig(level=logging.DEBUG)

print("save model start")
bentoml.transformers.save_model(
    "mri-rrg",
    pipeline=rrg,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)


'''
# mri-rrg
TASK_NAME = "mri-rrg"
TASK_DEFINITION = {
    "impl": InstructionTextGenerationPipeline,
    "tf": (),
    "pt": (AutoModelForCausalLM,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

checkpoint_path = "ckpt/llmcxr_mimic-cxr-256-txvloss-medvqa-stage1_2"

print("initialize model start")
qa = InstructionTextGenerationPipeline(
    model=AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    task="llm-cxr-qa",
    tokenizer=AutoTokenizer.from_pretrained(checkpoint_path, padding_side="left"),
)

logging.basicConfig(level=logging.DEBUG)

print("save model start")
bentoml.transformers.save_model(
    "llm_cxr_qa",
    pipeline=qa,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)

## vqgan
# gumbel-vq : torch-lightning
print("start vqgan")
checkpoint_path = "ckpt/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-4e-compat.ckpt"

# decoder
vq_decoder_wrapper = VQModelDecoderWrapper(
    **OmegaConf.load(
        "ckpt/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml"
    ).model.params,
    ckpt_path=checkpoint_path,
).to(torch.device("cuda"))
vq_decoder_wrapper.to_torchscript("decoder.pt", method="trace", example_inputs=indices)
decoder_model = torch.jit.load("decoder.pt").to("cuda:0")

image = decoder_model(indices)
plt.imsave("example.png", image.cpu().numpy())
print("save decoder_model start")
bentoml.torchscript.save_model("vqmodel_decoder", decoder_model)

# encoder
vq_encoder_wrapper = VQModelEncoderWrapper(
    **OmegaConf.load(
        "ckpt/vqgan_mimic-cxr-256-txvloss/2023-09-05T13-56-50_mimic-cxr-256-txvloss-project-compat.yaml"
    ).model.params,
    ckpt_path=checkpoint_path,
).to(torch.device("cuda"))
img = Image.open("example.png")
if not img.mode == "RGB":
    img = img.convert("RGB")
s = min(img.size)

if s < 256:
    raise ValueError(f"min dim for image {s} < {256}")

r = 256 / s
s = (round(r * img.size[1]), round(r * img.size[0]))
img = TF.resize(img, s, interpolation=Image.LANCZOS)
img = TF.center_crop(img, output_size=2 * [256])
img = T.ToTensor()(img)

vq_encoder_wrapper.to_torchscript("encoder.pt", method="trace", example_inputs=img)
encoder_model = torch.jit.load("encoder.pt").to("cuda:0")
print("save encoder_model start")
bentoml.torchscript.save_model("vqmodel_encoder", encoder_model)

'''