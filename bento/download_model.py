import logging
import os
from omegaconf import OmegaConf
import torch
import bentoml
#from vqmodel_wrapper import VQModelDecoderWrapper, VQModelEncoderWrapper
#from custom_pipeline import InstructionTextGenerationPipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from taming.models.vqgan import VQModel
from transformers.pipelines import SUPPORTED_TASKS
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

print("start")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# llm-cxr-qa
TASK_NAME = "llm-cxr-qa"
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

indices = torch.tensor(
    [
        955,
        245,
        63,
        127,
        981,
        1002,
        829,
        147,
        665,
        716,
        447,
        973,
        533,
        329,
        659,
        151,
        61,
        127,
        410,
        920,
        439,
        203,
        600,
        921,
        202,
        573,
        742,
        885,
        687,
        71,
        22,
        807,
        22,
        621,
        764,
        764,
        66,
        742,
        742,
        716,
        807,
        551,
        860,
        410,
        15,
        144,
        719,
        1012,
        401,
        764,
        87,
        122,
        339,
        716,
        258,
        144,
        193,
        551,
        203,
        122,
        333,
        764,
        428,
        921,
        634,
        921,
        860,
        245,
        961,
        637,
        973,
        345,
        665,
        63,
        909,
        1012,
        200,
        905,
        322,
        680,
        133,
        660,
        322,
        339,
        551,
        699,
        22,
        621,
        87,
        742,
        250,
        961,
        127,
        845,
        333,
        1012,
        77,
        295,
        205,
        203,
        82,
        71,
        144,
        468,
        202,
        807,
        1012,
        333,
        69,
        331,
        144,
        680,
        468,
        921,
        331,
        981,
        955,
        232,
        1002,
        467,
        446,
        551,
        410,
        716,
        533,
        845,
        406,
        147,
        260,
        514,
        331,
        533,
        243,
        955,
        468,
        634,
        534,
        742,
        127,
        1012,
        921,
        119,
        600,
        981,
        22,
        534,
        528,
        127,
        807,
        973,
        634,
        401,
        77,
        345,
        404,
        932,
        528,
        814,
        243,
        71,
        250,
        808,
        885,
        716,
        932,
        447,
        634,
        790,
        999,
        345,
        428,
        243,
        406,
        814,
        514,
        637,
        659,
        845,
        1012,
        468,
        790,
        421,
        790,
        468,
        243,
        34,
        719,
        824,
        193,
        163,
        842,
        250,
        77,
        193,
        930,
        295,
        105,
        119,
        790,
        119,
        439,
        428,
        829,
        147,
        660,
        401,
        529,
        699,
        200,
        808,
        30,
        133,
        30,
        529,
        151,
        200,
        845,
        202,
        845,
        660,
        66,
        331,
        82,
        961,
        818,
        467,
        61,
        785,
        108,
        955,
        818,
        163,
        687,
        329,
        533,
        528,
        330,
        15,
        329,
        961,
        71,
        1012,
        764,
        439,
        304,
        1012,
        860,
        596,
        163,
        842,
        999,
        845,
        905,
        514,
        716,
        122,
    ]
)

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
