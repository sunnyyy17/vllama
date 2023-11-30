from typing import TYPE_CHECKING, cast
import PIL
import bentoml
from pydantic import BaseModel
from bentoml.io import JSON, Multipart, Text, NumpyNdarray, Image
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import PIL

if TYPE_CHECKING:
    from PIL.Image import Image


TARGET_IMAGE_SIZE = 256
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging purpose
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for debugging purpose

llm_runner = bentoml.models.get("llm_cxr_qa:latest").to_runner()
vq_decoder_runner = bentoml.torchscript.get("vqmodel_decoder:latest").to_runner()
vq_encoder_runner = bentoml.torchscript.get("vqmodel_encoder:latest").to_runner()
svc = bentoml.Service(
    name="llm_cxr_qa", runners=[llm_runner, vq_decoder_runner, vq_encoder_runner]
)


# TextToText
class TextToTextRequestDTO(BaseModel):
    query: str

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class TextResponseDTO(BaseModel):
    answer: str

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


@svc.api(
    input=JSON(pydantic_model=TextToTextRequestDTO),
    output=JSON(pydantic_model=TextResponseDTO),
)
async def t2t(dto: TextToTextRequestDTO) -> TextResponseDTO:
    out = await llm_runner.async_run((dto.query, None), max_new_tokens=512)
    answer = out[0]["generated_text"]
    return TextResponseDTO(
        answer=answer,
    )


# ImageToText
@svc.api(
    input=Multipart(query=Text(), img=Image()),
    output=JSON(pydantic_model=TextResponseDTO),
)
async def i2t(query: str, img: Image) -> TextResponseDTO:
    if not img.mode == "RGB":
        img = img.convert("RGB")
    s = min(img.size)

    if s < TARGET_IMAGE_SIZE:
        raise ValueError(f"min dim for image {s} < {TARGET_IMAGE_SIZE}")

    r = TARGET_IMAGE_SIZE / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [TARGET_IMAGE_SIZE])
    img = T.ToTensor()(img)
    # encode image
    image_indices = await vq_encoder_runner.async_run(img)
    out = await llm_runner.async_run((query, image_indices), max_new_tokens=512)
    answer = out[0]["generated_text"]
    return TextResponseDTO(
        answer=answer,
    )


# TextToImage
class TextToImageRequestDTO(BaseModel):
    instruction: str
    query: str

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


@svc.api(input=JSON(pydantic_model=TextToImageRequestDTO), output=NumpyNdarray())
async def t2i(dto: TextToImageRequestDTO) -> np.ndarray:
    instruction = dto.instruction
    query = dto.query
    out = await llm_runner.async_run((instruction, query), max_new_tokens=512)
    generated_vq = cast(np.ndarray, out[0]["generated_vq"])
    img = await vq_decoder_runner.async_run(torch.tensor(generated_vq))
    return img
