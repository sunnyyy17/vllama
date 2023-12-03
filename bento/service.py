from typing import TYPE_CHECKING, cast
import PIL
import bentoml
from pydantic import BaseModel
from bentoml.io import JSON, Multipart, Text, NumpyNdarray, Image
import numpy as np
import os
import torch
#import torchvision.transforms.functional as TF
#import torchvision.transforms as T
#import PIL

#if TYPE_CHECKING:
    #from PIL.Image import Image


TARGET_IMAGE_SIZE = 256

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging purpose
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for debugging purpose

rrg_runner = bentoml.models.get("brain-mri-rrg:latest").to_runner()

svc = bentoml.Service(name="brain-mri-rrg", runners=[rrg_runner])


class TextResponseDTO(BaseModel):
    answer: str

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

# ImageToText
@svc.api(
    input=Multipart(query=Text(), img=NumpyNdarray)),
    output=JSON(pydantic_model=TextResponseDTO),
)

async def reportgen(query: str, img: np.ndarray) -> TextResponseDTO:
    '''
    c = img.size[3]
    m = max(img.size)
    
    if c != 3:
        raise ValueError(f"Incorrect number of channels for MRI image processing")

    if m < TARGET_IMAGE_SIZE:
        raise ValueError(f"min dim for image {s} < {TARGET_IMAGE_SIZE}")
    '''
    ###Prompt Generation
    img_tag = '<Img><ImageHere></Img>'
    prompt = img_tag + query
    model_input = (img, prompt)

    ###Report Generation
    image_indices = await rrg_runner.async_run(model_input, )
    answer = await llm_runner.async_run((query, image_indices), max_new_tokens=200)
    return TextResponseDTO(
        answer=answer,
    )

