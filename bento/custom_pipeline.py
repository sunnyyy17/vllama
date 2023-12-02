import logging
import numpy as np
import kornia
from transformers import Pipeline
from vllama.models.vllamaita import vllamaIta
#from ct_datasets import brainMRIDataset


class InvalidFileError(Exception):
    pass

class InvalidDimensionError(Exception):
    pass

def mri_preprocess(np_img):

    try:

        volume = np.load(np_img)

        if volume.ndim != expected_dimensions:
            raise InvalidDimensionError(f"The file has incorrect dimensions: expected {expected_dimensions}, got {volume.ndim}")
        
        preproc_frames = preproc_frames.astype(np.float32)
        preproc_frames = torch.from_numpy(preproc_frames)
        
        preproc_frames = preproc_frames.permute(0, 3, 1, 2)
        
        # Brain window
        preproc_frames_brain = preproc_frames[:, 0, :, :].clone()
        preproc_frames_brain[preproc_frames_brain > 80] = 80
        preproc_frames_brain[preproc_frames_brain < 0] = 0
        preproc_frames_brain = (preproc_frames_brain - preproc_frames_brain.min()) / (preproc_frames_brain.max() - preproc_frames_brain.min())
        preproc_frames_brain = preproc_frames_brain.unsqueeze(1)
        
        # Subdural window
        preproc_frames_subdural = preproc_frames[:, 1, :, :].clone()
        preproc_frames_subdural[preproc_frames_subdural > 170] = 170
        preproc_frames_subdural[preproc_frames_subdural < -10] = -10
        preproc_frames_subdural = (preproc_frames_subdural - preproc_frames_subdural.min()) / (preproc_frames_subdural.max() - preproc_frames_subdural.min())
        preproc_frames_subdural = preproc_frames_subdural.unsqueeze(1)
        
        # Bone window
        preproc_frames_bone = preproc_frames[:, 2, :, :].clone().float()
        preproc_frames_bone[preproc_frames_bone > 1500] = 1500
        preproc_frames_bone[preproc_frames_bone < -500] = -500
        preproc_frames_bone = (preproc_frames_bone - preproc_frames_bone.min()) / (preproc_frames_bone.max() - preproc_frames_bone.min())
        preproc_frames_bone = preproc_frames_bone.unsqueeze(1)

        preproc_frames_cat = torch.cat([preproc_frames_brain, preproc_frames_brain, preproc_frames_brain], dim=1)
        preproc_frames_cat = kornia.geometry.transform.resize(preproc_frames_cat,
                                                              size=(224, 224))

        return preproc_frames_cat
    
    except FileNotFoundError:
        raise InvalidFileError("File not found or not a valid numpy file")

    except IOError:
        raise InvalidFileError("Error reading the file. It might not be a valid numpy file")

    except InvalidDimensionError as e:
        # You can handle specific dimension error here if needed
        raise e


class ReportGenerationPipeline(Pipeline):

    def __init__(self, model, task, **kwargs):

        super().__init__(self, model, task, **kwargs)
        
        self.model = model
        self.task = task

    def preprocess(self, image):
        
        preproc_img = mri_preprocess(image)
        img_emb, _ = self.model.encode_img(preproc_img)
        
        return img_emb
        
    def get_mixed_emb(self, prompt, img_emb):
        
        if prompt:
            prompt_segs = prompt.split('<ImageHere>')
            assert len(prompt_segs) == len(image_emb) + 1
            seg_tokens = [
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            
            seg_embs = [self.model.llama_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
            #print('seg_embs.shape', seg_embs.shape)
            wrapped_img_embs = torch.cat([seg_embs[:-1], img_emb, seg_embs[-1]], dim=1)
            #mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]

            return wrapped_img_embs
        else:
            return img_emb
    
    def _forward(self, model_input, **generate_kwargs):
        
        image = model_input[0]
        prompt = model_input[1]
        img_emb = self.preprocess(image)
        input_emb = self.get_mixed_emb(prompt, img_emb)
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
        
        return 
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        
        return output_text 
        
        