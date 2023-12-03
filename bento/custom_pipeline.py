import logging
import numpy as np
import torch
import torch.nn as nn
import kornia
#from transformers import Pipeline, AutoConfig
from transformers import StoppingCriteria, StoppingCriteriaList
#from vllama.models.vllamaita import vllamaIta
#from ct_datasets import brainMRIDataset


class InvalidFileError(Exception):
    pass

class InvalidDimensionError(Exception):
    pass

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

#device = 'cuda:1'
#stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]
#stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

def mri_preprocess(np_img):

    try:

        #volume = np.load(np_img)
        
        expected_dimensions=4
        preproc_frames = np_img

        if preproc_frames.ndim != expected_dimensions:
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


class ReportGenerationPipeline(nn.Module):

    def __init__(self, model, **kwargs):
        super().__init__()

        self.model = model
        self.device = kwargs.get('device', 'cuda:1')
        self.default_params = {
            'max_new_tokens': kwargs.get('max_new_tokens', 300),
            'num_beams': kwargs.get('num_beams', 1),
            'min_length': kwargs.get('min_length', 1),
            'top_p': kwargs.get('top_p', 0.9),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.0),
            'length_penalty': kwargs.get('length_penalty', 1),
            'temperature': kwargs.get('temperature', 1.0),
            'stopping_criteria': kwargs.get('stopping_criteria', StoppingCriteriaList([StoppingCriteriaSub(stops=[torch.tensor([835]).to(self.device), torch.tensor([2277, 29937]).to(self.device)])])),
        }
        '''
        self.max_new_tokens = kwargs.get('max_new_tokens', 300)
        self.stopping_criteria = kwargs.get('stopping_criteria', stopping_criteria)
        self.num_beams = kwargs.get('num_beams', 1)
        self.min_length = kwargs.get('min_length', 1)
        self.top_p = kwargs.get('top_p', 0.9)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.length_penalty = kwargs.get('length_penalty', 1)
        self.temperature = kwargs.get('temperature', 1.0)
        '''
        #self.config = AutoConfig.from_pretrained(model)
        #self.model = model
        #self.task = task
    
    def preprocess(self, image):
        
        preproc_img = mri_preprocess(image)
        img_emb, _ = self.model.encode_img(preproc_img)
        
        return img_emb
        
    def get_mixed_emb(self, prompt, img_emb):
        
        if prompt:
            prompt_segs = prompt.split('<ImageHere>')
            assert len(prompt_segs) == len(img_emb) + 1
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

    '''
    def _sanitize_parameters(self, **kwargs):

        preprocess_params = {}
        forward_params = {key: kwargs[key] for key in kwargs if key in ["max_new_tokens", "num_beams", "min_length", "top_p", "repetition_penalty", "length_penalty", "temperature"]}
        postprocess_params = {}

        return preprocess_params, forward_params, postprocess_params
    '''
    
    def forward(self, model_input, **generate_kwargs):
        
        image = model_input[0]
        prompt = model_input[1]
        img_emb = self.preprocess(image)
        input_emb = self.get_mixed_emb(prompt, img_emb)
        outputs = model.llama_model.generate(
            inputs_embeds=input_emb, 
            max_new_tokens=self.default_params['max_new_tokens'],
            stopping_criteria=self.default_params['stopping_criteria'],
            num_beams=self.default_params['num_beams'],
            do_sample=True,
            min_length=self.default_params['min_length'],
            top_p=self.default_params['top_p'],
            repetition_penalty=self.default_params['repetition_penalty'],
            length_penalty=self.default_params['length_penalty'],
            temperature=self.default_params['temperature'],
        )
        
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        
        return output_text 
    '''
    def postprocess(self):
        return
    '''