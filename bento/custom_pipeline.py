import logging
from transformers import Pipeline
from vllama.models.mini_gpt4_ita import vllamaIta

class ReportGenerationPipeline(Pipeline):
    def __init__(
        self, 
        *args,
        max_new_tokens = 
        top_p = 
        top_k = 
    )

        super().__init__(
            *args, 
            max_new_tokens=,
            top_p=,
            top_k=,
            **kwargs,
        )

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            #p_before_embeds = self.llama_model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            #p_after_embeds = self.llama_model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    
    def preprocess(self, image, text):

        img_emb, _ = self.model.encode_img(image)
        
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
    
    def forward(self, model_input, **generate_kwargs):
        
        
        