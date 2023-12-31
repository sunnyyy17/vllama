"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import random
import os
import csv
import pdb
import torch
import torch.distributed as dist
from vllama.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from vllama.common.logger import MetricLogger, SmoothedValue
from vllama.common.registry import registry
from vllama.datasets.data_utils import prepare_sample
from transformers import StoppingCriteria, StoppingCriteriaList

#from vllama.models.vllama_ita_frozen import vllamaItaFrozen 
#from vllama.models.vllama_ita import vllamaIta
from nltk.translate.bleu_score import sentence_bleu
#from torchmetrics.text.rouge import ROUGEScore

from torch.profiler import profile, record_function, ProfilerActivity

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            stop = stop.to('cuda')
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
max_new_tokens=200
num_beams=1
min_length=1
top_p=0.9
repetition_penalty=1.0
length_penalty=1
temperature=1.0
max_length=2000
#stop_words_ids = [torch.tensor([835]), torch.tensor([2277, 29937])]
#stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
total_bleu_four = 0
total_rouge_l_prec = 0
total_rouge_l_rec = 0
total_rouge_l_f = 0

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.
    
        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        
        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            #print(name)
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio
            
            datasets[name] = dataset
        
        return datasets
    
    def train_step(self, model, samples):
        #with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        #print('samples', samples)
        #print('samples[0]', samples[0])
        loss = model(samples)["loss"]
        #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print('loss:', loss)
        return loss 

    def valid_step(self, model, samples):
        
        #image = samples[0]
        #text = samples[1]
        img_emb, atts_img, _ = model.encode_img(samples[0], samples[2])
                
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            input_embeds, atts_img = model.prompt_wrap(img_emb, atts_img, vqa_prompt)
        elif model.prompt_list:
            prompt = random.choice(model.prompt_dict[samples[2][0]])
            input_embeds, _ = model.prompt_wrap(img_emb, atts_img, prompt)
        
        input_embeds = input_embeds.to('cuda')
        outputs = model.llama_model.generate(
            inputs_embeds=input_embeds, 
            max_new_tokens=max_new_tokens
        )
        #raise NotImplementedError
        outputs = outputs.cpu().detach()

        return outputs
    
    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass
    
    def inference_step(self):
        raise NotImplementedError
    
    def evaluation(self, model, data_loader, eval_iters_per_epoch, save_txt_path, cuda_enabled=True):
        
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10
        
        #rouge = ROUGEScore(rouge_keys='rougeL')

        total_bleu_four = 0
        #total_rouge_l_prec = 0
        #total_rouge_l_rec = 0
        total_rouge_l_f = 0

        results = {}
            
        #for samples in metric_logger.log_every(data_loader, print_freq, header):
        for i in metric_logger.log_every(range(eval_iters_per_epoch), print_freq, header):
            print("Index: ", i)
            if i >=eval_iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            eval_output = self.valid_step(model=model, samples=samples)
            text = samples[1]
            output_token = eval_output[0]

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
            #rouge_score = rouge(output_text, text)

            total_bleu_four += bleu_score
            #total_rouge_l_prec += rouge_score['rougeL_precision']
            #total_rouge_l_rec += rouge_score['rougeL_recall']
            #total_rouge_l_f += rouge_score['rougeL_fmeasure']
        
        avg_bleu_four = total_bleu_four / eval_iters_per_epoch
        #avg_rouge_l_prec = total_rouge_l_prec / len(data_loader)
        #avg_rouge_l_rec = total_rouge_l_rec / len(data_loader)
        #avg_rouge_l_f = total_rouge_l_f / len(data_loader)
        
        
        results['answer'] = output_text 
        results['gt'] = text
        results['BLEU-4'] = avg_bleu_four
        #results['ROUGE-L'] = avg_rouge_l_f
        with open(save_txt_path, 'a') as csv_file:
            new_gen = [results['answer'], results['gt'], results['BLEU-4']]
            writer = csv.writer(csv_file)
            writer.writerow(new_gen)
        if is_dist_avail_and_initialized():
                dist.barrier()
        
        return results
    
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )
    
    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )
    
    def plot_grad_flow(named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    
        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            
            
            #print('data_loader', data_loader)
            #print('len(data_loader)', len(data_loader))
            #print('data_loader[0]', data_loader[0])
            samples = next(data_loader)
            
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            #print(type(samples))
            #print('len :', len(samples))
            #print(samples[0].shape)
            #print(print(samples[1]))
            
            '''
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            '''
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            #with torch.cuda.amp.autocast(enabled=use_amp):
            loss = self.train_step(model=model, samples=samples)
            
            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
                #plot_grad_flow(model.named_parameters())
                print('loss backward activated')
            else:
                loss.backward()
            
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    #scaler.unscale_(optimizer)##
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)##
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)
        
        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()
        
        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res
            
            if remove_duplicate:
                result_new = []
                
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file) 
        
        return final_result_file
