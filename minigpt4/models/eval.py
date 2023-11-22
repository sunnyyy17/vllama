import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.datasets import rectalMRIDataset

from minigpt4.models.mini_gpt4_ita_frozen import MiniGPT4ItaFrozen 
from minigpt4.models.mini_gpt4_ita import MiniGPT4Ita

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

pretrained_checkpoint = 'checkpoint_10.pth'

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

dataset = rectalMRIDataset(img_path, txt_path, None, False)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = 'cuda'

text = 
for idx, item in enumerate(tqdm(test_dataloader)):
    image = item[0]
    text = item[1]
    input_image = vis_processor(image).to(device)
    image_emb, _ = model.encode_img(image)
    