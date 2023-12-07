import os
import logging
import warnings

from vllama.common.registry import registry
from vllama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
#from vllama.datasets.datasets.laion_dataset import LaionDataset
#from vllama.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from vllama.datasets.datasets.ct_datasets import CTDataset, CTSegDataset, CTSeg3DDataset, ImgEmbedDataset, rectalMRIDataset, brainMRIDataset


@registry.register_builder("brain-mri-3d")
class brainMRI3dbuilder(BaseDatasetBuilder):
    train_dataset_cls = brainMRIDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/brain-mri-3d/defaults.yaml"}
    def build(self):
        build_info = self.config.build_info
        
        datasets = dict()
        split = ["train", "val"]
        dataset_cls = self.train_dataset_cls
        datasets[split[0]] = dataset_cls(img_path=build_info.img_path, txt_path=build_info.txt_path, transform=None, is_train=True)
        datasets[split[1]] = dataset_cls(img_path=build_info.img_path, txt_path=build_info.txt_path, transform=None, is_train=False)
        return datasets

@registry.register_builder("rectal-mri-3d")
class rectalMRI3dbuilder(BaseDatasetBuilder):
    train_dataset_cls = rectalMRIDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/rectal-mri-3d/defaults.yaml"}
    def build(self):
        build_info = self.config.build_info
        
        datasets = dict()
        split = "train"
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(img_path=build_info.img_path, txt_path=build_info.txt_path, transform=None, is_train=True)
        
        return datasets

@registry.register_builder("ct-seg-3d")
class CTSeg3dBuilder(BaseDatasetBuilder):
    train_dataset_cls = CTSeg3DDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/ct-seg-3d/defaults.yaml"}
    def build(self):
        #self.build_processors()
        
        build_info = self.config.build_info
        
        datasets = dict()
        split = "train"
        
        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        #print('len(CTSeg)', len(dataset_cls))
        datasets[split] = dataset_cls(img_path=build_info.img_path,
        txt_path=build_info.txt_path , column='report', transform=None, is_train=True)
        
        return datasets


@registry.register_builder("ct-seg")
class CTSegBuilder(BaseDatasetBuilder):
    train_dataset_cls = CTSegDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/ct-seg/defaults.yaml"}
    def build(self):
        #self.build_processors()
        
        build_info = self.config.build_info
        
        
        datasets = dict()
        split = "train"
        
        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        #print('len(CTSeg)', len(dataset_cls))
        datasets[split] = dataset_cls(img_path=build_info.img_path,
        txt_path=build_info.txt_path , column='impression', size=None, transform=None, is_train=True)

        return datasets

@registry.register_builder("ct")
class CTBuilder(BaseDatasetBuilder):
    train_dataset_cls = CTDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/ct/defaults.yaml"}
    def build(self):
        #self.build_processors()
        build_info = self.config.build_info
        datasets = dict()
        split = "train"
        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(csv_dir=build_info.csv, 
                data_dir=build_info.storage,
                z_length = build_info.z_length,
                image_res= build_info.image_res, 
                is_train=True, is_val=False, is_large=False)

        return datasets
    
@registry.register_builder("ct-img-embed")
class CTEmbedBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImgEmbedDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/ct-image-embed/defaults.yaml"}
    def build(self):
        #self.build_processors()
        build_info = self.config.build_info
        datasets = dict()
        split = "train"
        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(img_embed_path=build_info.img_embed_path, text_path=build_info.text_path)

        return datasets

'''
@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset
    
    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}
    
    def _download_ann(self):
        pass
    
    def _download_vis(self):
        pass
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset
    
    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()
        
        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets
'''