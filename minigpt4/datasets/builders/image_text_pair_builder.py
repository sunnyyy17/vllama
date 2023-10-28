import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.ct_datasets import CTDataset, CTSegDataset


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
        datasets[split] = dataset_cls(img_path=build_info.img_path,
        txt_path=build_info.txt_path , column='impression', size=None, transform=None )

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
