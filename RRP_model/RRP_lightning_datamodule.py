import pytorch_lightning as pl
from torch.utils.data import DataLoader
from RRP_model.RRP_dataset import RRP_Dataset

class FlonaDataModule(pl.LightningDataModule):
    def __init__(self, data_config: dict, batch_size: int, num_workers: int, eval_batch_size: int = None):
        super().__init__()
        self.data_config = data_config
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = RRP_Dataset(
                data_folder=self.data_config["data_folder"],
                data_splits_path=self.data_config["data_splits"],
                split="train",
                rgb_image_size=self.data_config["rgb_img_size"],
                floorplan_img_size=self.data_config["floorplan_img_size"],
            )
            self.test_dataset = RRP_Dataset(
                data_folder=self.data_config["data_folder"],
                data_splits_path=self.data_config["data_splits"],
                split="test",
                rgb_image_size=self.data_config["rgb_img_size"],
                floorplan_img_size=self.data_config["floorplan_img_size"],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
