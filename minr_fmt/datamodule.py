"""LightningDataModule with Hydra configuration support"""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset.proj_dataset import MultiProjDataset


class TrainingDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule with Hydra config support
    
    Encapsulates all data loading logic with parameter management via Hydra config.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize DataModule
        
        Args:
            cfg: Complete Hydra config (DictConfig)
        """
        super().__init__()
        self.cfg = cfg

        # Extract data configuration
        data_cfg = cfg.data
        self.train_dir = data_cfg.train_dir
        self.val_dir = data_cfg.val_dir
        self.test_dir = data_cfg.test_dir
        self.block_dir = data_cfg.block_dir
        
        if not self.block_dir:
            raise ValueError("data.block_dir must be specified in the Hydra config")
        
        # DataLoader parameters
        self.batch_size = data_cfg.batch_size
        self.eval_batch_size = data_cfg.eval_batch_size
        self.num_workers = data_cfg.num_workers
        self.shuffle = data_cfg.shuffle
        self.pin_memory = data_cfg.pin_memory
        self.persistent_workers = data_cfg.persistent_workers
        self.prefetch_factor = data_cfg.prefetch_factor

        # Dataset objects (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        """
        Create datasets for train/val/test stages
        
        Args:
            stage: 'fit', 'validate', 'test' or None
        """
        if stage in ("fit", "validate", None):
            # Training dataset
            if self.train_dataset is None:
                self.train_dataset = MultiProjDataset(
                    data_dir=self.train_dir,
                    block_dir=self.block_dir,
                    config=self.cfg,
                    is_training=True,
                    split="train",
                )

            # Validation dataset
            if self.val_dataset is None:
                self.val_dataset = MultiProjDataset(
                    data_dir=self.val_dir,
                    block_dir=self.block_dir,
                    config=self.cfg,
                    is_training=False,
                    split="val",
                )

        if stage in ("test", None) and self.test_dir:
            if self.test_dataset is None:
                self.test_dataset = MultiProjDataset(
                    data_dir=self.test_dir,
                    block_dir=self.block_dir,
                    config=self.cfg,
                    is_training=False,
                    split="test",
                )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        if self.train_dataset is None:
            self.setup(stage="fit")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        if self.val_dataset is None:
            self.setup(stage="validate")

        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        if self.test_dataset is None:
            self.setup(stage="test")

        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def get_dataset_info(self) -> dict:
        """Get dataset size information"""
        if self.train_dataset is None or self.val_dataset is None:
            self.setup()

        info = {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
        }
        
        if self.test_dataset:
            info["test_size"] = len(self.test_dataset)
        
        return info
