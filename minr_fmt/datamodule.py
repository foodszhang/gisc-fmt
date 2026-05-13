"""LightningDataModule with Hydra configuration support"""

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset.fmt_simgen_dataset import FmtSimGenProjDataset
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
        self.dataset_type = str(data_cfg.get("dataset_type", "multiproj"))
        self.train_dir = data_cfg.train_dir
        self.val_dir = data_cfg.val_dir
        self.test_dir = data_cfg.test_dir
        self.block_dir = data_cfg.get("block_dir")

        if self.dataset_type != "fmt_simgen" and not self.block_dir:
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
                self.train_dataset = self._make_dataset(self.train_dir, True, "train")

            # Validation dataset
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset(self.val_dir, False, "val")

        if stage in ("test", None) and self.test_dir:
            if self.test_dataset is None:
                self.test_dataset = self._make_dataset(self.test_dir, False, "test")

    def _make_dataset(self, data_dir, is_training: bool, split: str):
        if self.dataset_type == "fmt_simgen":
            return FmtSimGenProjDataset(
                data_dir=data_dir,
                config=self.cfg,
                is_training=is_training,
                split=split,
            )
        return MultiProjDataset(
            data_dir=data_dir,
            block_dir=self.block_dir,
            config=self.cfg,
            is_training=is_training,
            split=split,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        if self.train_dataset is None:
            self.setup(stage="fit")

        kwargs = self._loader_kwargs(shuffle=self.shuffle, batch_size=self.batch_size)
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        if self.val_dataset is None:
            self.setup(stage="validate")

        kwargs = self._loader_kwargs(shuffle=False, batch_size=self.eval_batch_size)
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        if self.test_dataset is None:
            self.setup(stage="test")

        kwargs = self._loader_kwargs(shuffle=False, batch_size=self.eval_batch_size)
        return DataLoader(self.test_dataset, **kwargs)

    def _loader_kwargs(self, shuffle: bool, batch_size: int) -> dict:
        kwargs = {
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "shuffle": shuffle,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

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
