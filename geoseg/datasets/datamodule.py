from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch.utils.data import DataLoader


@runtime_checkable
class SegmentationDataModule(Protocol):
    def setup(self, stage: str | None = None) -> None: ...

    def train_dataloader(self) -> DataLoader: ...

    def val_dataloader(self) -> DataLoader: ...

    def test_dataloader(self) -> DataLoader: ...
