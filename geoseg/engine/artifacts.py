from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from geoseg.datasets.aquav3_dataset import save_aquav3_prediction_preview


@dataclass(frozen=True)
class BinaryValidationCPUBuffer:
    probabilities: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor


@torch.no_grad()
def build_binary_validation_cpu_buffer(
    probabilities: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> BinaryValidationCPUBuffer:
    packed = torch.stack((probabilities, mask.detach().to(probabilities.dtype))).detach().cpu()
    return BinaryValidationCPUBuffer(
        probabilities=packed[0],
        predictions=packed[0] > float(threshold),
        targets=packed[1] > 0.5,
    )


class ValidationArtifactWriter:
    """Write validation artifacts from already-computed probabilities."""

    def __init__(self, config: Any) -> None:
        self.enabled = bool(getattr(config, "save_val_predictions", False))
        self.threshold = float(getattr(config, "metric_threshold", 0.5))
        self.output_root = Path(
            getattr(
                config,
                "val_prediction_dir",
                Path(getattr(config, "weights_path")) / "val_predictions",
            )
        )
        self.async_writes = bool(getattr(config, "async_val_writes", getattr(config, "async_writes", False)))
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="geoseg-artifacts") if self.async_writes else None
        self._pending: list[Future[None]] = []

    @staticmethod
    def _write_item(
        epoch_dir: Path,
        image_id: str,
        image: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        pred_array = prediction.squeeze().numpy().astype(np.uint8) * 255
        Image.fromarray(pred_array).save(epoch_dir / f"{image_id}_pred.png")
        save_aquav3_prediction_preview(
            image,
            target,
            prediction,
            epoch_dir / f"{image_id}_preview.jpg",
        )

    @torch.no_grad()
    def save_binary_batch(
        self,
        batch: Mapping[str, Any],
        *,
        probabilities: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
        global_step: int,
        cpu_buffer: BinaryValidationCPUBuffer | None = None,
    ) -> None:
        if not self.enabled:
            return
        epoch_dir = self.output_root / f"epoch_{int(epoch):03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        ids = batch.get("img_id", batch.get("tile_id"))
        images_cpu = batch["img"].detach().cpu()
        buffer = cpu_buffer or build_binary_validation_cpu_buffer(
            probabilities,
            mask,
            threshold=self.threshold,
        )
        masks_cpu = buffer.targets
        predictions_cpu = buffer.predictions
        for index in range(int(predictions_cpu.shape[0])):
            image_id = str(ids[index]) if ids is not None else f"batch{int(global_step):08d}_{index:02d}"
            arguments = (
                epoch_dir,
                image_id,
                images_cpu[index].clone() if self._executor else images_cpu[index],
                masks_cpu[index].clone() if self._executor else masks_cpu[index],
                predictions_cpu[index].clone() if self._executor else predictions_cpu[index],
            )
            if self._executor is None:
                self._write_item(*arguments)
            else:
                self._pending.append(self._executor.submit(self._write_item, *arguments))

    def flush(self) -> None:
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def close(self) -> None:
        self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
