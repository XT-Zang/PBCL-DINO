from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

import run
from geoseg.pretrained import (
    DINOV3_CHECKPOINT_SHA256,
    DINOV3_CHECKPOINT_URL,
    pretrained_weight_path,
    resolve_pretrained_model_path,
)


class PublicDefaultsTest(unittest.TestCase):
    def test_runner_defaults_to_seed_42(self) -> None:
        parser = run.build_parser()

        for command in ("inspect", "preflight", "train"):
            with self.subTest(command=command):
                self.assertEqual(parser.parse_args([command]).seed, 42)

    def test_default_pretrained_path_follows_pytorch_hub(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hub_dir = Path(directory) / "hub"
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(torch.hub, "get_dir", return_value=str(hub_dir)):
                    path = pretrained_weight_path("weights.pth")

        self.assertEqual(path, (hub_dir / "checkpoints" / "weights.pth").resolve())

    def test_missing_pretrained_weights_are_downloaded_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(os.environ, {"GEOSEG_PRETRAINED_ROOT": str(root)}):
                with mock.patch.object(torch.hub, "download_url_to_file") as download:
                    download.side_effect = lambda _url, destination, **_kwargs: Path(destination).write_bytes(
                        b"checkpoint"
                    )

                    first = resolve_pretrained_model_path("dinov3-vitl16-sat493m")
                    second = resolve_pretrained_model_path("dinov3-vitl16-sat493m")

        self.assertEqual(first, second)
        download.assert_called_once_with(
            DINOV3_CHECKPOINT_URL,
            str(first),
            hash_prefix=DINOV3_CHECKPOINT_SHA256,
            progress=True,
        )


if __name__ == "__main__":
    unittest.main()
