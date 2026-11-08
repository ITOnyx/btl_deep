"""
Unit tests for manifest integrity and load_index verification.
Ensures that any tampering of checkpoint_hash or catalog_hash causes load_index to refuse execution.
"""
import unittest
import tempfile
import pathlib
import json
import shutil
import faiss
import numpy as np

import sys
sys.path.insert(0, ".")
from src.search import load_index, verify_manifest, compute_file_sha256


class TestManifestVerification(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = pathlib.Path(self.temp_dir.name)

        self.index_dir = self.temp_path / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy checkpoint file
        self.dummy_ckpt = self.temp_path / "dummy_checkpoint.pt"
        self.dummy_ckpt.write_bytes(b"dummy checkpoint weights content 12345")
        self.ckpt_hash = compute_file_sha256(str(self.dummy_ckpt))

        # Create dummy catalog file
        self.dummy_cat = self.temp_path / "dummy_catalog.parquet"
        self.dummy_cat.write_bytes(b"dummy parquet table content 67890")
        self.cat_hash = compute_file_sha256(str(self.dummy_cat))

        # Create dummy faiss index
        dim = 8
        dummy_index = faiss.IndexHNSWFlat(dim, 4)
        dummy_index.add(np.random.randn(5, dim).astype(np.float32))
        self.hnsw_file = self.index_dir / "faiss_hnsw_index.bin"
        faiss.write_index(dummy_index, str(self.hnsw_file))

        # Create dummy id_map
        self.id_map_file = self.index_dir / "id_map.json"
        with open(self.id_map_file, "w", encoding="utf-8") as f:
            json.dump([{"item_id": i} for i in range(5)], f)

        # Create valid manifest
        self.valid_manifest = {
            "checkpoint_path": str(self.dummy_ckpt),
            "checkpoint_hash": self.ckpt_hash,
            "catalog_path": str(self.dummy_cat),
            "catalog_hash": self.cat_hash,
            "files": {
                "hnsw_index": "faiss_hnsw_index.bin",
                "id_map": "id_map.json"
            }
        }
        self.manifest_file = self.index_dir / "manifest.json"
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.valid_manifest, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_manifest_loads_successfully(self):
        index, id_map, manifest = load_index(
            index_dir=self.index_dir,
            checkpoint_path=self.dummy_ckpt,
            catalog_path=self.dummy_cat
        )
        self.assertEqual(len(id_map), 5)
        self.assertEqual(manifest["checkpoint_hash"], self.ckpt_hash)
        self.assertEqual(manifest["catalog_hash"], self.cat_hash)

    def test_checkpoint_hash_mismatch_raises_value_error(self):
        tampered_manifest = dict(self.valid_manifest)
        tampered_manifest["checkpoint_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f, indent=2)

        with self.assertRaises(ValueError) as ctx:
            load_index(
                index_dir=self.index_dir,
                checkpoint_path=self.dummy_ckpt,
                catalog_path=self.dummy_cat
            )
        self.assertIn("CHECKPOINT HASH MISMATCH", str(ctx.exception))

    def test_catalog_hash_mismatch_raises_value_error(self):
        tampered_manifest = dict(self.valid_manifest)
        tampered_manifest["catalog_hash"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(tampered_manifest, f, indent=2)

        with self.assertRaises(ValueError) as ctx:
            load_index(
                index_dir=self.index_dir,
                checkpoint_path=self.dummy_ckpt,
                catalog_path=self.dummy_cat
            )
        self.assertIn("CATALOG HASH MISMATCH", str(ctx.exception))

    def test_missing_manifest_raises_file_not_found(self):
        self.manifest_file.unlink()
        with self.assertRaises(FileNotFoundError):
            load_index(
                index_dir=self.index_dir,
                checkpoint_path=self.dummy_ckpt,
                catalog_path=self.dummy_cat
            )


if __name__ == "__main__":
    unittest.main()
