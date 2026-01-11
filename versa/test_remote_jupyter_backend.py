from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest

from versa import remote_jupyter_backend as rj


class TestNormalizeServerUrl(unittest.TestCase):
    def test_strips_trailing_slash(self) -> None:
        self.assertEqual(rj.normalize_server_url("https://example.com/foo/"), "https://example.com/foo")

    def test_drops_query_fragment(self) -> None:
        self.assertEqual(
            rj.normalize_server_url("https://example.com/foo?token=abc#frag"),
            "https://example.com/foo",
        )


class TestGlob(unittest.TestCase):
    def test_matches_any_glob(self) -> None:
        globs = ["**/__pycache__/**", "*.pyc", "build/**", ".git/**"]
        self.assertTrue(rj.matches_any_glob("__pycache__/x.py", globs))
        self.assertTrue(rj.matches_any_glob("a/b/c.pyc", globs))
        self.assertTrue(rj.matches_any_glob("build/out.txt", globs))
        self.assertTrue(rj.matches_any_glob(".git/config", globs))
        self.assertFalse(rj.matches_any_glob("src/main.py", globs))


class TestTar(unittest.TestCase):
    def test_create_tar_gz_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"), exist_ok=True)
            os.makedirs(os.path.join(td, "__pycache__"), exist_ok=True)
            with open(os.path.join(td, "src", "a.py"), "w", encoding="utf-8") as f:
                f.write("print('a')\n")
            with open(os.path.join(td, "__pycache__", "a.pyc"), "wb") as f:
                f.write(b"\x00\x01")
            with open(os.path.join(td, "notes.txt"), "w", encoding="utf-8") as f:
                f.write("hello\n")

            out = rj.create_tar_gz(td, ["**/__pycache__/**", "*.pyc"], max_bytes=10 * 1024 * 1024)
            tf = tarfile.open(fileobj=io.BytesIO(out["bytes"]), mode="r:gz")
            names = sorted(tf.getnames())
            self.assertIn("src/a.py", names)
            self.assertIn("notes.txt", names)
            self.assertNotIn("__pycache__/a.pyc", names)


if __name__ == "__main__":
    unittest.main()

