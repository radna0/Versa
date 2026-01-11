from __future__ import annotations

import io
import os
import tarfile
import tempfile
import unittest

from mcp_servers import remote_jupyter_mcp_server as rj


class TestNormalizeServerUrl(unittest.TestCase):
    def test_strips_trailing_slash(self) -> None:
        self.assertEqual(
            rj._normalize_server_url("https://example.com/foo/"),
            "https://example.com/foo",
        )

    def test_drops_query_and_fragment(self) -> None:
        self.assertEqual(
            rj._normalize_server_url("https://example.com/foo?token=abc#frag"),
            "https://example.com/foo",
        )

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            rj._normalize_server_url("")

    def test_rejects_missing_scheme(self) -> None:
        with self.assertRaises(ValueError):
            rj._normalize_server_url("example.com/foo")


class TestAuthHeaders(unittest.TestCase):
    def test_empty_token(self) -> None:
        self.assertEqual(rj._auth_headers(""), {})

    def test_bearer_token(self) -> None:
        self.assertEqual(rj._auth_headers("abc"), {"Authorization": "Bearer abc"})


class TestCsvAndGlob(unittest.TestCase):
    def test_split_csv(self) -> None:
        self.assertEqual(rj._split_csv("a,b, ,c"), ["a", "b", "c"])

    def test_matches_any_glob(self) -> None:
        globs = ["**/__pycache__/**", "*.pyc", "build/**", ".git/**"]
        self.assertTrue(rj._matches_any_glob("__pycache__/x.py", globs))
        self.assertTrue(rj._matches_any_glob("a/b/c.pyc", globs))
        self.assertTrue(rj._matches_any_glob("build/out.txt", globs))
        self.assertTrue(rj._matches_any_glob(".git/config", globs))
        self.assertFalse(rj._matches_any_glob("src/main.py", globs))


class TestCreateTarGz(unittest.TestCase):
    def test_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "src"), exist_ok=True)
            os.makedirs(os.path.join(td, "__pycache__"), exist_ok=True)
            with open(os.path.join(td, "src", "a.py"), "w", encoding="utf-8") as f:
                f.write("print('a')\n")
            with open(os.path.join(td, "__pycache__", "a.cpython-310.pyc"), "wb") as f:
                f.write(b"\x00\x01")
            with open(os.path.join(td, "notes.txt"), "w", encoding="utf-8") as f:
                f.write("hello\n")

            out = rj._create_tar_gz(td, ["**/__pycache__/**", "*.pyc"], max_bytes=10 * 1024 * 1024)
            self.assertGreater(out["size_bytes"], 0)
            self.assertEqual(out["included"], 2)  # src/a.py + notes.txt

            tf = tarfile.open(fileobj=io.BytesIO(out["bytes"]), mode="r:gz")
            names = sorted(tf.getnames())
            self.assertIn("src/a.py", names)
            self.assertIn("notes.txt", names)
            self.assertNotIn("__pycache__/a.cpython-310.pyc", names)


if __name__ == "__main__":
    unittest.main()
