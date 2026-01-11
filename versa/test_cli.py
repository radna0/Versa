from __future__ import annotations

import unittest

from typer.testing import CliRunner

from versa.cli import app


class TestVersaCli(unittest.TestCase):
    def test_root_help_shows_run_command(self) -> None:
        r = CliRunner().invoke(app, ["--help"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("Commands", r.output)
        self.assertIn("run", r.output)

    def test_run_help_is_subcommand(self) -> None:
        r = CliRunner().invoke(app, ["run", "--help"])
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertIn("Usage:", r.output)
        self.assertIn("run", r.output)


if __name__ == "__main__":
    unittest.main()

