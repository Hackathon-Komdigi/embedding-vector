import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AppConfigTests(unittest.TestCase):
    def test_load_app_env_uses_module_directory_and_preserves_process_environment(self):
        from app_config import load_app_env

        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "DENSE_MODEL=from-file\nBUCKET_NAME=from-file\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            other_dir = Path(temp_dir) / "other"
            other_dir.mkdir()
            try:
                os.chdir(other_dir)
                with patch.dict(
                    os.environ,
                    {"DENSE_MODEL": "from-process"},
                    clear=True,
                ):
                    load_app_env(Path(temp_dir))
                    self.assertEqual(os.environ["DENSE_MODEL"], "from-process")
                    self.assertEqual(os.environ["BUCKET_NAME"], "from-file")
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
