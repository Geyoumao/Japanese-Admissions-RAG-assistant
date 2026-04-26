import os
import unittest
from pathlib import Path

from app.config import load_env_file


class EnvLoaderTests(unittest.TestCase):
    def test_load_env_file_sets_missing_variables_only(self) -> None:
        original = os.environ.get("TEST_API_KEY")
        os.environ.pop("TEST_API_KEY", None)
        env_path = Path(__file__).resolve().parent / ".tmp.env"
        try:
            env_path.write_text("TEST_API_KEY=demo-key\n", encoding="utf-8")
            load_env_file(env_path)
            self.assertEqual(os.environ.get("TEST_API_KEY"), "demo-key")
        finally:
            env_path.unlink(missing_ok=True)
            if original is None:
                os.environ.pop("TEST_API_KEY", None)
            else:
                os.environ["TEST_API_KEY"] = original


if __name__ == "__main__":
    unittest.main()
