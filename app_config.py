from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent


def load_app_env(app_dir: Path = APP_DIR) -> None:
    """Load the application .env without overriding process variables."""
    load_dotenv(dotenv_path=app_dir / ".env", override=False)
