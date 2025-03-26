from utils.file_utils import get_file_path
import logging

logger = logging.getLogger(__name__)


def get_application_version():
        version_str = "vx.x.x.alpha"
        try:
            with open(get_file_path('__version__'), 'r') as file:
                version_str = file.read().strip()
        except Exception as e:
            print(f"Error loading version file ({type(e).__name__}). {e}")
        finally:
            return version_str
