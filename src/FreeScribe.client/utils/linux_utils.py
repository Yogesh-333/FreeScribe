"""
Linux-specific utilities for process and window management.
"""
import logging
import os
import fcntl
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Lock file path for Linux
LINUX_LOCK_PATH = f'/run/user/{os.getuid()}/FreeScribe.lock'

def check_instance() -> Tuple[Optional[object], bool]:
    """
    Check if another instance is running using a lock file on Linux.
    
    Returns:
        Tuple[Optional[object], bool]: A tuple containing:
            - The lock file object if successful, None otherwise
            - True if another instance is running, False otherwise
    """
    try:
        lock_file = open(LINUX_LOCK_PATH, 'w')
        fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file, False
    except IOError:
        return None, True

def cleanup_lock(lock_file: object) -> None:
    """
    Clean up the lock file on Linux.
    
    Args:
        lock_file: The lock file object to clean up
    """
    if lock_file:
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.remove(LINUX_LOCK_PATH)
        except Exception as e:
            logger.exception(f"Error cleaning up lock file: {e}")

def bring_to_front(app_name: str) -> bool:
    """
    Bring the window with the given name to the front on Linux.
    
    Args:
        app_name (str): The name of the application window to bring to the front
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # TODO: Implement Linux window activation using X11 or Wayland
        # This is a placeholder for future implementation
        logger.warning("Window activation not yet implemented for Linux")
        return False
    except Exception as e:
        logger.error(f"Failed to bring window to front: {e}")
        return False 