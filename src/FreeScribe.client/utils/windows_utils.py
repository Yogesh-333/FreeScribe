"""
Windows-specific utilities for process and window management.
"""
import ctypes
import logging
import subprocess
import platform
import time
import psutil

logger = logging.getLogger(__name__)

def bring_to_front(app_name: str) -> bool:
    """
    Bring the window with the given handle to the front on Windows.
    
    Args:
        app_name (str): The name of the application window to bring to the front
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        U32DLL = ctypes.WinDLL('user32')
        SW_SHOW = 5
        hwnd = U32DLL.FindWindowW(None, app_name)
        U32DLL.ShowWindow(hwnd, SW_SHOW)
        U32DLL.SetForegroundWindow(hwnd)
        return True
    except Exception as e:
        logger.error(f"Failed to bring window to front: {e}")
        return False

def kill_with_admin_privilege(pids: list[int]) -> bool:
    """
    Attempt to kill processes with elevated administrator privileges on Windows.
    
    Args:
        pids (list[int]): List of process IDs to terminate
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not pids:
            return True
            
        pids_str = [str(pid) for pid in pids]
        logger.info(f"Killing {pids_str=} with administrator privileges")
        
        # Build the taskkill command
        taskkill_args = f'/c taskkill /F /PID {" /PID ".join(pids_str)}'
        logger.info(f"Running command: powershell Start-Process cmd -ArgumentList \"{taskkill_args}\" -Verb runAs")
        
        # Run the command with admin privileges
        proc = subprocess.run(
            [
                "powershell",
                "Start-Process",
                "cmd",
                "-ArgumentList",
                f'"{taskkill_args}"',
                "-Verb",
                "runAs"
            ],
            check=True
        )
        
        logger.info(f"Killed {pids_str=} with administrator privileges, Exit code {proc.returncode=}")
        # wait a little bit for windows to clean the proc list
        time.sleep(0.5)
        return True
    except Exception as e:
        logger.exception(f"Failed to kill processes with admin privileges: {e}")
        return False 