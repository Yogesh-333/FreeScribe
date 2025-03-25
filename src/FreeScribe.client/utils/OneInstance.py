"""
Application lock class to prevent multiple instances of an app from running.
"""
import logging
import tkinter as tk
from tkinter import messagebox
import psutil  # For process management
import os
import sys

from utils.system import is_windows, is_linux, is_macos
from utils.windows_utils import bring_to_front as windows_bring_to_front, kill_with_admin_privilege
from utils.linux_utils import check_instance as linux_check_instance, cleanup_lock as linux_cleanup_lock, bring_to_front as linux_bring_to_front
from utils.macos_utils import check_instance as macos_check_instance, cleanup_lock as macos_cleanup_lock, bring_to_front as macos_bring_to_front

logger = logging.getLogger(__name__)

class OneInstance:
    """
    Controls application instances to ensure only one is running.
    
    Args:
        app_name: Window title of the application
        app_task_manager_name: Process name as shown in Task Manager
    """
    def __init__(self, app_name, app_task_manager_name):
        self.app_name = app_name
        self.app_task_manager_name = app_task_manager_name
        self.root = None
        self.lock_file = None
        
    def get_running_instance_pids(self):
        """
        Finds PIDs of any running instances of the application, excluding the current process.
        
        Returns:
            list: PIDs of running instances, excluding the current process
        """
        current_pid = os.getpid()
        possible_ids = []
        for proc in psutil.process_iter(['pid', 'name', 'status']):
            try:
                info = proc.info
                # more criteria, they may remain in proc list in different states right after killing
                if (info['name'] == f"{self.app_task_manager_name}"
                        and info['pid'] != current_pid
                        and info['status'] != psutil.STATUS_ZOMBIE
                        and proc.is_running()):
                    possible_ids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return possible_ids
        
    def kill_instance(self, pid):
        """
        Terminates specified process instance(s).
        
        Args:
            pid: Process ID (int) or list of PIDs to terminate
        
        Returns:
            bool: True if termination successful, False otherwise
        """
        logger.info(f"Killing {pid=}")
        try:
            if type(pid) == int:
                process = psutil.Process(pid)
                process.terminate()
                return True
            elif type(pid) == list:
                for pid in pid:
                    process = psutil.Process(pid)
                    process.terminate()
                return True
        except psutil.NoSuchProcess:
            return False
        return False

    def bring_to_front(self, app_name: str):
        """
        Bring the window with the given handle to the front.
        
        Args:
            app_name (str): The name of the application window to bring to the front
            
        Returns:
            bool: True if successful, False otherwise
        """
        if is_windows():
            return windows_bring_to_front(app_name)
        elif is_linux():
            return linux_bring_to_front(app_name)
        elif is_macos():
            return macos_bring_to_front(app_name)
        return False

    def _handle_kill(self, dialog, pids):
        """Handles clicking 'Close Existing Instance' button"""
        # try killing other instance
        try:
            self.kill_instance(pids)
        except psutil.AccessDenied:
            logger.info(f"Access Denied: {pids=}")
            # try elevating privilege and kill instance again
            if is_windows():
                kill_with_admin_privilege(pids)
        # check again if they are really killed
        pids = self.get_running_instance_pids()
        logger.info(f"not killed {pids=}")
        if not pids:
            dialog.destroy()
            dialog.return_status = False
        else:
            messagebox.showerror("Error", "Failed to terminate existing instance")
            dialog.destroy()
            dialog.return_status = True
    
    def _handle_cancel(self, dialog):
        """Handles clicking 'Cancel' button"""
        dialog.destroy()
        self.bring_to_front(self.app_name)
        dialog.return_status = True

    def show_instance_dialog(self):
        """
        Shows dialog when another instance is detected.
        Allows user to close existing instance or cancel.
        
        Returns:
            bool: True if existing instance continues, False if terminated
        """
        pids = self.get_running_instance_pids()

        if not pids:
            return False

        dialog = tk.Tk()
        dialog.title("FreeScribe Instance")
        dialog.geometry("300x150")
        dialog.attributes("-topmost", True)
        dialog.lift()
        dialog.focus_force()
        
        dialog.return_status = True

        label = tk.Label(dialog, text="Another instance of FreeScribe is already running.\nWhat would you like to do?")
        label.pack(pady=20)
        
        tk.Button(dialog, text="Close Existing Instance", command=lambda: self._handle_kill(dialog, pids)).pack(padx=5, pady=5)
        tk.Button(dialog, text="Cancel", command=lambda: self._handle_cancel(dialog)).pack(padx=5, pady=2)
        
        dialog.mainloop()
        return dialog.return_status
        
    def run(self):
        """
        Main entry point to check for existing instances.
        
        Returns:
            bool: True if existing instance continues, False if none exists or terminated
        """
        # Check for running instances using platform-specific methods
        if is_windows():
            pids = self.get_running_instance_pids()
            if pids:
                return self.show_instance_dialog()
        elif is_linux():
            self.lock_file, already_running = linux_check_instance()
            if already_running:
                return self.show_instance_dialog()
        elif is_macos():
            self.lock_file, already_running = macos_check_instance()
            if already_running:
                return self.show_instance_dialog()
                
        return False
        
    def cleanup(self):
        """
        Clean up platform-specific resources.
        """
        if self.lock_file:
            if is_linux():
                linux_cleanup_lock(self.lock_file)
            elif is_macos():
                macos_cleanup_lock(self.lock_file)
            self.lock_file = None
