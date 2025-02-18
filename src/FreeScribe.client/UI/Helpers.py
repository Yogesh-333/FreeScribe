"""
This software is released under the AGPL-3.0 license
Copyright (c) 2023-2025 Braedon Hendy

Further updates and packaging added in 2024-2025 through the ClinicianFOCUS initiative, 
a collaboration with Dr. Braedon Hendy and Conestoga College Institute of Applied 
Learning and Technology as part of the CNERG+ applied research project, 
Unburdening Primary Healthcare: An Open-Source AI Clinician Partner Platform". 
Prof. Michael Yingbull (PI), Dr. Braedon Hendy (Partner), 
and Research Students (Software Developers) - 
Alex Simko, Pemba Sherpa, Naitik Patel, Yogesh Kumar and Xun Zhong.
"""

import utils.system

def enable_parent_window(parent, child_window = None):
    """
    Enable the parent window after a child window has been closed.
    
    :param parent: The parent window to enable
    :type parent: tk.Tk or tk.Toplevel
    :param child_window: The child window that was closed
    :type child_window: tk.Toplevel
    """
    if utils.system.is_windows():
        parent.wm_attributes('-disabled', False)
    elif utils.system.is_macos():
        if child_window:
            child_window.grab_release()
        else:
            print("Child window not provided")
    #TODO: ADD LINUX SUPPORT

def disable_parent_window(parent, child_window = None):
    """
    Disable the parent window when a child window is opened.
    
    :param parent: The parent window to disable
    :type parent: tk.Tk or tk.Toplevel
    :param child_window: The child window that is opened
    :type child_window: tk.Toplevel
    """
    if utils.system.is_windows():
        parent.wm_attributes('-disabled', True)
    elif utils.system.is_macos():
        if child_window:
            child_window.transient(parent)
            child_window.grab_set()
            child_window.attributes('-topmost', True)
        else:
            print("Child window not provided")
    #TODO: ADD LINUX SUPPORT