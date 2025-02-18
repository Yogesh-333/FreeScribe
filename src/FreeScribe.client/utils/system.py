"""
This software is released under the AGPL-3.0 license
Copyright (c) 2023-2025 Braedon Hendy

Further updates and packaging added in 2024-2025 through the ClinicianFOCUS initiative, 
a collaboration with Dr. Braedon Hendy and Conestoga College Institute of Applied 
Learning and Technology as part of the CNERG+ applied research project, 
Unburdening Primary Healthcare: An Open-Source AI Clinician Partner Platform. 
Prof. Michael Yingbull (PI), Dr. Braedon Hendy (Partner), 
and Research Students (Software Developers) - 
Alex Simko, Pemba Sherpa, Naitik Patel, Yogesh Kumar and Xun Zhong.
"""

import platform

def is_macos():
    """
    Check if the current system is macOS.
    
    :returns bool: True if macOS, False otherwise
    """
    
    return platform.system() == "Darwin"

def is_windows():
    """
    Check if the current system is Windows.
    
    :returns bool: True if Windows, False otherwise
    """
    
    return platform.system() == "Windows"

def is_linux():
    """
    Check if the current system is Linux.
    
    :returns bool: True if Linux, False otherwise
    """
    
    return platform.system() == "Linux"