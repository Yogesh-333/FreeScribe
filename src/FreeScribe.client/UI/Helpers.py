import utils.system

def enable_parent_window(parent, child_window = None):
    if utils.system.is_windows():
        parent.wm_attributes('-disabled', False)
    elif utils.system.is_macos():
        if child_window:
            child_window.grab_release()
        else:
            print("Child window not provided")
    #TODO: ADD LINUX SUPPORT

def disable_parent_window(parent, child_window = None):
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