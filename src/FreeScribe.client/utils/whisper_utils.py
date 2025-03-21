# utils/whisper_utils.py
import threading
import time
from urllib.parse import urlparse
import requests
from UI.LoadingWindow import LoadingWindow
from UI.Widgets.PopupBox import PopupBox


def validate_whisper_endpoint(settings, parent_window, endpoint_url, verify_ssl, api_key):
    """
    Validates connectivity to the Whisper speech-to-text endpoint and prompts the user if validation fails.
    
    Shows a loading window during the validation process and displays a prompt 
    dialog if the connection fails, allowing the user to proceed or cancel.

    Args:
        settings: The settings object containing application configuration.
        parent_window: The parent window for displaying UI components.
        endpoint_url (str): The URL of the Whisper endpoint to validate.
        verify_ssl (bool): Whether to verify SSL certificates.
        api_key (str): The API key for authorization.

    Returns:
        bool: True if the endpoint is reachable or user chooses to proceed anyway,
            False if the endpoint is not reachable and user cancels.
    """
    # Create a result container to share data between threads
    result_container = {"result": False, "done": False}
    
    # Define the worker function to run in a separate thread
    def worker_thread():
        try:
            parsed_url = urlparse(endpoint_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                result_container["result"] = False
                result_container["done"] = True
                return

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            try:
                # Send empty request to check endpoint availability
                # This should return a 400 or 422 error for missing file
                response = requests.post(endpoint_url, headers=headers, json={}, verify=verify_ssl)

                # Check for expected response codes
                # 400/422 indicate server is up but rejecting empty request (expected behavior)
                if response.status_code in [400, 422]:
                    json_response = response.json()
                    result_container["result"] = "detail" in json_response
                else:
                    result_container["result"] = response.status_code < 400

            except requests.exceptions.RequestException:
                result_container["result"] = False

        except Exception:
            result_container["result"] = False
        finally:
            result_container["done"] = True
    
    try:
        # Create loading window
        loading = LoadingWindow(
            parent=parent_window,
            title="Checking Connection",
            initial_text="Checking Whisper endpoint connection...",
            note_text="This may take a few seconds."
        )
        
        # Start the worker thread
        thread = threading.Thread(target=worker_thread)
        thread.daemon = True
        thread.start()
        
        # Keep updating the UI while the thread is working
        while not result_container["done"]:
            if parent_window and hasattr(parent_window, 'update'):
                parent_window.update()
            time.sleep(0.1)  # Small delay to prevent UI freezing
        
        # Close the loading window
        loading.destroy()
        
        # If endpoint is not reachable, prompt the user
        if not result_container["result"]:
            # Show error popup
            popup = PopupBox(
                parent=parent_window,
                message=f"Unable to connect to the Whisper endpoint at: {endpoint_url}.\nWould you like to proceed with saving anyway?",
                title="Connection Error",
                button_text_1="Continue",
                button_text_2="Cancel"
            )
            return popup.response == "button_1"  # Return True if Continue, False if Cancel
        
        # Endpoint is reachable
        return True
    
    except Exception:
        # Return True on exception to allow the process to continue
        return True