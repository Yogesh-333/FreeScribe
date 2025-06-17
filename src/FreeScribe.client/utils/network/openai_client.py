import asyncio
import json
import httpx
from typing import Dict, Any, Optional, List
from .base import BaseNetworkClient, NetworkConfig, NetworkRequestError
from utils.log_config import logger
import threading
import tkinter as tk
import time


class OpenAIClient(BaseNetworkClient):
    """Client for communicating with OpenAI API."""
    
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self.cancel_monitor_thread = None
        self.monitoring_stop_event = threading.Event()
        self.stop_event = asyncio.Event()
    
    def send_chat_completion_sync(
        self, 
        text: str, 
        model: str, 
        threading_cancel_event: threading.Event,
        system_message: Optional[str] = None,
        **options
    ) -> str:
        """
        Synchronous wrapper for send_chat_completion that manages event loop.
        
        Args:
            text (str): The text to send to the API
            model (str): The model name to use
            threading_cancel_event (threading.Event): Event to signal cancellation
            system_message (str, optional): System message to set context
            **options: Additional model options
            
        Returns:
            str: The response text or error message
        """
        try:
            return asyncio.run(
                self.send_chat_completion(
                    text=text,
                    model=model,
                    stop_event=self.stop_event,
                    threading_cancel_event=threading_cancel_event,
                    system_message=system_message,
                    **options
                )
            )
        except Exception as e:
            logger.exception(f"Error running async function: {e}")
            return f'Error: {str(e)}'
    
    async def send_chat_completion(
        self, 
        text: str, 
        model: str, 
        stop_event: asyncio.Event,
        threading_cancel_event: threading.Event,
        system_message: Optional[str] = None,
        **options
    ) -> str:
        """
        Sends text to OpenAI API using httpx AsyncClient in a cancellable async format.
        
        Args:
            text (str): The text to send to the API
            model (str): The model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            stop_event (asyncio.Event): Event to signal cancellation
            threading_cancel_event (threading.Event): Threading event for cancellation
            system_message (str, optional): System message to set context
            **options: Additional model options (temperature, max_tokens, etc.)
            
        Returns:
            str: The response text or 'Error' if cancelled/failed
        """
        self.threading_cancel_event = threading_cancel_event

        self.start_cancel_monitoring()
        try:
            # Check for cancellation before starting
            if stop_event.is_set():
                return 'Error: Operation cancelled'
            
            # Create httpx client
            self._client = await self._create_client()
            
            # Check for cancellation after client creation
            if stop_event.is_set():
                return 'Error: Operation cancelled'

            # Prepare payload for OpenAI API
            payload = self._build_payload(text, model, system_message, **options)

            # Check for cancellation before sending request
            if stop_event.is_set():
                return 'Error: Operation cancelled'

            # Send POST request to OpenAI chat endpoint
            response_data = await self._send("chat/completions", payload)

            # Check for cancellation after response
            if stop_event.is_set():
                return 'Error: Operation cancelled'
            
            self.stop_cancel_monitoring()
            
            # Extract and return response text
            return self._parse_response_data(response_data)

        except Exception as e:
            error_message = self._handle_error(e)
            logger.exception(f"Error in OpenAI API call: {e}")
            return f'Error: {error_message}'
        
        finally:
            await self._close_client()
    
    async def send_completion(
        self, 
        prompt: str, 
        model: str, 
        stop_event: asyncio.Event,
        **options
    ) -> str:
        """
        Sends text to OpenAI Completion API (for older models like text-davinci-003).
        
        Args:
            prompt (str): The prompt to send to the API
            model (str): The model name to use
            stop_event (asyncio.Event): Event to signal cancellation
            **options: Additional model options
            
        Returns:
            str: The response text or 'Error' if cancelled/failed
        """
        try:
            if stop_event.is_set():
                return 'Error: Operation cancelled'
            
            self._client = await self._create_client()
            
            if stop_event.is_set():
                return 'Error: Operation cancelled'

            payload = self._build_completion_payload(prompt, model, **options)

            if stop_event.is_set():
                return 'Error: Operation cancelled'

            response = await self._send("completions",payload)
            
            if stop_event.is_set():
                return 'Error: Operation cancelled'
            
            return self._parse_completion_response(response)

        except Exception as e:
            error_message = self._handle_error(e)
            logger.exception(f"Error in OpenAI Completion API call: {e}")
            return f'Error: {error_message}'
        
        finally:
            await self._close_client()
    
    def _apply_options(self, payload: Dict[str, Any], options: Dict[str, Any]) -> None:
        # Set default values
        payload.update({
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        })
        
        # Handle stop parameter separately as it can be string or list
        if "stop" in options:
            payload["stop"] = options['stop']  # Can be string or list

        casts = {
            "temperature": float,
            "max_tokens": int,
            "top_p": float,
            "frequency_penalty": float,
            "presence_penalty": float,
            "stream": bool,
        }
        for key, cast in casts.items():
            if (v := options.get(key)) is not None:
                try:
                    payload[key] = cast(v)
                except ValueError:
                    logger.warning(f"Invalid {key}={v}, skipping")

    def _build_payload(
        self, 
        text: str, 
        model: str, 
        system_message: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Build the request payload for OpenAI Chat API."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": text})
        
        payload = {
            "model": model.strip(),
            "messages": messages
        }
        
        # Apply additional options to the payload
        self._apply_options(payload, options)
        
        return payload
    
    def _build_completion_payload(self, prompt: str, model: str, **options) -> Dict[str, Any]:
        """Build the request payload for OpenAI Completion API."""
        payload = {
            "model": model.strip(),
            "prompt": prompt
        }

        self._apply_options(payload, options)
        
        return payload
    
    async def _send(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Unified method to send HTTP requests to OpenAI API endpoints."""
        url = f"{self.config.host}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def _parse_response_data(self, response_data: Dict[str, Any]) -> str:
        """Parse the response data from OpenAI Chat API."""
        try:
            return response_data['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenAI response structure: {e}")
            return 'Error: Invalid response format from OpenAI API'

    def _parse_completion_response_data(self, response_data: Dict[str, Any]) -> str:
        """Parse the response data from OpenAI Completion API."""
        try:
            return response_data['choices'][0]['text'].strip()
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing OpenAI completion response structure: {e}")
            return 'Error: Invalid response format from OpenAI Completion API'
    
    def _handle_error(self, error: Exception) -> str:
        """Handle and categorize different types of errors specific to OpenAI."""
        if isinstance(error, httpx.ReadError):
            return "Network connection lost to OpenAI API. Please check your internet connection."
        elif isinstance(error, httpx.ConnectError):
            return "Cannot connect to OpenAI API. Please check your internet connection."
        elif isinstance(error, httpx.TimeoutException):
            return "Request timed out to OpenAI API. Please try again."
        elif isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code == 401:
                return "Invalid OpenAI API key. Please check your API key."
            elif status_code == 429:
                return "OpenAI API rate limit exceeded. Please wait and try again."
            elif status_code == 400:
                return f"Bad request to OpenAI API: {error.response.text}"
            elif status_code == 404:
                return "OpenAI API endpoint not found. Please check the model name."
            else:
                return f"OpenAI API error {status_code}: {error.response.text}"
        else:
            return f"OpenAI API request failed: {str(error)}"
    
    async def cancel_request(self):
        """Cancel an ongoing API request by setting the stop event and closing the client."""
        try:
            logger.info("Cancelling OpenAI API request...")
            await self._close_client()
            logger.info("OpenAI API request cancelled successfully.")
        except Exception as e:
            logger.exception(f"Error during OpenAI API cancellation: {e}")

    def start_cancel_monitoring(self):
        """Start the cancellation monitoring loop in a separate thread."""
        if self.threading_cancel_event:
            self.checking_active = True
            self.monitoring_stop_event.clear()
            self.stop_event.clear()
            self.cancel_monitor_thread = threading.Thread(
                target=self._monitor_cancellation, 
                daemon=True
            )
            self.cancel_monitor_thread.start()
    
    def _monitor_cancellation(self):
        """Monitor cancellation event in a separate thread."""
        while self.checking_active and not self.monitoring_stop_event.is_set():
            try:
                if self.threading_cancel_event is None:
                    self.f
                    logger.info("No cancellation event provided. Continuing with the request.")
                    break

                if self.threading_cancel_event.is_set():
                    logger.info("Cancellation event detected.")
                    # Schedule async close in a new thread
                    def close_client():
                        try:
                            asyncio.run(self.cancel_request())
                        except Exception as e:
                            logger.exception(f"Error closing client: {e}")
                    
                    threading.Thread(target=close_client, daemon=True).start()
                    self.checking_active = False  # Stop checking
                    break
                
                if hasattr(self, '_client') and self._client and self._client.is_closed:
                    logger.info("LLM client is closed. Stopping the request.")
                    self.checking_active = False
                    break
                
                # Wait for 0.1 seconds before checking again
                time.sleep(0.1)
                
            except Exception as e:
                logger.exception(f"Error in cancellation monitoring: {e}")
                self.checking_active = False
                break
    
    def stop_cancel_monitoring(self):
        """Stop the cancellation monitoring."""
        logger.info("Stopping OpenAI API cancellation monitoring.")
        self.checking_active = False
        self.monitoring_stop_event.set()
        self.stop_event.set()
        
        # Wait for the monitoring thread to finish
        if self.cancel_monitor_thread and self.cancel_monitor_thread.is_alive():
            self.cancel_monitor_thread.join(timeout=1.0)