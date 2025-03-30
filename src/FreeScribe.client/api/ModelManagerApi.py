import logging
import threading
import os
from llama_cpp import Llama
from typing import Optional, Dict, Any
from enum import Enum

class ModelStatus(Enum):
    """
    Enum class for model loading status.
    """
    ERROR = 1

class Model:
    """
    Model class for handling GPU-accelerated text generation using the Llama library.
    """
    def __init__(
        self,
        model_path: str,
        chat_template: str = None,
        context_size: int = 1024,
        gpu_layers: int = -1,
        main_gpu: int = 0,
        tensor_split: Optional[list] = None,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        seed: int = 1337
    ):
        """
        Initializes the GGUF model with GPU acceleration.
        """
        try:
            # Set environment variables for GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
            
            # Initialize model with GPU settings
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_gpu_layers=gpu_layers,
                n_batch=n_batch,
                n_threads=n_threads or os.cpu_count(),
                seed=seed,
                tensor_split=tensor_split,
                chat_format=chat_template,
            )
            
            # Store configuration
            self.config = {
                "gpu_layers": gpu_layers,
                "main_gpu": main_gpu,
                "context_size": context_size,
                "n_batch": n_batch
            }
            logging.info("Model initialized successfully.")
        except Exception as e:
            self.model = None
            logging.error(f"Failed to initialize model: {e}")
            raise e

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1
    ) -> str:
        """
        Generates a response using GPU-accelerated inference.
        """
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.model.create_chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"GPU inference error: {e}")
            return f"Error: {str(e)}"

    def close(self):
        """
        Unloads the model from GPU memory.
        """
        if self.model:
            self.model.close()
            self.model = None

    def __del__(self):
        """Cleanup GPU memory on deletion"""
        if self.model is not None:
            self.model.close()
        self.model = None

class ModelManagerAPI:
    """
    Manages the lifecycle of a local LLM model including setup and unloading operations.
    """
    local_model = None

    @staticmethod
    def setup_model(
        model_path: str,
        chat_template: str = None,
        context_size: int = 4096,
        gpu_layers: int = -1,
        main_gpu: int = 0,
        tensor_split: Optional[list] = None,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        seed: int = 1337
    ):
        """
        Initialize and load the LLM model based on provided settings.
        """
        if ModelManagerAPI.local_model is not None:
            ModelManagerAPI.unload_model()

        try:
            logging.info("Loading model...")
            ModelManagerAPI.local_model = Model(
                model_path=model_path,
                chat_template=chat_template,
                context_size=context_size,
                gpu_layers=gpu_layers,
                main_gpu=main_gpu,
                tensor_split=tensor_split,
                n_batch=n_batch,
                n_threads=n_threads,
                seed=seed
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            ModelManagerAPI.local_model = ModelStatus.ERROR

    @staticmethod
    def generate_response_api(prompt: str, max_tokens: int = 128, temperature: float = 0.1, top_p: float = 0.95, repeat_penalty: float = 1.1) -> str:
        """
        Generates a response using the loaded model.
        """
        if ModelManagerAPI.local_model is None:
            logging.error("Model is not loaded. Please load the model before generating responses.")
            return "Model is not loaded. Please load the model before generating responses."

        try:
            response = ModelManagerAPI.local_model.generate_response(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty
            )
            return response
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    @staticmethod
    def unload_model():
        """
        Safely unload and cleanup the currently loaded model.
        """
        if ModelManagerAPI.local_model is not None:
            ModelManagerAPI.local_model.close()
            ModelManagerAPI.local_model = None
        logging.debug(f"{ModelManagerAPI.local_model=}")