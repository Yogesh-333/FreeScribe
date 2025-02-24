from enum import Enum
from dataclasses import dataclass
from typing import Optional
import utils.system

@dataclass
class ModelInfo:
    """
    Dataclass to store information about a whisper models
    """
    label: str
    windows_value: str
    macos_value: str

class WhisperModels(Enum):
    """
    Enum to represent the available whisper models
    """

    TINY = ModelInfo(label="Tiny", windows_value="tiny", macos_value="openai/whisper-tiny")
    TINY_EN = ModelInfo(label="Tiny (English Only)", windows_value="tiny.en", macos_value="openai/whisper-tiny.en")
    SMALL = ModelInfo(label="Small", windows_value="small", macos_value="openai/whisper-small")
    SMALL_EN = ModelInfo(label="Small (English Only)", windows_value="small.en", macos_value="openai/whisper-small.en")
    MEDIUM = ModelInfo(label="Medium", windows_value="medium", macos_value="openai/whisper-medium")
    MEDIUM_EN = ModelInfo(label="Medium (English Only)", windows_value="medium.en", macos_value="openai/whisper-medium.en")
    BASE = ModelInfo(label="Base", windows_value="base", macos_value="openai/whisper-base")
    BASE_EN = ModelInfo(label="Base (English Only)", windows_value="base.en", macos_value="openai/whisper-base.en")
    LARGE = ModelInfo(label="Large", windows_value="large", macos_value="openai/whisper-large")
    
    @property
    def label(self) -> str:
        """
        Get the label of the model
        """
        return self.value.label
    
    @property
    def windows_value(self) -> str:
        """
        Get the value of the model for Windows
        """
        return self.value.windows_value
    
    @property
    def macos_value(self) -> str:
        """
        Get the value of the model for MacOS
        """
        return self.value.macos_value
    
    def get_platform_value(self) -> str:
        """Get the appropriate value for the specified platform"""
        if utils.system.is_windows():
            return self.windows_value
        elif utils.system.is_macos():
            return self.macos_value
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    @classmethod
    def get_all_labels(cls) -> list[str]:
        """
        Get a list of all the labels of the models
        """
        return [model.label for model in cls]
    
    @classmethod
    def find_by_label(cls, label: str) -> Optional["WhisperModels"]:
        """
        Find a model by its label
        """

        for model in cls:
            if model.label == label:
                return model
        return None

    def __str__(self) -> str:
        """
        Get the string representation of the model
        """
        return self.label