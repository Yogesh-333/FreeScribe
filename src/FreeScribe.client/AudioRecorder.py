import torch
import numpy as np
import pyaudio
import soundfile as sf
import queue
import threading
import time
from datetime import datetime
import os
from typing import Callable, Optional

class AudioRecorder:
    def __init__(self, sample_rate=16000, device=None, chunk_callback: Optional[Callable[[np.ndarray], None]] = None):
        self.chunk_size = 512
        self.format = pyaudio.paInt16
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Initialize Silero VAD
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False)
        self.model.eval()
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.device = device
        
        # Callback for real-time chunk access
        self.chunk_callback = chunk_callback
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.complete_recording_buffer = []
        self.silence_duration = 0
        self.speech_detected = False
        
        # Output settings
        self.output_dir = None
        self.base_filename = None
        self.segment_count = 0

    @staticmethod
    def get_default_device():
        """Get the default input device"""
        p = pyaudio.PyAudio()
        default_device = p.get_default_input_device_info()
        p.terminate()
        return default_device['name']
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Stream status: {status}")
            
        # Convert bytes to numpy array
        indata = np.frombuffer(in_data, dtype=np.int16).reshape(-1, 1)
        
        # Get speech probability for the chunk
        try:
            audio_normalized = indata.flatten().astype(np.float32) / np.iinfo(np.int16).max
            tensor = torch.from_numpy(audio_normalized).float()
            speech_prob = self.model(tensor, self.sample_rate).item()
        except Exception as e:
            speech_prob = 0
            print(f"Error processing VAD: {e}")

        # Put in queue for normal processing
        self.audio_queue.put((indata.copy(), speech_prob))
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self, whole_audio: bool):
        """Process audio chunks and detect speech"""
        while self.is_recording:
            try:
                audio_chunk, speech_prob = self.audio_queue.get(timeout=1)
                audio_chunk = audio_chunk.flatten()
                
                # Normalize audio to [-1, 1] range for VAD
                audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
                
                # Speech detection logic
                if speech_prob > 0.6:  # Adjust threshold as needed
                    self.silence_duration = 0
                    self.speech_detected = True
                    self.speech_buffer.append(audio_chunk)
                    self.complete_recording_buffer.append(audio_chunk.copy())
                else:
                    if self.speech_detected:
                        self.silence_duration += len(audio_chunk) / self.sample_rate
                        
                        if self.silence_duration > 0.5:  # Adjust silence threshold as needed
                            if not whole_audio:
                                self.save_speech_segment()
                            self.speech_buffer = []
                            self.speech_detected = False
                            self.silence_duration = 0
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                continue
    
    def set_chunk_callback(self, callback: Callable[[np.ndarray], None]):
        """Set the callback function to process audio chunks"""
        self.chunk_callback = callback

    def save_speech_segment(self):
        """Save the detected speech segment to a file"""
        if not self.speech_buffer:
            return
            
        try:
            speech_data = np.concatenate(self.speech_buffer)

            if self.chunk_callback:
                try:
                    print("CALLING BACK")
                    self.chunk_callback(speech_data)
                except Exception as e:
                    print(f"Error in user callback: {e}")
            
            self.segment_count += 1
        except Exception as e:
            print(f"Error saving speech segment: {e}")
    
    def save_complete_recording(self):
        """Save the complete recording to a file"""
        if not self.complete_recording_buffer:
            return
            
        try:
            complete_data = np.concatenate(self.complete_recording_buffer)
            complete_filename = os.path.join(
                self.output_dir,
                "recording.wav"
            )
            
            sf.write(complete_filename, complete_data, self.sample_rate)
            print(f"Saved complete recording to {complete_filename}")

            if self.chunk_callback:
                try:
                    self.chunk_callback(complete_data)
                except Exception as e:
                    print(f"Error in user callback: {e}")
            
        except Exception as e:
            print(f"Error saving complete recording: {e}")
    
    def start_recording(self, segments: bool):
        """Start recording audio"""
        self.output_dir = "./"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = f"recording_{timestamp}"
        
        self.is_recording = True
        self.speech_buffer = []
        self.complete_recording_buffer = []
        self.segment_count = 0
        
        try:
            device_index = None
            if self.device:
                for i in range(self.p.get_device_count()):
                    if self.p.get_device_info_by_index(i)['name'] == self.device:
                        device_index = i
                        break
            
            self.stream = self.p.open(
                format=self.format,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_index,
                stream_callback=self.audio_callback
            )
            
            self.process_thread = threading.Thread(target=self.process_audio, args=(segments,))
            self.process_thread.start()
            
            self.stream.start_stream()
            device_info = self.p.get_device_info_by_index(device_index if device_index is not None else self.p.get_default_input_device_info()['index'])
            print(f"Recording started... (using device: {device_info['name']})")
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self, segments: bool):
        """Stop recording audio"""
        print("FALSE")
        self.is_recording = False
        
        print("Closing streams")
        self.stream.stop_stream()
        self.stream.close()
        
        print("Joining process thread")
        self.process_thread.join()
        
        print("Saving final speech segment1")
        print(self.speech_buffer)
        print(segments)
        if segments:
            print("Saving final speech segment2.")
            self.save_speech_segment()
        
        print("Attempting to save complete recording.")
        if not segments:
            print("Saving complete recording.")
            self.save_complete_recording()

        print("Clearing audio queue.")
        self.audio_queue.queue.clear()
        
        print("Recording stopped.")

    def cleanup(self):
        """Clean up PyAudio resources"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.is_recording = False
        self.audio_queue.queue.clear()