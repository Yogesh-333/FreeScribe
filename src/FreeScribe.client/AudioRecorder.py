import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import threading
import time
from datetime import datetime
import os
from typing import Callable, Optional

class AudioRecorder:
    def __init__(self, sample_rate=16000, device=None, chunk_callback: Optional[Callable[[np.ndarray], None]] = None):
        # Silero
        # 16000 Hz = 512
        # 8000 Hz = 256
        self.chunk_size = 512
        
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
        try:
            return sd.query_devices(kind='input')['name']
        except:
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(f"Stream status: {status}")
            
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
                    
                    # Store the chunk in complete recording buffer
                    self.complete_recording_buffer.append(audio_chunk.copy())
                else:
                    if self.speech_detected:
                        self.silence_duration += len(audio_chunk) / self.sample_rate
                        
                        # If silence duration exceeds threshold, save the buffer
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
            # Combine all chunks in the buffer
            speech_data = np.concatenate(self.speech_buffer)

            # Call user callback if provided
            if self.chunk_callback:
                try:
                    self.chunk_callback(speech_data)
                except Exception as e:
                    print(f"Error in user callback: {e}")
            
            # Generate filename for segment
            self.segment_count += 1
            
        except Exception as e:
            print(f"Error saving speech segment: {e}")
    
    def save_complete_recording(self):
        """Save the complete recording to a file"""
        if not self.complete_recording_buffer:
            return
            
        try:
            # Combine all chunks
            complete_data = np.concatenate(self.complete_recording_buffer)

            # Generate filename for complete recording
            complete_filename = os.path.join(
                self.output_dir,
                "recording.wav"
            )
            
            # Save the complete recording
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
        # Create output directory if it doesn't exist
        self.output_dir = "./"
        
        # Generate base filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = f"recording_{timestamp}"
        
        # Initialize recording state
        self.is_recording = True
        self.speech_buffer = []
        self.complete_recording_buffer = []
        self.segment_count = 0
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.int16,
                callback=self.audio_callback
            )
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_audio, args=(segments,))
            self.process_thread.start()
            
            self.stream.start()
            device_info = sd.query_devices(self.device) if self.device is not None else sd.query_devices(kind='input')
            print(f"Recording started... (using device: {device_info['name']})")
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self, segments: bool):
        """Stop recording audio"""
        self.is_recording = False
        
        # Stop and close the audio stream
        self.stream.stop()
        self.stream.close()
        
        # Wait for processing thread to finish
        self.process_thread.join()
        
        # Save any remaining speech in the buffer
        if self.speech_buffer and segments:
            self.save_speech_segment()
        
        # Save the complete recording
        if not segments:
            self.save_complete_recording()

        self.audio_queue.queue.clear()
        
        print("Recording stopped.")

    def cleanup(self):
        """
        Stops and cleans up the audio recording stream.
        
        Stops the stream, closes it, sets recording flag to False, and clears the audio queue.
        """
        self.stream.stop()
        self.stream.close()
        self.is_recording = False
        self.audio_queue.queue.clear()