import os
import speech_recognition as sr
from faster_whisper import WhisperModel
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from dhi.ui import console

# Suppress ALSA C-level errors that bypass normal Python stderr.
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass  # Ignore ALSA complaints.
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except OSError:
    pass  # Ignore if libasound is not found.

class Ear:
    def __init__(self, model_size="distil-small.en", device="cpu", compute_type="int8"):
        """Initialize the hearing system with dynamic silence detection."""
        console.print(f"[info]ℹ Loading Whisper Model ({model_size})...[/info]")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.recognizer = sr.Recognizer()
            
            # Configure dynamic silence settings.
            self.recognizer.energy_threshold = 300 
            self.recognizer.dynamic_energy_threshold = True
            # Stop recording after 2.5 seconds of silence.
            self.recognizer.pause_threshold =  2.5
            
            console.print(f"[success]✓ Model loaded successfully.[/success]")
        except Exception as e:
            console.print(f"[error]⨯ CRITICAL ERROR loading model: {e}[/error]")
            raise e

    def listen_and_transcribe(self, filename="/tmp/pragma_voice.wav") -> str:
        """Listen dynamically until speech stops, then transcribe."""
        
        with sr.Microphone() as source:
            console.print(f"[info]ℹ Adjusting for ambient noise...[/info]")
            # Calibrate against background noise for half a second.
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            try:
                with console.status("[bold cyan]Listening... (Auto-stops when you pause)[/bold cyan]", spinner="dots"):
                    # Record until silence is detected.
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)

                # Save audio to /tmp RAM disk for fast I/O.
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())

                console.print(f"[system]✦ Transcribing...[/system]")
                segments, info = self.model.transcribe(
                    filename, 
                    beam_size=5, 
                    language="en", 
                    vad_filter=True, 
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                full_text = " ".join([segment.text for segment in segments]).strip()

                # Clean up temporary file.
                if os.path.exists(filename):
                    os.remove(filename)

                return full_text

            except sr.WaitTimeoutError:
                console.print(f"[warning]⚠ No speech detected.[/warning]")
                return ""
            except Exception as e:
                console.print(f"[error]⨯ Error: {e}[/error]")
                return ""

# Unit Tests
if __name__ == "__main__":
    try:
        ear = Ear(model_size="base.en")
        console.print("Prepare to speak after hitting ENTER...")
        input()
        
        text = ear.listen_and_transcribe()
        
        console.print(f"\n[success]You said: {text}[/success]")
            
    except KeyboardInterrupt:
        console.print("\n[warning][EAR] Stopped by user.[/warning]")
