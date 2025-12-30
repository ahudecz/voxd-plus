#!/usr/bin/env python3
"""
Voxd Overlay Integration & Standalone Trigger Mode
===================================================

This module provides:
1. Integration patterns for adding the overlay to voxd
2. A standalone trigger mode that mimics voxd's --trigger-record behavior

Usage (standalone):
    # Start the overlay daemon
    python trigger_integration.py --daemon
    
    # Trigger record start/stop (from hotkey)
    python trigger_integration.py --trigger

Usage (integration with voxd):
    See the VoxdOverlayIntegration class for integration patterns.
"""

import os
import sys
import json
import socket
import argparse
import tempfile
import threading
from pathlib import Path
from typing import Optional, Callable

import numpy as np

# Import our overlay
from recording_overlay import RecordingOverlay, RecordingOverlayManager, AudioCue

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
except ImportError:
    print("PyQt6 required: pip install PyQt6")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice required: pip install sounddevice")
    sys.exit(1)


# Socket-based IPC for trigger communication
SOCKET_PATH = Path(tempfile.gettempdir()) / "voxd_overlay.sock"
STATE_FILE = Path(tempfile.gettempdir()) / "voxd_overlay_state.json"


class TriggerServer:
    """
    Unix socket server that listens for trigger commands.
    This allows the overlay daemon to receive hotkey triggers.
    """
    
    def __init__(self, overlay: RecordingOverlay, 
                 on_recording_done: Optional[Callable[[np.ndarray], None]] = None):
        self.overlay = overlay
        self.on_recording_done = on_recording_done
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the trigger server"""
        # Clean up old socket
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(str(SOCKET_PATH))
        self.socket.settimeout(0.5)  # Allow periodic checks
        
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        
        # Save state file
        self._update_state(running=True, recording=False)
        
        print(f"Trigger server listening on: {SOCKET_PATH}")
    
    def stop(self):
        """Stop the trigger server"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.running:
            try:
                data, _ = self.socket.recvfrom(1024)
                command = data.decode('utf-8').strip()
                self._handle_command(command)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Socket error: {e}")
    
    def _handle_command(self, command: str):
        """Handle incoming command"""
        if command == "toggle":
            # Must dispatch to main Qt thread
            QTimer.singleShot(0, self._do_toggle)
        elif command == "start":
            QTimer.singleShot(0, self.overlay.start_recording)
        elif command == "stop":
            QTimer.singleShot(0, self.overlay.stop_recording)
        elif command == "status":
            # Just update state file
            self._update_state(recording=self.overlay.is_recording())
        elif command == "quit":
            QTimer.singleShot(0, QApplication.quit)
    
    def _do_toggle(self):
        """Toggle recording on main thread"""
        self.overlay.toggle_recording()
        self._update_state(recording=self.overlay.is_recording())
    
    def _update_state(self, running: bool = True, recording: bool = False):
        """Update state file for external queries"""
        state = {
            "running": running,
            "recording": recording,
            "pid": os.getpid()
        }
        STATE_FILE.write_text(json.dumps(state))


def send_trigger_command(command: str = "toggle") -> bool:
    """Send a command to the running overlay daemon"""
    if not SOCKET_PATH.exists():
        return False
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(command.encode('utf-8'), str(SOCKET_PATH))
        sock.close()
        return True
    except Exception as e:
        print(f"Failed to send command: {e}")
        return False


def get_daemon_status() -> dict:
    """Get the current daemon status"""
    if not STATE_FILE.exists():
        return {"running": False}
    
    try:
        return json.loads(STATE_FILE.read_text())
    except:
        return {"running": False}


def run_daemon(sample_rate: int = 16000, 
               audio_cues: bool = True,
               on_complete: Optional[Callable[[np.ndarray], None]] = None):
    """
    Run the overlay as a daemon that listens for trigger commands.
    
    This is meant to be started once and left running.
    Use send_trigger_command("toggle") from your hotkey.
    """
    app = QApplication(sys.argv)
    
    # Default handler just prints info
    def default_handler(audio: np.ndarray):
        duration = len(audio) / sample_rate
        print(f"\n✓ Recording: {duration:.2f}s ({len(audio)} samples)")
        
        # Save to temp file for demo
        temp_wav = Path(tempfile.gettempdir()) / "voxd_overlay_last.wav"
        try:
            import wave
            with wave.open(str(temp_wav), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                # Convert float32 to int16
                audio_int = (audio * 32767).astype(np.int16)
                wf.writeframes(audio_int.tobytes())
            print(f"  Saved to: {temp_wav}")
        except Exception as e:
            print(f"  (Could not save: {e})")
    
    handler = on_complete or default_handler
    
    # Create overlay
    overlay = RecordingOverlay(
        sample_rate=sample_rate,
        play_audio_cues=audio_cues
    )
    overlay.recording_complete.connect(handler)
    
    # Position
    screen = app.primaryScreen().geometry()
    overlay.move(
        screen.width() - overlay.width() - 20,
        screen.height() - overlay.height() - 80
    )
    
    # Start trigger server
    server = TriggerServer(overlay, handler)
    server.start()
    
    # Show overlay
    overlay.show()
    
    print("=" * 50)
    print("Recording Overlay Daemon")
    print("=" * 50)
    print(f"\nTrigger with: python {__file__} --trigger")
    print("Or configure your hotkey to run that command.")
    print("\nPress Ctrl+C to quit.\n")
    
    try:
        app.exec()
    finally:
        server.stop()


class VoxdOverlayIntegration:
    """
    Integration patterns for adding the overlay to voxd.
    
    This class provides examples of how to modify voxd's recorder.py
    to use the overlay for visual feedback.
    
    Integration approaches:
    
    1. SUBPROCESS MODE (easiest, no voxd changes):
       - Run the overlay daemon separately
       - Have voxd send trigger commands via socket
    
    2. EMBEDDED MODE (cleaner, requires voxd modifications):
       - Import and instantiate the overlay in voxd's GUI/tray code
       - Call overlay methods directly when recording starts/stops
    """
    
    @staticmethod
    def subprocess_integration_example():
        """
        Example of integrating via subprocess/socket.
        
        Add this to voxd's trigger_record() function:
        """
        code = '''
# In voxd's recorder.py or main.py, modify trigger_record():

import subprocess
import socket
from pathlib import Path
import tempfile

OVERLAY_SOCKET = Path(tempfile.gettempdir()) / "voxd_overlay.sock"

def trigger_with_overlay():
    """Enhanced trigger that also updates the overlay"""
    # Send toggle to overlay if running
    if OVERLAY_SOCKET.exists():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.sendto(b"toggle", str(OVERLAY_SOCKET))
            sock.close()
        except:
            pass
    
    # Continue with normal voxd recording...
    # (existing trigger_record code)
'''
        return code
    
    @staticmethod  
    def embedded_integration_example():
        """
        Example of embedding the overlay directly in voxd.
        
        Modify voxd's GUI class to include the overlay:
        """
        code = '''
# In voxd's gui.py, add to the main window class:

from recording_overlay import RecordingOverlay

class VoxdGui(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... existing init code ...
        
        # Add the recording overlay
        self.recording_overlay = RecordingOverlay(
            sample_rate=16000,  # Match your whisper config
            play_audio_cues=True
        )
        
        # Position it (or let user drag)
        screen = QApplication.primaryScreen().geometry()
        self.recording_overlay.move(
            screen.width() - self.recording_overlay.width() - 20,
            screen.height() - self.recording_overlay.height() - 80
        )
        
        # Connect to your transcription pipeline
        self.recording_overlay.recording_complete.connect(
            self.on_overlay_recording_done
        )
    
    def show_overlay(self):
        """Show the overlay (call on app start or from menu)"""
        self.recording_overlay.show()
    
    def on_hotkey_pressed(self):
        """Modified hotkey handler"""
        # Toggle overlay recording instead of/in addition to internal
        self.recording_overlay.toggle_recording()
    
    def on_overlay_recording_done(self, audio_data):
        """Handle completed recording from overlay"""
        # Pass to whisper transcription
        self.transcribe_audio(audio_data)
'''
        return code
    
    @staticmethod
    def audio_cue_only_integration():
        """
        Minimal integration: just add audio cues to existing voxd.
        
        If you only want audio cues without the visual overlay:
        """
        code = '''
# Add to voxd's recorder.py:

import numpy as np
import sounddevice as sd
import threading

class AudioCue:
    SAMPLE_RATE = 44100
    
    @staticmethod
    def _tone(freq, dur, vol=0.25):
        t = np.linspace(0, dur, int(44100*dur), dtype=np.float32)
        tone = np.sin(2 * np.pi * freq * t) * vol
        # Fade
        fade = int(44100 * 0.01)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone
    
    @staticmethod
    def start():
        t1 = AudioCue._tone(880, 0.08)
        t2 = AudioCue._tone(1320, 0.12)
        gap = np.zeros(int(44100 * 0.02), dtype=np.float32)
        threading.Thread(
            target=lambda: sd.play(np.concatenate([t1, gap, t2]), 44100),
            daemon=True
        ).start()
    
    @staticmethod
    def stop():
        t1 = AudioCue._tone(1320, 0.08)
        t2 = AudioCue._tone(880, 0.12)
        gap = np.zeros(int(44100 * 0.02), dtype=np.float32)
        threading.Thread(
            target=lambda: sd.play(np.concatenate([t1, gap, t2]), 44100),
            daemon=True
        ).start()

# Then in your recording functions:
def start_recording(self):
    AudioCue.start()
    time.sleep(0.15)  # Wait for cue to play
    # ... existing start code ...

def stop_recording(self):
    # ... existing stop code ...
    AudioCue.stop()
'''
        return code


def main():
    parser = argparse.ArgumentParser(
        description="Recording Overlay - Standalone trigger mode"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as daemon (shows overlay, listens for triggers)"
    )
    parser.add_argument(
        "--trigger", action="store_true",
        help="Send toggle trigger to running daemon"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Get daemon status"
    )
    parser.add_argument(
        "--quit", action="store_true",
        help="Tell daemon to quit"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--no-audio-cues", action="store_true",
        help="Disable audio cues"
    )
    parser.add_argument(
        "--show-integration", action="store_true",
        help="Print voxd integration examples"
    )
    
    args = parser.parse_args()
    
    if args.show_integration:
        print("\n" + "=" * 60)
        print("VOXD INTEGRATION EXAMPLES")
        print("=" * 60)
        
        print("\n--- SUBPROCESS MODE ---")
        print(VoxdOverlayIntegration.subprocess_integration_example())
        
        print("\n--- EMBEDDED MODE ---")
        print(VoxdOverlayIntegration.embedded_integration_example())
        
        print("\n--- AUDIO CUES ONLY ---")
        print(VoxdOverlayIntegration.audio_cue_only_integration())
        return
    
    if args.status:
        status = get_daemon_status()
        if status.get("running"):
            rec = "recording" if status.get("recording") else "idle"
            print(f"Daemon running (PID {status.get('pid')}), {rec}")
        else:
            print("Daemon not running")
        return
    
    if args.trigger:
        if send_trigger_command("toggle"):
            print("Toggle sent")
        else:
            print("Daemon not running. Start with: --daemon")
        return
    
    if args.quit:
        if send_trigger_command("quit"):
            print("Quit sent")
        else:
            print("Daemon not running")
        return
    
    if args.daemon:
        run_daemon(
            sample_rate=args.sample_rate,
            audio_cues=not args.no_audio_cues
        )
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
