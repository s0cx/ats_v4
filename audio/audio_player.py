# audio/audio_player.py
import numpy as np
import simpleaudio as sa
import threading


class AudioPlayer:
    """
    Audio playback wrapper supporting:
      - blocking or async playback
      - waterfall callback for spectrogram
    """

    def __init__(self, waterfall_callback=None):
        self.wave_obj = None
        self.play_obj = None
        self.waterfall_callback = waterfall_callback
        self._stop = False
        self.chunk_size = 4096

    def stop(self):
        self._stop = True
        if self.play_obj:
            try:
                self.play_obj.stop()
            except:
                pass

    def play(self, audio: np.ndarray, sample_rate: int):
        """
        Plays audio array using simpleaudio.
        Streams chunks to waterfall_callback for spectrogram.
        """
        self._stop = False

        # ensure int16 for simpleaudio
        audio16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        # simpleaudio requires interleaved stereo
        if audio16.ndim == 2:
            wav_bytes = audio16.tobytes()
        else:
            wav_bytes = audio16.tobytes()

        self.wave_obj = sa.WaveObject(wav_bytes, 
                                      num_channels=audio16.shape[1] if audio16.ndim == 2 else 1,
                                      bytes_per_sample=2,
                                      sample_rate=sample_rate)

        # play asynchronously
        self.play_obj = self.wave_obj.play()

        # thread to feed spectrogram
        def stream_thread():
            pos = 0
            total = len(audio16)
            while pos < total and not self._stop:
                block = audio16[pos:pos+self.chunk_size]

                # feed block to waterfall
                if self.waterfall_callback:
                    self.waterfall_callback(block)

                pos += self.chunk_size
                sa.sleep(0.01)

        threading.Thread(target=stream_thread, daemon=True).start()
