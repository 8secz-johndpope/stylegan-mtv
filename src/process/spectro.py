import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base import BaseOfflineProcessor


DEFAULT_SPECTRO_PARAMS = dict(
    # n_fft=2048,
    n_fft=8192,
    hop_length=512,
    n_mels=256,
    fmin=20,
    fmax=20000
)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class SpectrogramOfflineProcessor(BaseOfflineProcessor):
    def __init__(self, model_name='cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)

    def get_spectrogram_vec(self, spectrograms, frame_num, window_size=1, displacement_factor=0.1):
        if window_size <= 1:
            return spectrograms[frame_num]
        start_idx = max(frame_num - window_size, 0)
        end_idx = max(frame_num, 1)

        frames = np.array(spectrograms[start_idx: end_idx])
        # todo: play with this weighting function
        averaged = np.average(
            frames,
            axis=0,
            # weights=np.geomspace(start=0.1, stop=50, num=len(frames))
            weights=np.geomspace(1, len(frames), num=len(frames))
            # weights = np.arange(1, len(frames))
        )
        frame = averaged.reshape((1, -1))
        return self.latent_seed + (frame * displacement_factor)

    def sound_to_mel_spectrogram(self, sound_data, sample_rate, spectrogram_params: dict):
        hop_length = sample_rate // self.fps
        # n_fft = hop_length * 4
        # # n_fft = int((self.fps / 2) * hop_length)
        n_fft = hop_length

        spectrogram_params.update(dict(
            n_fft=n_fft,
            hop_length=hop_length,
            # n_mels=self.model.input_shape
        ))
        # todo: clamp/normalize this?
        # todo: decompose percussive + harmonic spectrogram?
        mel_spectrogram = librosa.feature.melspectrogram(
            sound_data,
            sample_rate,
            **spectrogram_params
        )
        if mel_spectrogram.shape[0] != self.model.input_shape:
            n_repeats = self.model.input_shape / mel_spectrogram.shape[0]  # n_mels must be a divisor of input shape
            mel_spectrogram = np.repeat(mel_spectrogram, n_repeats, 0)

        reshaped_spec = mel_spectrogram.T
        n_frames = len(reshaped_spec)
        reshaped_spec = reshaped_spec.reshape(n_frames, 1, -1)

        return reshaped_spec

    def get_images(self, sound_data, sample_rate, spectrogram_params: dict, window_size=1, displacement_factor=0.1):
        spectrogram = self.sound_to_mel_spectrogram(sound_data, sample_rate, spectrogram_params)

        images = {}
        for i, frame in tqdm(enumerate(spectrogram), total=len(spectrogram)):
            # todo: window size based on beats?
            spectrogram_vec = self.get_spectrogram_vec(spectrogram, i, window_size, displacement_factor)

            latent_vec = spectrogram_vec  # todo: anything else to add here?

            images[i] = self.model.run_image(latent_vec, as_bytes=False)

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, input_path: str, output_path: str, start=0, duration=None, sr=None,
                     write=True, window_size=1, displacement_factor=0.1, spectrogram_params=None):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        if spectrogram_params is None:
            spectrogram_params = DEFAULT_SPECTRO_PARAMS
        sound_data, sample_rate = librosa.load(input_path, sr=sr, offset=start, duration=duration)

        self.get_images(sound_data, sample_rate, spectrogram_params, window_size, displacement_factor)

        duration = sound_data.shape[0] / sample_rate
        return self.create_video(duration, input_path, output_path, write=write, start=start)


if __name__ == '__main__':
    pass
