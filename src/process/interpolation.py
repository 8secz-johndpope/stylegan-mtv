import math
import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .base import BaseOfflineProcessor


class InterpolationOfflineProcessor(BaseOfflineProcessor):
    def __init__(self, model_name='cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)

    def make_checkpoints(self, duration, n_points=3):
        """
        Returns a list of tuples with timestamps of when to hit which random point
        """
        points = self.get_random_points(n_points)
        checkpoints = []
        for i in range(n_points):
            timestamp = duration * (i / (n_points - 1))
            checkpoints.append((timestamp, points[i]))

        return checkpoints

    def interp_between_checkpoints(self, timestamp: float, beginning: tuple, end: tuple):
        beginning_ts, beginning_vec = beginning
        end_ts, end_vec = end
        timestamp_centered = timestamp - beginning_ts
        ratio = timestamp_centered / (end_ts - beginning_ts)
        return (((1 - ratio) * beginning_vec) + (end_vec * ratio)).reshape(1, -1)

    def get_images(self, sound_data, total_frames, duration, n_points):
        images = {}

        checkpoints = self.make_checkpoints(duration, n_points)

        beginning, end = checkpoints[0], checkpoints[1]
        checkpoint_idx = 0

        chunks = np.array_split(sound_data, total_frames)
        for i, frame in tqdm(enumerate(chunks), total=total_frames):
            timestamp = i / self.fps
            if timestamp > end[0]:
                checkpoint_idx += 1
                beginning, end = checkpoints[checkpoint_idx], checkpoints[checkpoint_idx + 1]

            latent_vec = self.interp_between_checkpoints(timestamp, beginning, end)

            images[i] = self.model.run_image(latent_vec, as_bytes=False)

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, input_path: str, output_path: str, start=0, duration=None, sr=None,
                     write=True, n_points=3):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        if n_points > 2:
            print('WARN: n_points must be >=2, setting to 2')
            n_points = 2

        sound_data, sample_rate = librosa.load(input_path, sr=sr, offset=start, duration=duration)
        duration = sound_data.shape[0] / sample_rate
        total_frames = math.ceil(duration * self.fps)

        self.get_images(sound_data, total_frames, duration, n_points)

        return self.create_video(duration, input_path, output_path, write=write, start=start)


if __name__ == '__main__':
    pass
