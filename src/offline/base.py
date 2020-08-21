import cv2
import math
import ffmpeg
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

from ..utils import warn, find_tf_models
from src.model import StyleGANModel

AVAILABLE_TF_MODELS = find_tf_models()


class BaseOfflineProcessor:
    def __init__(self, model_name: str = 'cats', fps=5, random_seed=False, frame_chunk_size=500):
        self.fps = fps
        self.frame_chunk_size = frame_chunk_size
        if model_name not in AVAILABLE_TF_MODELS and model_name not in AVAILABLE_TF_MODELS:
            print(f'Model {model_name} not found!!!')
        self.random_seed = random_seed
        self.temp_dir, self.temp_path = None, None

    @staticmethod
    def open_wav_file(path: str):
        sample_rate, data = wavfile.read(path)
        duration = len(data) / sample_rate
        return sample_rate, data, duration

    def write_chunk_to_temp(self, images):
        processed_images = self.postprocess_images(images)
        print('writing frame chunk...')

        for idx, img in processed_images.items():
            # write image to temp dir
            img_path = self.temp_path / f'frame_{idx:09}.bmp'
            Image.fromarray(img).save(img_path)

    def get_images(self, sound_data, total_frames, *args, **kwargs):
        images = {}

        chunks = np.array_split(sound_data, total_frames)
        for i, chunk in tqdm(enumerate(chunks), total=total_frames):
            latent_vec = self.get
            timestamp = i / total_frames
            latent_vec = self.controller.get_latent_vecs(sound_data, timestamp)[0]
            image = self.model.run_image(latent_vec, as_bytes=False)
            images[i] = image

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, input_path: str, output_path: str, write=True, start=0):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        sample_rate, sound_data, duration = self.open_wav_file(input_path)
        total_frames = math.ceil(duration * self.fps)

        print('generating images...')
        images = self.get_images(sound_data, total_frames)

        return self.create_video(duration, input_path, output_path, write=write, start=start)

    def postprocess_images(self, images):
        # A place to reformat images, if necessary
        print('post processing images...')
        return images

    def create_video(self, duration, input_path, output_path, write=True, start=0):
        print(f'images generated, outputting to clip starting at {start} for {duration} seconds...')

        vid_in = ffmpeg.input(str(self.temp_path / '*.bmp'), pattern_type='glob', framerate=self.fps)
        audio_in = ffmpeg.input(input_path, ss=start, t=duration)

        joined = ffmpeg.concat(vid_in, audio_in, v=1, a=1).node
        out = ffmpeg.output(joined[0], joined[1], output_path).overwrite_output()
        out.run()

        self.temp_dir.cleanup()


class BaseTFOfflineProcessor(BaseOfflineProcessor):
    def __init__(self, model_name: str = 'cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)
        if model_name in AVAILABLE_TF_MODELS:
            self.model_name = model_name
        else:
            fallback_model = list(AVAILABLE_TF_MODELS.keys())[0]
            warn(f'Model {model_name} not available, falling back to {fallback_model}')
            self.model_name = fallback_model

        self.model = StyleGANModel(AVAILABLE_TF_MODELS[self.model_name], random_seed=random_seed, reduced_memory=False)
        self.controller = None

    def postprocess_images(self, images):
        # Flip the coloring...
        print('post processing images...')
        final_images = {}
        for idx, img in images.items():
            final_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return final_images


if __name__ == '__main__':
    pass



