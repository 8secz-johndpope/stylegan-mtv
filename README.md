# stylegan-mtv
Use a StyleGAN model to generate videos reacting to sound.

# Spectrogram Only
```
python process.py spectro [OPTIONS] INPUT_PATH OUTPUT_PATH

Options:
  --model_name TEXT            model name
  --fps INTEGER                frames per second
  --random_seed INTEGER        random seed
  --start INTEGER              Start time
  --duration INTEGER           Duration of video to make
  --sr INTEGER                 sample rate
  --window_size INTEGER        Window size
  --displacement_factor FLOAT  Displacement factor
  --no_write                   Do not write out video.
  --help                       Show help and exit.
```
