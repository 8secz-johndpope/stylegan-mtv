import click

from src import add_src_to_sys_path, SpectrogramOfflineProcessor

add_src_to_sys_path()

@click.group()
def cli():
    pass


@click.command()
@click.option('--model_name', default='wikiart', help='model name', type=str)
@click.option('--fps', default=24, help='frames per second', type=int)
@click.option('--random_seed', default=False, help='random seed', type=int)
@click.option('--start', default=0, help='Start time', type=int)
@click.option('--duration', default=None, help='Duration of video to make', type=int)
@click.option('--sr', default=None, help='sample rate', type=int)
@click.option('--window_size', default=5, help='Window size', type=int)
@click.option('--displacement_factor', default=0.1, help='Displacement factor', type=float)
@click.option('--frame_chunk_size', default=500, help='Number of frames to batch before writing to disk', type=int)
@click.option('--no_write', is_flag=True, help='Do not write out video.')
@click.argument('input_path')
@click.argument('output_path')
def spectro(model_name, fps, random_seed, start, duration, sr, window_size, displacement_factor,
            frame_chunk_size, no_write, input_path, output_path):
    print('================ PARAMETERS')
    print(model_name, fps, random_seed, input_path, output_path, duration, )

    processor = SpectrogramOfflineProcessor(model_name, fps, random_seed, frame_chunk_size)
    processor.process_file(input_path, output_path, start, duration, sr, not no_write, window_size, displacement_factor)


@click.command()
def dropdb():
    click.echo('Dropped the database')


cli.add_command(spectro)
cli.add_command(dropdb)

if __name__ == '__main__':
    cli()
