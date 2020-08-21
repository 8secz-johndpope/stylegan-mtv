import sys
import numpy as np

from .settings import MODEL_DIR, LATENT_DIR, LIB_DIR


def warn(*values):
    print(*values, file=sys.stderr)


def add_src_to_sys_path():
    print('********', LIB_DIR)
    sys.path.append(str(LIB_DIR))
    print(sys.path)


def make_model_map(model_files):
    model_map = {}
    for p in model_files:
        split = p.stem.split('_')
        if len(split[0]) > 20:  # dumb hack to drop out the url MD5 hash:
            print('splitting this filepath:', p.stem)
            name = '_'.join(split[1:])
        else:
            name = p.stem
        model_map[name] = str(p)
    return model_map


def find_tf_models():
    model_files = MODEL_DIR.glob('*.pkl')
    return make_model_map(model_files)


def find_latent_representations():
    reps = LATENT_DIR.glob('*.npy')
    return {i.stem: str(i) for i in reps}


def load_latent_reps():
    latent_reps_paths = find_latent_representations()
    reps = {name: np.expand_dims(np.load(path), axis=0)  # is this expand_dims necessary?
            for name, path in latent_reps_paths.items()}
    return reps


# def find_torch_models():
#     model_files = MODEL_DIR.glob('*.pt')
#     return make_model_map(model_files)


if __name__ == '__main__':
    warn('this is a test')
    warn('this is another', 'test', 'haha cool')
    print(find_tf_models())
    # print(find_torch_models())
