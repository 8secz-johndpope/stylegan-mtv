# import inspect
# from ipywidgets import get_ipython
#
# BYPASS_FILES = ['process.py']
#
#
# def is_notebook():
#     try:
#         for frame in inspect.stack()[1:]:
#             if frame.filename[0] != '<':
#                 if frame.filename in BYPASS_FILES:
#                     print('bypassing because it is', frame.filename)
#                     return True
#                 # break
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell' or 'colab' in shell or shell == 'Shell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter
#
#
# use_tf = True
# using_click = False
# using_sound = True
#
# if not is_notebook():
#     print('loading up globals...')
#
#     # shared_latent_vecs = sharedMemory()
#
#     if use_tf:
#         print('using tensorflow!')
#         from .camera import TFCamera
#         # gan = TFCamera(shared_latent_vecs)
#         gan = TFCamera()
#     else:
#         print('using torch!')
#         from .camera import TorchCamera
#         gan = TorchCamera()
#
#     from .controllers.click import ClickController
#     from .controllers.sound import SpectrogramSoundController
#     from .sound import BaseAudioPlayer
#
#     if using_click:
#         gan.controller = ClickController(gan.current_model)
#     elif using_sound:
#         gan.controller = SpectrogramSoundController(gan.current_model)
#
#     # audio_player = BaseAudioPlayer()
#     # gan.controller = BaseSoundController(gan.current_model, audio_player)
# else:
#     print('in notebook, not loading globals')
#
# if __name__ == '__main__':
#     print('main!')
