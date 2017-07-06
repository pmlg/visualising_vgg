"""
Small functions to make life easier.
"""

import inspect

from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython

import numpy as np


def display_python(code):
    """ Pretty prints raw code text in jupyter notebooks. """
    formatter = HtmlFormatter()
    x = IPython.display.HTML(
        '<style type="text/css">{}</style>    {}'.format(
            formatter.get_style_defs('.highlight'),
            highlight(code, PythonLexer(), formatter))
        )
    return x


def iterate_np(arr, axis=3):
    """ A generator that allows you to iterate through numpy arrays.
    
    NB. This is only partially implemented because I haven't sorted out the
    splice along variable axes yet.
    """
    for i in range(arr.shape[axis]):
        yield arr[:, :, :, i]


def deprocess_image(x):
    """ Transforms numpy arrays into valid images.

    Code is from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
    """
    from keras import backend as K

    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
