"""
Functions for plotting the output of keras layers.
"""

from math import ceil

from matplotlib import pyplot as plt
from keras.models import Model

from .utils import iterate_np

def plot_outputs(image, layer, model, indices, ncols=8, figsize=(15,5)):
    """ Plots the outputs from a truncated model after passing an image.

    Keyword arguments:
    image --  A numpy array of size 224 * 224 * 3 in standard image encoding.
    layer -- A string specifying the layer to truncate the model to.
        Requires that the layers are actually named.
    model -- A keras model object.
    indices -- The indices of the filters to visualise the output from.
    ncols -- The number of columns to show in the grid.
    figsize -- The size of the figure to show in inches.
    """

    # Handle defaults for indices
    if isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = np.arange(model.get_layer(layer).output_shape[3])

    # Convert input image to keras compatible shape
    if len(image.shape) == 3:
        image = image.reshape((1, 224, 224, 3))

    # Construct a truncated model
    input_layer = model.input
    output_layer = model.get_layer(layer).output

    trunc_model = Model(inputs=input_layer, outputs=output_layer)
    
    # Run the prediction to get output from truncated model
    outputs = trunc_model.predict(image)

    # Just to keep our plots nice.
    if ncols > len(indices):
        ncols = len(indices)

    nrows = ceil(len(indices) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={'xticks': [], 'yticks': []},
        figsize=figsize
        )

    try:
        faxes = axes.flatten()
    except:
        faxes = [axes]

    for ax, filt in zip(faxes, iterate_np(outputs[:, :, :, indices], axis=3)):
        ax.imshow(filt[0], cmap="Greys")

    return fig, axes

