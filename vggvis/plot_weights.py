"""
Functions and utilities to visualise weights of layers in CNNs
"""

from math import ceil

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Dense

from vggvis.utils import iterate_np
from vggvis.utils import deprocess_image


def plot_filters(layer, indices=None, ncols=8, figsize=(6, 6)):
    """ Takes keras layers weights and plots the filter weights in a grid.

    keyword arguments:
    layer -- A keras convolutional layer.
    indices -- The indices of the filters to visualise the output from.
    ncols -- The number of columns to plot for the grid. If plot_num = 64 and
        ncols = 8 then the plot would be 8 * 8.
    figsize -- The size of the image in inches.
    """
    # Weights returned by `.get_weights()` is a tuple
    filters, biases = layer.get_weights()


    # Handle defaults for indices
    if isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        indices = np.arange(biases.shape[0])

    if ncols > len(indices):
        ncols = len(indices)

    nrows = ceil(len(indices) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={'xticks': [], 'yticks': []},
        figsize=figsize
        )

    for ax, filt in zip(axes.flatten(),
                        iterate_np(filters[:, :, :, indices], axis=3)):
        ax.imshow(deprocess_image(filt))
    return fig, axes


def max_act(
        layer,
        index,
        model,
        initial=None,
        steps=100,
        lr=1,
        l2_reg=0.9,
        mult_lr=0.995,
        blur=True,
        blur_amount=0.5
        ):
    """ Computes the image that maximally activates a specific layer.

    Keyword arguments:
    layer -- The name of the layer in the model. Layers must be named in model.
    index -- The index of the layer to visualise.
    model -- The keras model to optimise to.
    initial -- Can provide an image to start from, otherwise noise.
    steps -- The number of iterations to improve the model with.
    lr -- The rate at which we climb the gradient.
    mult_lr -- Multiply lr by constant at each step to change over time.
    l2_reg -- The regularisation amount to penalise large pixel values.
    blur -- Use gaussian blur.
    blur_amount -- The amount of blur to apply.
    """

    img_shape = model.input_shape[1:3]

    input_ = model.input

    # Need to get different axes depending on the layer type
    output_layer = model.get_layer(layer)

    if isinstance(output_layer, Dense):
        output_ = output_layer.output[:, index]
    else:
        output_ = output_layer.output[:, :, :, index]

    # Build a loss function that maximises the activation of the filter
    # NB `K` is the keras backend.
    loss = K.mean(output_)

    # Compute the gradient of the input picture wrt this loss.
    grads = K.gradients(loss, input_)[0]

    # Normalise the gradients, similar to other grad optimisers
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Function that returns the loss and gradient given an input pic
    iterate = K.function([input_], [loss, grads])

    # Initialise the image as gray noise if no input picture provided.

    from numpy import random as rand
    if initial is None:
        img_data = rand.random((1, img_shape[0], img_shape[1], 3)) * 20 + 128.
    else:
        img_data = initial.reshape(1, img_shape[0], img_shape[1], 3) * 1.

    output_images = [np.copy(img_data[0])]
    #output_gradients = []

    for i in range(steps):
        loss_value, grads_value = iterate([img_data])

        # Step the image to increase loss/climb gradient
        img_data += grads_value * lr

        # Save the image and gradient at each step for output.
        output_images.append(np.copy(img_data[0]))
        #output_gradients.append(np.copy(grads_value[0]))

        # Apply l2 style regularisation
        img_data *= l2_reg

        # If using blur, apply it.
        if blur:
            sigma = (0, blur_amount, blur_amount, 0)
            img_data = gaussian_filter(img_data, sigma=sigma)

        # Adjust learning rate
        lr *= mult_lr

    return np.array(output_images)


def plot_grid_max_acts(
        indices,
        layer,
        model,
        ncols=8,
        figsize=(15, 15),
        fn=max_act,
        **kwargs
        ):
    """ Plots a grid of activation images.

    Keyword arguments:
    indices -- The indices of the filters that you'd like to visualise
    layer -- The layer to get the filters from
    model -- The keras model
    ncols -- The number of columns to plot in the grid
    figsize -- The size of the grid in inches
    fn -- The function to use to generate the images.
        Must return a single array of images.
    kwargs -- Any valid argument to `fn`.
    """

    # Handle defaults for indices
    if isinstance(indices, int):
        indices = [indices]
    elif indices is None:
        nfilter = model.get_layer(layer).output_shape[3]
        indices = np.arange(nfilter)

    if ncols > len(indices):
        ncols = len(indices)

    nrows = ceil(len(indices) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={'xticks': [], 'yticks': []},
        figsize=figsize
        )

    images = list()
    for index in indices:
        image = fn(layer=layer, index=index, model=model, **kwargs)
        images.append(image[-1])

    for ax, filt in zip(axes.flatten(), images):
        ax.imshow(deprocess_image(filt))
    return fig, axes


def plot_anim_max_acts(
        index,
        layer,
        model,
        figsize=(6, 6),
        interval=100,
        repeat_delay=1000,
        blit=True,
        fn=max_act,
        **kwargs
        ):
    """ Plots the gradient ascent of an activation image as an animation.

    Keyword arguments:
    layer -- The name of the layer in the model. Layers must be named in model.
    index -- The filter index to visualise, currently only one at a time.
    model -- The keras model to optimise to.
    figsize -- The size of the grid in inches
    interval -- Passed to matplotlib ArtistAnimation.
    repeat_delay -- Passed to matplotlib ArtistAnimation.
    blit -- Passed to matplotlib ArtistAnimation.
    fn -- The function to use to generate the images.
        Must return a single array of images.
    kwargs -- Any valid argument to `fn`.
    """
    # Initialise the figure
    fig = plt.figure(figsize=figsize)

    # Get the image(s)
    images = list(fn(layer=layer, index=index, model=model, **kwargs))

    # Generate a list of image plots.
    imshows = [[plt.imshow(deprocess_image(x), animated=True)] for x in images]
    plt.axis('off')

    # Get an animation object
    anim = ArtistAnimation(
        fig,
        imshows,
        interval=interval,
        repeat_delay=repeat_delay,
        blit=True
        )

    # Closing the plot prevents showing the static plot.
    plt.close()

    return anim

