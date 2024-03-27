from scipy.ndimage import zoom
import numpy as np
def gaussian_function_parametric(a, b, c, x=2.):
    return a * np.exp(-1 * np.divide(np.power((x - b), 2.), (2. * np.power(c, 2.))))

def normal_distribution_pdf(mu, sigma, x):
    a = 1 / (sigma * np.sqrt(2. * np.pi))
    b = mu
    c = sigma
    return gaussian_function_parametric(a, b, c, x)

def upsample_channel(channel, target_size):
    """
    Upsamples a specific channel of an image to the target size.

    Parameters:
        image: The input image with dimensions (C, H, W), where C is the number of channels.
        channel_index: The index of the channel to be upsampled.
        target_size: The target size (height and width) of the upsampled channel.

    Returns:
        A new image with the specified channel upsampled to the target size.
    """
    # Extract the channel to be upsampled

    # Calculate the zoom factor for each dimension
    zoom_factor = [target_size[0] / channel.shape[0], target_size[1] / channel.shape[1]]

    # Upsample the channel
    upsampled_channel = zoom(channel, zoom=zoom_factor, order=0)
    return upsampled_channel
    
def radial_basis_obs(target_x, target_y, target_value, dim=None):
    if dim is None:
        dim = [84, 84]
    # For stability, this is necessary
    if target_value < 0.005:
        target_value = 0.005
    pos_x_range = [-1, 1]
    mag_x_range = pos_x_range[1] - pos_x_range[0]
    pos_y_range = [-1, 1]
    mag_y_range = pos_y_range[1] - pos_y_range[0]

    mu_x = target_x
    sigma_x = target_value * (mag_x_range / 16)

    mu_y = target_y
    sigma_y = target_value * (mag_y_range / 16)

    grid_xi = np.linspace(start=pos_x_range[0], stop=pos_x_range[1], num=dim[0])
    pdf_x = normal_distribution_pdf(mu=mu_x, sigma=sigma_x, x=grid_xi)
    grid_yi = np.linspace(start=pos_y_range[0], stop=pos_y_range[1], num=dim[1])
    pdf_y = normal_distribution_pdf(mu=mu_y, sigma=sigma_y, x=grid_yi)

    pdf_x = np.reshape(pdf_x, newshape=(dim[0], 1))
    pdf_y = np.reshape(pdf_y, newshape=(1, dim[1]))

    grid_obs = pdf_x * pdf_y
    grid_obs /= (grid_obs.max())
    return grid_obs