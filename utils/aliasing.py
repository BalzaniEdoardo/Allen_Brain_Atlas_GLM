import numpy as np

# normalization constant of a gaussian
GAUSSIAN_SUM = 2 * 1.753314144021452772415339526931980189073725635759454989253 - 1
def gaussian(x, std_dev=1):
    r"""Simple gaussian with mean 0, and adjustable std dev

    Possible alternative mother window, giving the weighting in each
    direction for the spatial pooling performed during the construction
    of visual metamers

    Parameters
    ----------
    x : float or array_like
        The distance in a direction
    std_dev : float or None, optional
        The standard deviation of the Gaussian window.

    Returns
    -------
    array
        The value of the window at each value of `x`

    Notes
    -----
    We normalize in here in order to make sure that the windows sum to
    1. In order to do that, we note that each Gaussian is centered at
    integer x values: 0, 1, 2, 3, etc. If we're summing at ``x=0``, we
    then note that the first window will be centered there and so have
    its max amplitude, its two nearest neighbors will be 1 away from
    their center (these Gaussians are symmetric), their two nearest
    neighbors will be 2 away from their center, etc. Therefore, we'll
    have one Gaussian at max value (1), two at
    :math:`\exp(\frac{-1^2}{2\sigma^2})`, two at
    :math:`\exp(\frac{-2^2}{2\sigma^2})`, etc.

    Summing at this location will give us the value we need to normalize
    by, :math:`S`. We work through this with :math:`\sigma=1`:

    ..math::

        S &= 1 + 2 * \exp(\frac{-(1)^2}{2\sigma^2}) + 2 * \exp(\frac{-(2)^2}{2\sigma^2}) + ...
        S &= 1 + 2 * \sum_{n=1}^{\inf} \exp({-n^2}{2})
        S &= -1 + 2 * \sum_{n=0}^{\inf} \exp({-n^2}{2})

    And we've stored this number as the constant ``GAUSSIAN_SUM`` (the
    infinite sum computed in the equation above was using Wolfram Alpha,
    https://www.wolframalpha.com/input/?i=sum+0+to+inf+e%5E%28-n%5E2%2F2%29+)

    When ``std_dev>1``, the windows overlap more. As with the
    probability density function of a normal distribution, we divide by
    ``std_dev`` to keep the integral constant for different values of
    ``std_dev`` (though the integral is not 1). This means that summing
    across multiple windows will still give us a value of 1.

    """
    return np.exp(-(x**2 / (2 * std_dev**2))) / (std_dev * GAUSSIAN_SUM)


def raised_cosine_linear(x, alpha=0.5):
    basis_func = 0.5 * (
            np.cos(
                np.clip(
                    np.pi * (x - 1) / alpha,
                    -np.pi,
                    np.pi,
                )
            )
            + 1
    )
    return basis_func


def raised_cosine_log(x, alpha=0.5):
    # if equi-spaced samples, this is equivalent to
    # log_spaced_pts = np.logspace(
    #   np.log10((self.n_basis_funcs - 1) * np.pi),
    #   -1,
    #   sample_pts.shape[0]
    # ) - 0.1
    # log_spaced_pts = log_spaced_pts / (np.pi * (self.n_basis_funcs - 1))
    # base = np.pi * (num_peaks - 1) * 10
    # log_spaced_pts = base ** (-x) - 1 / base
    basis_funcs = 0.5 * (
            np.cos(
                np.clip(
                    np.pi * x / alpha,
                    -np.pi,
                    np.pi,
                )
            )
            + 1
    )

    return basis_funcs


def check_sampling(val_sampling=.5, pix_sampling=None, func=gaussian, x=np.linspace(-5, 5, 101),
                   **func_kwargs):
    r"""check how sampling relates to interpolation quality

    Given a function, a domain, and how to sample that domain, this
    function will use linear algebra (``np.linalg.lstsq``) to determine
    how to interpolate the function so that it's centered on each
    pixel. You can then use functions like ``plot_coeffs`` and
    ``create_movie`` to see the quality of this interpolation

    The idea here is to take a function (for example,
    ``po.simul.pooling.gaussian``) and say that we have this function
    defined at, e.g., every 10 pixels on the array ``linspace(-5, 5,
    101)``. We want to answer then, the question of how well we can
    interpolate to all the intermediate functions, that is, the
    functions centered on each pixel in the array.

    You can either specify the spacing in pixels (``pix_sampling``) xor
    in x values (``val_sampling``), but exactly one of them must be set.

    Your function can either be a torch or numpy function, but ``x``
    must be the appropriate type, we will not cast it for you.

    Parameters
    ----------
    val_sampling : float or None, optional.
        If float, how far apart (in x-values) each sampled function
        should be. This doesn't have to align perfectly with the pixels,
        but should be close. If None, we use ``pix_sampling`` instead.
    pix_sampling : int or None, optional
        If int, how far apart (in pixels) each sampled function should
        be. If None, we use ``val_sampling`` instead.
    func : callable, optional
        the function to check interpolation for. must take ``x`` as its
        first input, all additional kwargs can be specified in
        ``func_kwargs``
    x : np.array, optional
        the 1d tensor/array to evaluate ``func`` on.
    func_kwargs :
        additional kwargs to pass to ``func``

    Returns
    -------
    sampled : np.array
        the array of sampled functions. will have shape ``(len(x),
        ceil(len(x)/pix_sampling))``
    full : np.array
        the array of functions centered at each pixel. will have shape
        ``(len(x), len(x))``
    interpolated : np.array
        the array of functions interpolated to each pixel. will have
        shape ``(len(x), len(x))``
    coeffs : np.array
        the array of coefficients to transform ``sampled`` to
        ``full``. This has been transposed from the array returned by
        ``np.linalg.lstsq`` and thus will have the same shape as
        ``sampled`` (this is to make it easier to restrict which coeffs
        to look at, since they'll be more easily indexed along first
        dimension)
    residuals : np.array
        the errors for each interpolation, will have shape ``len(x)``

    """
    if val_sampling is not None:
        if pix_sampling is not None:
            raise Exception("One of val_sampling or pix_sampling must be None!")
        # this will get us the closest value, if there's no exactly
        # correct one.
        pix_sampling = np.argmin(abs((x+val_sampling)[0] - x))
        if pix_sampling == 0 or pix_sampling == (len(x)-1):
            # the above works if x is increasing. if it's decreasing,
            # then pix_sampling will be one of the extremal values, and
            # we need to try the following
            pix_sampling = np.argmin(abs((x-val_sampling)[0] - x))
    if func.__name__ != "raised_cosine_log":
        X = x[:, None] + x[::pix_sampling]
    else:
        num_peaks = x[::pix_sampling].shape[0]
        base = np.pi * (num_peaks - 1) * 10
        log_spaced_pts = base ** (-x) - 1 / base
        peaks = np.linspace(0, 1, num_peaks)
        X = log_spaced_pts[:,None] - peaks[None]

    sampled = func(X, **func_kwargs)
    if func.__name__ != "raised_cosine_log":
        full_X = x[:, None] + x
    else:
        num_peaks = x.shape[0]
        base = np.pi * (num_peaks - 1) * 10
        log_spaced_pts = base ** (-x) - 1 / base
        peaks = np.linspace(0, 1, num_peaks)
        full_X = log_spaced_pts[:, None] - peaks[None]
    full = func(full_X, **func_kwargs)
    coeffs, residuals, rank, s = np.linalg.lstsq(sampled, full, rcond=None)
    interpolated = np.matmul(sampled, coeffs)
    return sampled, full, interpolated, coeffs.T, residuals
