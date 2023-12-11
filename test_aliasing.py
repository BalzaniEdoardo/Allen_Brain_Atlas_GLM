import numpy as np
import matplotlib.pylab as plt

import utils.aliasing as aliasing


fig, axs = plt.subplots(2,2)
for base in [100]:
    x = np.linspace(0,1, 250)
    pix_sampling = 20
    alpha = 2
    #base = 400#10 * np.pi * (x.shape[0] // pix_sampling - 1)

    sampled, full, interpolated, coeffs, residuals = aliasing.check_sampling(
        val_sampling=None,
        pix_sampling=pix_sampling,
        func=aliasing.raised_cosine_log,
        x=x,
        alpha=alpha,
        base=base,
    )

    print(residuals)
    axs[0][0].set_title("sampled")
    axs[0][0].plot(sampled)
    axs[0][1].plot(sampled.sum(axis=1))
    axs[1][0].set_title("residuals")
    axs[1][0].plot(residuals)
    axs[1][1].set_title("interpolated")
    axs[1][1].plot(interpolated[:,::50],'--')
    axs[1][1].set_title("test")
    axs[1][1].plot(full[:, ::50])
plt.tight_layout()

