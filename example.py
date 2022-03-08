# %%
from visfit.main import VisFit1D
#%load_ext autoreload
#%autoreload 2

obsmsfile = "/raid/work/yamato/eDisk_data/L1489IRS/eDisk_calibrated_data/L1489IRS_SB1_continuum.ms"

model = VisFit1D(obsmsfile=obsmsfile)
model.load_vis()

# %%
fig, axes = model.plot_vis(
    axes=None, incl=73, PA=69, binsize=5, uvmax=1000, fmt="o", capsize=3, markersize=5
)  # , ecolor='black', markeredgecolor='black', color='black')
axes[0].set(ylim=(0.0, 0.05), xlim=(0,1000))
axes[1].set(ylim=(-0.0055, 0.0055), xlim=(0, 1000))
for ax in axes:
    ax.grid()

# %%
import numpy as np
from multiprocessing import Pool
from multiprocessing import set_start_method
from pathos.multiprocessing import ProcessingPool as Pool


def model_func_1d(r, I_g, sigma_g, I_p, sigma_p):
    return 10 ** I_g * np.exp(-0.5 * r ** 4 / sigma_g ** 4) + 10 ** I_p * np.exp(
        -0.5 * r ** 4 / sigma_p ** 4
    )


param_dict = {
    "I_g": {"p0": 6.0, "bound": (-2.0, 11.0), "fixed": False},
    "sigma_g": {"p0": 1.0, "bound": (0.1, 10), "fixed": False},
    "I_p": {"p0": 8.8, "bound": (-2.0, 11.0), "fixed": False},
    "sigma_p": {"p0": 0.01, "bound": (1e-5, 0.1), "fixed": False},
}

#with Pool(node=4) as pool:
model.fit_vis(
    model_func_1d=model_func_1d,
    param_dict=param_dict,
    nwalker=32,
    nstep=100,
    nthread=4,
    pool=None
)

# %%



