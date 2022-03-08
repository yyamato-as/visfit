from scipy import stats
import numpy as np
import emcee
import multiprocessing
from multiprocess import set_start_method
from pathos.multiprocessing import Pool
from multiprocess import Pool


def bin_weighted_average(x, y, weights, bins=np.arange(0, 100, 10), std_err=False):
    w, edge_w, num_w = stats.binned_statistic(x, weights, bins=bins, statistic="sum")
    yw, edge_yw, num_yw = stats.binned_statistic(
        x, y * weights, bins=bins, statistic="sum"
    )

    assert np.all(edge_w == edge_yw)
    assert np.all(num_w == num_yw)

    if std_err:
        err, edge_e, num_e = stats.binned_statistic(x, y, bins=bins, statistic="std")
        assert np.all(edge_e == edge_yw)
        assert np.all(num_e == num_yw)
    else:
        err = 1.0 / np.sqrt(w)

    return yw / w, err, edge_yw, num_yw

def run_emcee(
    log_probability,
    initial_state,
    args=None,
    nwalker=200,
    nstep=500,
    initial_blob_mag=1e-4,
    nthread=1,
    pool=None,
    progress=True,
    blobs_dtype=None,
):

    # set dimension and initial guesses
    ndim = len(initial_state)
    p0 = initial_state + initial_blob_mag * np.random.randn(nwalker, ndim)

    # set smapler
    #with ProcessPool(node=nthread) as pool:
    #with multiprocessing.Pool(processes=nthread) as pool:
    #with Pool(nthread) as pool:
    pool = None
    sampler = emcee.EnsembleSampler(
        nwalker, ndim, log_probability, args=args, pool=pool, blobs_dtype=blobs_dtype,
    )

    # run
    print(
        "starting to run the MCMC sampling with: \n \t initial state:",
        initial_state,
        "\n \t number of walkers:",
        nwalker,
        "\n \t number of steps:",
        nstep
    )
    sampler.run_mcmc(p0, nstep, progress=progress)

    return sampler
