import numpy as np
import cmath
from scipy import stats
import astropy.constants as ac
import astropy.units as u
import emcee
import corner
import matplotlib.pyplot as plt
import multiprocessing

# from multiprocessing import set_start_method
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from galario.double import get_image_size, check_image_size, sampleProfile, chi2Profile
from .utils import bin_weighted_average, run_emcee
import casatools
import casatasks
import subprocess

c = ac.c.to(u.m / u.s).value
pi = np.pi
deg = pi / 180.0  # in rad
arcsec = pi / 180.0 / 3600.0  # in rad

geometrical_param = ["PA", "incl", "dRA", "dDec"]
gp_default = {
    "PA": {"p0": 0.0, "bound": (0.0, 180.0), "fixed": True},
    "incl": {"p0": 0.0, "bound": (0.0, 90.0), "fixed": True},
    "dRA": {"p0": 0.0, "bound": (-5.0, 5.0), "fixed": True},
    "dDec": {"p0": 0.0, "bound": (-5.0, 5.0), "fixed": True},
}
accepted_chain_length = 100


tb = casatools.table()
ms = casatools.ms()


class VisFit1D:
    def __init__(self, obsmsfile, verbose=False):

        self.obsmsfile = obsmsfile
        self.verbose = verbose

    def load_vis(self, split_kwargs={"datacolumn": "data"}):
        # get number of channels for each spw
        tb.open(self.obsmsfile + "/SPECTRAL_WINDOW")
        self.nchans = tb.getcol("NUM_CHAN")
        tb.close()

        # split out to one channel per an spw  ### TODO: reconsider if this is proper: don't we need to consider the frequency differences between channels? ###
        tempvis = self.obsmsfile + ".split.temp"
        subprocess.run(["rm", "-r", tempvis])
        casatasks.split(
            vis=self.obsmsfile,
            outputvis=tempvis,
            keepflags=False,
            width=self.nchans,
            **split_kwargs
        )

        tb.open(tempvis)

        # get uvw coordinates and uv distance
        self.u, self.v, self.w = np.ascontiguousarray(tb.getcol("UVW"))
        self.uvdist = np.hypot(self.u.copy(), self.v.copy())

        # get weight
        weight_ori = tb.getcol("WEIGHT")

        datacolumn = split_kwargs.get("datacolumn", "data")
        if datacolumn.upper() in tb.colnames():
            data = np.ascontiguousarray(tb.getcol(datacolumn.upper()))
        else:
            raise KeyError("datacolumn {} not found.".format(datacolumn))

        # average over dual polarization
        V_XX = data[0, 0, :]
        V_YY = data[1, 0, :]
        weight_XX = weight_ori[0, :]
        weight_YY = weight_ori[1, :]

        self.V = (V_XX * weight_XX + V_YY * weight_YY) / (weight_XX + weight_YY)
        self.weight = weight_XX + weight_YY

        # self.amplitude = np.abs(self.V.copy())
        # self.phase = np.angle(self.V.copy(), deg=True)

        tb.close()

        # get frequency
        tb.open(tempvis + "/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ")  # Hz
        tb.close()

        self.obs_nu = freqs.mean()
        self.obs_wle = c / freqs.mean()  # m

        # remove temporary ms file
        subprocess.run(["rm", "-r", tempvis])

    def export_vis(self, filename):
        np.savetxt(
            filename,
            np.column_stack([self.u, self.v, self.V.real, self.V.imag, self.weight]),
            fmt="%10.6e",
            delimiter="\t",
            header="Extracted from {}.\nwavelength[m] = {}\nColumns:\tu[m]\tv[m]\tRe(V)[Jy]\tIm(V)[Jy]\tweight".format(
                self.obsmsfile, self.obs_wle
            ),
        )

    def plot_vis(
        self, axes=None, incl=0.0, PA=0.0, binsize=5, uvmax=500, **errorbar_kwargs
    ):
        if axes is None:
            return_fig = True
            fig, axes = plt.subplots(
                nrows=2, ncols=1, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
            )

        # apply the deprojection
        self.deproject_vis(incl=incl, PA=PA)

        # binning the uv data
        bins = np.arange(0, uvmax, binsize)
        self.bin_vis_deproj(bins=bins)

        # plot
        axes[0].errorbar(
            self.uvdist_deproj_binned,
            self.real_binned,
            yerr=self.real_binned_err,
            **errorbar_kwargs
        )
        axes[1].errorbar(
            self.uvdist_deproj_binned,
            self.imag_binned,
            yerr=self.imag_binned_err,
            **errorbar_kwargs
        )

        # label etc.
        axes[0].set(ylabel="Real [Jy]")
        axes[1].set(xlabel=r"Baseline [k$\lambda$]", ylabel="Imaginary [Jy]")

        if return_fig:
            return fig, axes

        else:
            return axes

    def bin_vis_deproj(self, bins=50):
        uvdist_klambda = self.uvdist_deproj / self.obs_wle / 1.0e3
        self.uvdist_deproj_binned, _, _ = stats.binned_statistic(
            uvdist_klambda, uvdist_klambda, bins=bins, statistic="mean"
        )
        self.real_binned, self.real_binned_err, _, _ = bin_weighted_average(
            uvdist_klambda, self.V.real, self.weight, bins=bins
        )
        self.imag_binned, self.imag_binned_err, _, _ = bin_weighted_average(
            uvdist_klambda, self.V.imag, self.weight, bins=bins
        )

    def deproject_vis(self, incl=0.0, PA=0.0):
        incl = np.radians(incl)
        PA = np.radians(PA)

        u = self.u.copy()
        v = self.v.copy()

        self.u_deproj = (u * np.cos(PA) - v * np.sin(PA)) * np.cos(incl)
        self.v_deproj = u * np.sin(PA) + v * np.cos(PA)

        self.uvdist_deproj = np.hypot(self.u_deproj, self.v_deproj)

    # def __call__(self, param):
    #     return self.log_probability(param)

    # def log_likelihood(self, param):

    #     param_dict = {name: p for name, p in zip(self.param_name, param)}

    #     # update fixed param
    #     param_dict.update(
    #         {name: self.param_dict[name]["p0"] for name in self.fixed_param_name}
    #     )

    #     # sample model visibility
    #     model_vis = self.sample_vis(self.model_func_1d, param_dict)

    #     # compute log likelihood
    #     rms = np.sqrt(1.0 / self.weight)
    #     ll = -0.5 * np.sum(
    #         (
    #             (model_vis.real - self.V.real) ** 2
    #             + (model_vis.imag - self.V.imag) ** 2
    #         )
    #         / rms ** 2
    #         + np.log(2 * pi * rms ** 2)
    #     )

    #     return ll

    # def log_probability(self, param):
    #     lp = log_prior(param, self.bound)
    #     if not np.isfinite(lp):
    #         return -np.inf
    #     ll = self.log_likelihood(param)
    #     return lp + ll

    def fit_vis(
        self,
        model_func_1d,
        param_dict,
        dish_D=12,
        nwalker=32,
        nstep=500,
        nthread=1,
        pool=None,
        blobs_dtype=None,
        progress=True,
        show_plot=True,
        labels=None,
        **corner_kwargs
    ):
        # setup radial grid
        self.set_grid(dish_D=dish_D)

        self.param_dict = param_dict

        self.model_func_1d = model_func_1d

        # add required geometrical params as needed
        # for gp in geometrical_param:
        #     if not gp in self.param_dict:
        #         self.param_dict.update(gp=gp_default[gp])

        # get lists of free/fixed parameter names
        self.param_name = [
            key for key in self.param_dict.keys() if not self.param_dict[key]["fixed"]
        ]
        self.fixed_param_name = [
            key for key in self.param_dict.keys() if self.param_dict[key]["fixed"]
        ]
        self.bound = [
            self.param_dict[key]["bound"]
            for key in self.param_dict.keys()
            if not self.param_dict[key]["fixed"]
        ]
        self.initial_state = [
            self.param_dict[key]["p0"]
            for key in self.param_dict.keys()
            if not self.param_dict[key]["fixed"]
        ]

        # parameters for likelihood function
        full_param_dict = self.param_dict.copy()
        free_param_name = self.param_name.copy()
        fixed_param_name = self.fixed_param_name.copy()
        obs_vis = self.V.copy()
        obs_weight = self.weight.copy()
        bound = self.bound.copy()
        initial_state = self.initial_state.copy()
        _r = self.r.copy()
        _nxy = self.nxy
        _dxy = self.dxy
        _u = self.u.copy()
        _v = self.v.copy()

        def sample_vis(model_func_1d, param_dict, r, nxy, dxy, u, v):

            # retrieve geometrical params
            PA = param_dict.pop("PA", gp_default["PA"]["p0"])
            incl = param_dict.pop("incl", gp_default["incl"]["p0"])
            dRA = param_dict.pop("dRA", gp_default["dRA"]["p0"])
            dDec = param_dict.pop("dDec", gp_default["dDec"]["p0"])

            # get model array
            model = model_func_1d(r / arcsec, **param_dict)

            # sampling by GALARIO
            vis = sampleProfile(
                intensity=model,
                Rmin=np.min(r),
                dR=np.diff(r)[0],
                nxy=nxy,
                dxy=dxy,
                u=u,
                v=v,
                dRA=dRA * arcsec,
                dDec=dDec * arcsec,
                PA=PA * deg,
                inc=incl * deg,
                check=False,
            )

            return vis

        # def log_likelihood(param):

        #     param_dict = {name: p for name, p in zip(free_param_name, param)}

        #     # update fixed param
        #     param_dict.update(
        #         {name: full_param_dict[name]["p0"] for name in fixed_param_name}
        #     )

        #     # sample model visibility
        #     # model_vis = sample_vis(model_func_1d, param_dict, _r, _nxy, _dxy, _u, _v)

        #     # # compute log likelihood
        #     # rms = np.sqrt(1.0 / obs_weight)
        #     # ll = -0.5 * np.sum(
        #     #     (
        #     #         (model_vis.real - obs_vis.real) ** 2
        #     #         + (model_vis.imag - obs_vis.imag) ** 2
        #     #     )
        #     #     / rms ** 2
        #     #     + np.log(2 * pi * rms ** 2)
        #     # )

        #     PA = param_dict.pop("PA", gp_default["PA"]["p0"])
        #     incl = param_dict.pop("incl", gp_default["incl"]["p0"])
        #     dRA = param_dict.pop("dRA", gp_default["dRA"]["p0"])
        #     dDec = param_dict.pop("dDec", gp_default["dDec"]["p0"])
        #     # PA = 0.0
        #     # incl = 0.0
        #     # dRA = 0.0
        #     # dDec = 0.0

        #     # get model array
        #     model = model_func_1d(_r / arcsec, **param_dict)

        #     chi2 = chi2Profile(
        #         intensity=model,
        #         Rmin=np.min(_r),
        #         dR=np.diff(_r)[0],
        #         nxy=_nxy,
        #         dxy=_dxy,
        #         u=_u,
        #         v=_v,
        #         vis_obs_re=np.ascontiguousarray(obs_vis.real),
        #         vis_obs_im=np.ascontiguousarray(obs_vis.imag),
        #         vis_obs_w=obs_weight,
        #         dRA=dRA * arcsec,
        #         dDec=dDec * arcsec,
        #         PA=PA * deg,
        #         inc=incl * deg,
        #         check=False,)

        #     return -0.5*chi2

        # def log_probability(param):
        #     lp = log_prior(param, bound)
        #     if not np.isfinite(lp):
        #         return -np.inf
        #     ll = log_likelihood(param)
        #     return lp + ll

        # ndim = len(self.initial_state)
        # p0 = self.initial_state + 1e-4 * np.random.randn(nwalker, ndim)

        # set smapler
        # with multiprocessing.Pool(processes=4) as pool:
        self.sampler = run_emcee(
            log_probability=log_probability,
            initial_state=initial_state,
            args=(bound,
                model_func_1d,
                full_param_dict,
                free_param_name,
                fixed_param_name,
                _r,
                _nxy,
                _dxy,
                _u,
                _v,
                obs_vis,
                obs_weight,),
            nwalker=nwalker,
            nstep=nstep,
            nthread=nthread,
            pool=pool,
            progress=progress,
            blobs_dtype=blobs_dtype,
        )
        # self.sampler = emcee.EnsembleSampler(
        #     nwalker, ndim, unwrap_log_probability, pool=pool, blobs_dtype=blobs_dtype, args=(model_func_1d,)
        # )

        # # run
        # print(
        #     "starting to run the MCMC sampling with: \n \t initial state:",
        #     self.initial_state,
        #     "\n \t number of walkers:",
        #     nwalker,
        #     "\n \t number of steps:",
        #     nstep,
        # )
        # self.sampler.run_mcmc(p0, int(nstep), progress=progress)

        # analyse chain
        ## check acceptance fraction: this should be between 0.2 and 0.5 (see Section 3 in https://arxiv.org/pdf/1202.3665.pdf)
        self.af = self.sampler.acceptance_fraction
        print(
            "Acceptance fraction: \n \t mean: {:g} \n \t min: {:g} \n \t max {:g} \n \t stddev: {:g}".format(
                np.mean(self.af), np.min(self.af), np.max(self.af), np.std(self.af)
            ),
            "*Acceptance fraction should be between 0.2 and 0.5 for convergence (see Section 3 in https://arxiv.org/pdf/1202.3665.pdf)",
        )

        ## autocorreration: automatically raise an error if tau > N/50
        # self.tau = self.sampler.get_autocorr_time()
        # print("Autocorreation time: {:g}".format(self.tau))

        # self.nburnin = int(3 * self.tau)
        # self.thin = int(self.tau / 2)
        self.nburnin = 50
        self.thin = 1

        self.sample = self.sampler.get_chain()
        self.sample_disc_flat = self.sampler.get_chain(
            discard=self.nburnin, thin=self.thin, flat=True
        )

        if show_plot:
            # corner plot
            corner_fig = corner.corner(
                self.sample_disc_flat,
                labels=labels if labels is not None else self.param_name,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                **corner_kwargs
            )

            # chain
            walker_figs = self.plot_walker(
                labels=labels if labels is not None else self.param_name
            )

            plt.show()

            return corner_fig, walker_figs

    def plot_walker(self, labels=None, histogram=True):
        #     # Check the length of the label list.

        # 	if labels is not None:
        # 		if sample.shape[0] != len(labels):
        # 			raise ValueError("Incorrect number of labels.")

        sample = self.sample.transpose((2, 0, 1))
        # Cycle through the plots.

        figset = []

        for i, s in enumerate(sample):
            fig, ax = plt.subplots()
            for walker in s.T:
                ax.plot(walker, alpha=0.1, color="k")
            ax.set_xlabel("Step number")
            if labels is not None:
                ax.set_ylabel(labels[i])
            ax.axvline(self.nburnin, ls="dotted", color="tab:blue")
            ax.set_xlim(0, s.shape[0])

            # Include the histogram.

            if histogram:
                fig.set_size_inches(
                    1.37 * fig.get_figwidth(), fig.get_figheight(), forward=True
                )
                ax_divider = make_axes_locatable(ax)
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(
                    s[self.nburnin :].flatten(), bins=bins, density=True
                )
                bins = np.average([bins[1:], bins[:-1]], axis=0)
                ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
                ax1.fill_betweenx(
                    bins,
                    hist,
                    np.zeros(bins.size),
                    step="mid",
                    color="darkgray",
                    lw=0.0,
                )
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which="both", left=0, bottom=0, top=0, right=0)
                ax1.spines["right"].set_visible(False)
                ax1.spines["bottom"].set_visible(False)
                ax1.spines["top"].set_visible(False)

                # get percentile
                q = np.percentile(s[self.nburnin :].flatten(), [16, 50, 84])
                for val in q:
                    ax1.axhline(val, ls="dashed", color="black")
                text = (
                    labels[i]
                    + r"$ = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                        q[1], np.diff(q)[0], np.diff(q)[1]
                    )
                    if labels is not None
                    else r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                        q[1], np.diff(q)[1], np.diff(q)[0]
                    )
                )
                ax1.text(0.5, 1.0, text, transform=ax1.transAxes, ha="center", va="top")

            figset.append(fig)

        return figset

    def sample_vis(self, model_func_1d, param_dict):

        # retrieve geometrical params
        PA = param_dict.pop("PA", gp_default["PA"]["p0"])
        incl = param_dict.pop("incl", gp_default["incl"]["p0"])
        dRA = param_dict.pop("dRA", gp_default["dRA"]["p0"])
        dDec = param_dict.pop("dDec", gp_default["dDec"]["p0"])

        # get model array
        model = model_func_1d(self.r / arcsec, **param_dict)

        # sampling by GALARIO
        vis = sampleProfile(
            intensity=model,
            Rmin=self.rmin,
            dR=self.dr,
            nxy=self.nxy,
            dxy=self.dxy,
            u=self.u,
            v=self.v,
            dRA=dRA * arcsec,
            dDec=dDec * arcsec,
            PA=PA * deg,
            inc=incl * deg,
            check=False,
        )

        return vis

    def set_grid(self, dish_D=12):
        # gridding parameter
        self.nxy, self.dxy = get_image_size(
            self.u, self.v, PB=1.22 * self.obs_wle / dish_D, verbose=self.verbose
        )  # in rad

        # condition for GALARIO interpolation: dxy/2 - Rmin > 5 * dR
        # not to make dR too small, adopt dxy/2/1000 as Rmin
        self.rmin = self.dxy / 2.0 * 1e-3  # in rad

        # dR = (dxy/2 - Rmin) / 5.1
        self.dr = (self.dxy / 2.0 - self.rmin) / 5.1  # in rad

        r_pad = 2 * self.dxy  # padding parameter
        self.rmax = self.dxy * self.nxy / np.sqrt(2) + r_pad  # in rad

        # radial grid in image plane
        self.r = np.arange(self.rmin, self.rmax, self.dr)  # in rad


def condition(p, b):
    if (p >= b[0]) and (p <= b[1]):
        return True
    return False


def log_prior(param, bound):
    for p, b in zip(param, bound):
        if not condition(p, b):
            return -np.inf
    return 0.0


# def log_probability(param, bound, log_likelihood):
#     lp = log_prior(param, bound)
#     if not np.isfinite(lp):
#         return -np.inf
#     ll = log_likelihood(param)
#     return lp + ll

# def unwrap_log_probability(arg, **kwargs):
#     return VisFit1D.log_probability(VisFit1D, arg, **kwargs)
