def log_likelihood(
    param,
    model_func_1d,
    full_param_dict,
    free_param_name,
    fixed_param_name,
    r,
    nxy,
    dxy,
    u,
    v,
    obs_vis,
    obs_weight,
):

    param_dict = {name: p for name, p in zip(free_param_name, param)}

    # update fixed param
    param_dict.update({name: full_param_dict[name]["p0"] for name in fixed_param_name})

    # sample model visibility
    # model_vis = sample_vis(model_func_1d, param_dict, _r, _nxy, _dxy, _u, _v)

    # # compute log likelihood
    # rms = np.sqrt(1.0 / obs_weight)
    # ll = -0.5 * np.sum(
    #     (
    #         (model_vis.real - obs_vis.real) ** 2
    #         + (model_vis.imag - obs_vis.imag) ** 2
    #     )
    #     / rms ** 2
    #     + np.log(2 * pi * rms ** 2)
    # )

    PA = param_dict.pop("PA", gp_default["PA"]["p0"])
    incl = param_dict.pop("incl", gp_default["incl"]["p0"])
    dRA = param_dict.pop("dRA", gp_default["dRA"]["p0"])
    dDec = param_dict.pop("dDec", gp_default["dDec"]["p0"])
    # PA = 0.0
    # incl = 0.0
    # dRA = 0.0
    # dDec = 0.0

    # get model array
    model = model_func_1d(r / arcsec, **param_dict)

    chi2 = chi2Profile(
        intensity=model,
        Rmin=np.min(r),
        dR=np.diff(r)[0],
        nxy=nxy,
        dxy=dxy,
        u=u,
        v=v,
        vis_obs_re=np.ascontiguousarray(obs_vis.real),
        vis_obs_im=np.ascontiguousarray(obs_vis.imag),
        vis_obs_w=obs_weight,
        dRA=dRA * arcsec,
        dDec=dDec * arcsec,
        PA=PA * deg,
        inc=incl * deg,
        check=False,
    )

    return -0.5 * chi2


def log_probability(
    param,
    bound,
    model_func_1d,
    full_param_dict,
    free_param_name,
    fixed_param_name,
    r,
    nxy,
    dxy,
    u,
    v,
    obs_vis,
    obs_weight,
):
    lp = log_prior(param, bound)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(
        param,
        model_func_1d,
        full_param_dict,
        free_param_name,
        fixed_param_name,
        r,
        nxy,
        dxy,
        u,
        v,
        obs_vis,
        obs_weight,
    )
    return lp + ll