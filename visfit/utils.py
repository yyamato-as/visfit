from scipy import stats
import numpy as np


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
