import torch


def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    if sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)

    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]

        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(
                cumprobs, r.sample([1]).to(data.device))]
    return init


def _krandinit(data: torch.Tensor, k: int, sample_size: int = -1):
    """Returns k samples of a random variable whose parameters depend on data.

    More precisely, it returns k observations sampled from a Gaussian random
    variable whose mean and covariances are the ones estimated from the data.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    References
    ----------
    .. [1] scipy/cluster/vq.py: _krandinit
    """
    mu = data.mean(axis=0)
    if sample_size > 0:
        data = data[torch.randint(0, int(data.shape[0]),
                                  [min(100000, data.shape[0])], device=data.device)]
    if data.ndim == 1:
        cov = torch.cov(data)
        x = torch.randn(k, device=data.device)
        x *= np.sqrt(cov)
    elif data.shape[1] > data.shape[0]:
        # initialize when the covariance matrix is rank deficient
        _, s, vh = data.svd(data - mu, full_matrices=False)
        x = torch.randn(k, s.shape[0])
        sVh = s[:, None] * vh / torch.sqrt(data.shape[0] - 1)
        x = x.dot(sVh)
    else:
        cov = torch.atleast_2d(torch.cov(data.T))

        # k rows, d cols (one row = one obs)
        # Generate k sample of a random variable ~ Gaussian(mu, cov)
        x = torch.randn(k, mu.shape[0], device=data.device)
        x = torch.matmul(x, torch.linalg.cholesky(cov).T)
    x += mu
    return x


def _kpoints(data, k, sample_size=-1):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int (not used)
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """

    return data[torch.randint(0, data.shape[0], size=[k], device=data.device)]

def _kfixedpoints(data, k, sample_size=-1):
    """Pick k points at random in data (one row = one observation).

    Parameters
    ----------
    data : ndarray
        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
        dimensional data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int (not used)
        sample data to avoid memory overflow during calculation

    Returns
    -------
    x : ndarray
        A 'k' by 'N' containing the initial centroids

    """

    return data[torch.arange(0, k, device=data.device)]


init_methods = {
    "gaussian": _krandinit,
    "kmeans++": _kpp,
    "random": _kpoints,
    "fixed": _kfixedpoints,
}
