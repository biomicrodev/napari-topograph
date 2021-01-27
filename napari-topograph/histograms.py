import dask.array as da
import numpy as np
from dask import is_dask_collection
from dask.array import Array, asarray
from dask.array.routines import _linspace_from_delayed
from dask.core import flatten
from dask.delayed import Delayed, tokenize, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from fast_histogram import histogram1d, histogram2d


def _block_fast_hist1d(x, bins, range=None, weights=None) -> np.ndarray:
    # to use fast_histogram, bins are assumed to be evenly spaced
    return histogram1d(x, len(bins), range, weights)[:-1][np.newaxis, ...]


def _block_fast_hist2d(x, y, bins, range=None, weights=None) -> np.ndarray:
    return histogram2d(x, y, bins, range, weights)[np.newaxis, ...]


def dask_hist1d(
    a: Array, bins=None, range=None, normed=False, weights=None, density=None
):
    """
    Blocked variant of :func:`numpy.histogram`, but using the fast-histogram module.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars, optional
        Either an iterable specifying the ``bins`` or the number of ``bins``
        and a ``range`` argument is required as computing ``min`` and ``max``
        over blocked arrays is an expensive operation that must be performed
        explicitly.
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.
    normed : bool, optional
        This is equivalent to the ``density`` argument, but produces incorrect
        results for unequal bin widths. It should not be used.
    weights : array_like, optional
        A dask.array.Array of weights, of the same block structure as ``a``.  Each value in
        ``a`` only contributes its associated weight towards the bin count
        (instead of 1). If ``density`` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.
        Overrides the ``normed`` keyword if given.
        If ``density`` is True, ``bins`` cannot be a single-number delayed
        value. It must be a concrete number, or a (possibly-delayed)
        array/sequence of the bin edges.
    Returns
    -------
    hist : dask Array
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : dask Array of dtype float
        Return the bin edges ``(length(hist)+1)``.


    Examples
    --------
    Using number of bins and range:

    >>> import dask.array as da
    >>> import numpy as np
    >>> x = da.from_array(np.arange(10000), chunks=10)
    >>> h, bins = da.histogram(x, bins=10, range=[0, 10000])
    >>> bins
    array([    0.,  1000.,  2000.,  3000.,  4000.,  5000.,  6000.,  7000.,
            8000.,  9000., 10000.])
    >>> h.compute()
    array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

    Explicitly specifying the bins:

    >>> h, bins = da.histogram(x, bins=np.array([0, 5000, 10000]))
    >>> bins
    array([    0,  5000, 10000])
    >>> h.compute()
    array([5000, 5000])
    """
    if isinstance(bins, Array):
        scalar_bins = bins.ndim == 0
        # ^ `np.ndim` is not implemented by Dask array.
    elif isinstance(bins, Delayed):
        scalar_bins = bins._length is None or bins._length == 1
    else:
        scalar_bins = np.ndim(bins) == 0

    if bins is None or (scalar_bins and range is None):
        raise ValueError(
            "dask.array.histogram requires either specifying "
            "bins as an iterable or specifying both a range and "
            "the number of bins"
        )

    if weights is not None and weights.chunks != a.chunks:
        raise ValueError("Input array and weights must have the same chunked structure")

    if normed is not False:
        raise ValueError(
            "The normed= keyword argument has been deprecated. "
            "Please use density instead. "
            "See the numpy.histogram docstring for more information."
        )

    if density and scalar_bins and isinstance(bins, (Array, Delayed)):
        raise NotImplementedError(
            "When `density` is True, `bins` cannot be a scalar Dask object. "
            "It must be a concrete number or a (possibly-delayed) array/sequence of bin edges."
        )

    for argname, val in [("bins", bins), ("range", range), ("weights", weights)]:
        if not isinstance(bins, (Array, Delayed)) and is_dask_collection(bins):
            raise TypeError(
                "Dask types besides Array and Delayed are not supported "
                "for `histogram`. For argument `{}`, got: {!r}".format(argname, val)
            )

    if range is not None:
        try:
            if len(range) != 2:
                raise ValueError(
                    f"range must be a sequence or array of length 2, but got {len(range)} items"
                )
            if isinstance(range, (Array, np.ndarray)) and range.shape != (2,):
                raise ValueError(
                    f"range must be a 1-dimensional array of two items, but got an array of shape {range.shape}"
                )
        except TypeError:
            raise TypeError(
                f"Expected a sequence or array for range, not {range}"
            ) from None

    token = tokenize(a, bins, range, weights, density)
    name = "histogram-sum-" + token

    if scalar_bins:
        bins = _linspace_from_delayed(range[0], range[1], bins + 1)
        # ^ NOTE `range[1]` is safe because of the above check, and the initial check
        # that range must not be None if `scalar_bins`
    else:
        if not isinstance(bins, (Array, np.ndarray)):
            bins = asarray(bins)
        if bins.ndim != 1:
            raise ValueError(
                f"bins must be a 1-dimensional array or sequence, got shape {bins.shape}"
            )

    (bins_ref, range_ref), deps = unpack_collections([bins, range])

    # Map the histogram to all bins, forming a 2D array of histograms, stacked for each chunk
    if weights is None:
        dsk = {
            (name, i, 0): (_block_fast_hist1d, k, bins_ref, range_ref)
            for i, k in enumerate(flatten(a.__dask_keys__()))
        }
        dtype = np.histogram([])[0].dtype
    else:
        a_keys = flatten(a.__dask_keys__())
        w_keys = flatten(weights.__dask_keys__())
        dsk = {
            (name, i, 0): (_block_fast_hist1d, k, bins_ref, range_ref, w)
            for i, (k, w) in enumerate(zip(a_keys, w_keys))
        }
        dtype = weights.dtype

    deps = (a,) + deps
    if weights is not None:
        deps += (weights,)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)

    # Turn graph into a 2D Array of shape (nchunks, nbins)
    nchunks = len(list(flatten(a.__dask_keys__())))
    nbins = bins.size - 1  # since `bins` is 1D
    chunks = ((1,) * nchunks, (nbins,))
    mapped = Array(graph, name, chunks, dtype=dtype)

    # Sum over chunks to get the final histogram
    n = mapped.sum(axis=0)

    # We need to replicate normed and density options from numpy
    if density is not None:
        if density:
            db = asarray(np.diff(bins).astype(float), chunks=n.chunks)
            return n / db / n.sum(), bins
        else:
            return n, bins
    else:
        return n, bins


def dask_hist2d(x: da.Array, y: da.Array, bins: int, range, density=False):
    if x.shape != y.shape:
        raise ValueError(
            f"Mismatch in argument shaoes: x.shape == {x.shape}; y.shape == {y.shape}"
        )

    token = tokenize(x, y, bins, range, density)
    name = "histogram2d-sum-" + token

    x_keys = flatten(x.__dask_keys__())
    y_keys = flatten(y.__dask_keys__())

    dsk = {
        (name, i, 0, 0): (_block_fast_hist2d, xi, yi, bins, range)
        for i, (xi, yi) in enumerate(zip(x_keys, y_keys))
    }
    dtype = np.histogram2d([], [])[0].dtype

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(x, y))

    # turn graph into a 3D array of shape (nchunks, nbins, nbins)
    nchunks = len(list(flatten(x.__dask_keys__())))
    chunks = ((1,) * nchunks, (bins,), (bins,))
    mapped = Array(graph, name, chunks, dtype=dtype)
    n = mapped.sum(axis=0)
    return n
