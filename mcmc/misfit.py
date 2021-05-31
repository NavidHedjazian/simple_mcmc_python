# -*- coding: utf-8 -*-

"""
    Functions to calculate misfit between models
    @author: Navid Hedjazian
"""

import numpy as np


# @jitit
def sqmahalanobis(u, v, vi):
    """
    Taken from scipy source code with modification: do not calculate the
    square root.
    Computes the squared Mahalanobis distance between two 1-D arrays.
    The squared Mahalanobis distance between 1-D arrays `u` and `v`,
    is defined as
    .. math::
       (u-v) V^{-1} (u-v)^T
    where ``V`` is the covariance matrix.  Note that the argument `VI`
    is the inverse of ``V``.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    vi : ndarray
        The inverse of the covariance matrix.

    Returns
    -------
    mahalanobis : double
        The squared Mahalanobis distance between vectors `u` and `v`.
    """
    delta = u - v
    m = np.dot(np.dot(delta, vi), delta)
    return m


def _sqeuclidean(u, v):
    """
    Computes the squared Euclidean distance between two 1-D arrays.
    The squared Euclidean distance between `u` and `v` is defined as
    .. math::
       {||u-v||}_2^2.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.

    Returns
    -------
    sqeuclidean : double
        The squared Euclidean distance between vectors `u` and `v`.
    """
    u_v = u - v
    return np.dot(u_v, u_v)

