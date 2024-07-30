import numpy as np


def nm_to_m(nm):
    """Convert nautical miles to meters"""
    return nm * 1852


def m_to_nm(m):
    """Convert meters to nautical miles"""
    return m / 1852


def mps_to_knots(mps):
    """Convert meters per second to knots"""
    return mps * (3600 / 1852)


def knots_to_mps(knots):
    """Convert knots to meters per second"""
    return knots * (1852 / 3600)


def knots_to_kmph(knots):
    """Convert knots to kilometers per hour"""
    return knots * 1.852


def kmph_to_knots(kmph):
    """Convert kilometers per hour to knots"""
    return kmph / 1.852

