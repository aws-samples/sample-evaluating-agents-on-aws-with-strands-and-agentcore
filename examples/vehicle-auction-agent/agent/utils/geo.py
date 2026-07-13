# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Geographic utilities for distance calculations.

Provides Haversine distance and bounding-box pre-filter constants
used by filter_by_distance tools across the project.
"""

import math

import numpy as np

# Earth radius in miles (WGS-84 mean radius)
EARTH_RADIUS_MILES = 3958.8

# Earth radius in kilometres
EARTH_RADIUS_KM = 6371.0

# Approximate miles per degree of latitude (constant across the globe)
MILES_PER_DEGREE_LAT = 69.0

# Approximate kilometres per degree of latitude
KM_PER_DEGREE_LAT = 111.0

# Floor for cos(latitude) to avoid division by zero near the poles
COS_FLOOR = 0.01


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> float:
    """Core Haversine formula returning great-circle distance for the given Earth radius."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points.

    Args:
        lat1: Latitude of point 1 in degrees
        lon1: Longitude of point 1 in degrees
        lat2: Latitude of point 2 in degrees
        lon2: Longitude of point 2 in degrees

    Returns:
        Distance in miles
    """
    return _haversine(lat1, lon1, lat2, lon2, EARTH_RADIUS_MILES)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points.

    Args:
        lat1: Latitude of point 1 in degrees
        lon1: Longitude of point 1 in degrees
        lat2: Latitude of point 2 in degrees
        lon2: Longitude of point 2 in degrees

    Returns:
        Distance in kilometres
    """
    return _haversine(lat1, lon1, lat2, lon2, EARTH_RADIUS_KM)


def bounding_box_miles(latitude: float, radius_miles: float) -> tuple[float, float]:
    """Compute lat/lon deltas for a bounding-box pre-filter in miles.

    Args:
        latitude: Center latitude in degrees
        radius_miles: Radius in miles

    Returns:
        Tuple of (lat_delta, lon_delta) in degrees
    """
    lat_delta = radius_miles / MILES_PER_DEGREE_LAT
    lon_delta = radius_miles / (
        MILES_PER_DEGREE_LAT * max(math.cos(math.radians(latitude)), COS_FLOOR)
    )
    return lat_delta, lon_delta


def haversine_miles_vectorized(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized great-circle distance in miles from a single point to N points.

    Args:
        lat1: Reference latitude in degrees (scalar)
        lon1: Reference longitude in degrees (scalar)
        lat2: Target latitudes in degrees (array of N)
        lon2: Target longitudes in degrees (array of N)

    Returns:
        Array of distances in miles (shape N)
    """
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_MILES * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bounding_box_km(latitude: float, radius_km: float) -> tuple[float, float]:
    """Compute lat/lon deltas for a bounding-box pre-filter in kilometres.

    Args:
        latitude: Center latitude in degrees
        radius_km: Radius in kilometres

    Returns:
        Tuple of (lat_delta, lon_delta) in degrees
    """
    lat_delta = radius_km / KM_PER_DEGREE_LAT
    lon_delta = radius_km / (KM_PER_DEGREE_LAT * max(math.cos(math.radians(latitude)), COS_FLOOR))
    return lat_delta, lon_delta
