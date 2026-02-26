"""
Flight trajectory projection utilities using skycam.
"""
import sys
import datetime
import numpy as np
import pandas as pd
import pytz
from pathlib import Path

from config import (
    WORKSPACE_DIR, FLIGHT_CSV, ECTL_LAT, ECTL_LON, ECTL_ALT,
    PROJECTION_CLOUD_HEIGHT, PROJECTION_SQUARE_SIZE, PROJECTION_RESOLUTION
)


def setup_skycam():
    """
    Setup skycam library and return projector.
    
    Returns:
        AircraftProjector instance
    """
    # Add skycam to path
    betatesting_path = WORKSPACE_DIR / "Betatesting" / "skycam" / "src"
    if str(betatesting_path) not in sys.path:
        sys.path.insert(0, str(betatesting_path))
    
    from skycam.domain.aircraft_projection import (
        AircraftProjector, 
        AircraftProjectionSettings
    )
    
    # Create projection settings
    projection_settings = AircraftProjectionSettings(
        cloud_height=PROJECTION_CLOUD_HEIGHT,
        square_size=PROJECTION_SQUARE_SIZE,
        resolution=PROJECTION_RESOLUTION
    )
    
    # Initialize projector
    aircraft_projector = AircraftProjector(
        camera_lat=ECTL_LAT,
        camera_lon=ECTL_LON,
        camera_alt=ECTL_ALT,
        settings=projection_settings
    )
    
    print("✓ AircraftProjector initialized")
    print(f"  Location: {ECTL_LAT}°N, {ECTL_LON}°E @ {ECTL_ALT}m")
    print(f"  Projection plane: {PROJECTION_CLOUD_HEIGHT/1000:.1f}km altitude")
    print(f"  Coverage: {PROJECTION_SQUARE_SIZE/1000:.0f}km × "
          f"{PROJECTION_SQUARE_SIZE/1000:.0f}km")
    print(f"  Resolution: {PROJECTION_RESOLUTION}×{PROJECTION_RESOLUTION} pixels")
    
    return aircraft_projector, projection_settings


def load_flight_data(flight_csv_path: Path = FLIGHT_CSV):
    """
    Load and standardize flight data.
    
    Args:
        flight_csv_path: Path to flight CSV file
        
    Returns:
        DataFrame with standardized flight data
    """
    if not flight_csv_path.exists():
        print(f"⚠ Flight data file not found: {flight_csv_path}")
        return pd.DataFrame()
    
    flights_raw = pd.read_csv(flight_csv_path)
    print(f"✓ Loaded {len(flights_raw)} flight records")
    
    # Standardize column names
    col_mapping = {}
    for col in flights_raw.columns:
        col_lower = col.lower()
        if 'lat_mean' in col_lower:
            col_mapping[col] = 'latitude'
        elif 'lon_mean' in col_lower:
            col_mapping[col] = 'longitude'
        elif 'alt_mean' in col_lower:
            col_mapping[col] = 'altitude_m'
        elif 'first_seen' in col_lower:
            col_mapping[col] = 'timestamp'
        elif col in ['flight_id', 'callsign']:
            if 'flight_id' not in col_mapping.values():
                col_mapping[col] = 'flight_id'
    
    flights_df = flights_raw.rename(columns=col_mapping)
    
    # Convert altitude from feet to meters if needed
    if 'altitude_m' in flights_df.columns:
        median_alt = flights_df['altitude_m'].median()
        if median_alt > 20000:
            flights_df['altitude_m'] = flights_df['altitude_m'] * 0.3048
    
    # Parse timestamps
    if 'timestamp' in flights_df.columns:
        flights_df['timestamp'] = pd.to_datetime(flights_df['timestamp'], errors='coerce')
    
    print(f"✓ Flight data standardized")
    print(f"  Total records: {len(flights_df)}")
    if 'flight_id' in flights_df.columns:
        print(f"  Unique flights: {flights_df['flight_id'].nunique()}")
    
    return flights_df


def filter_flights_by_timestamp(flights_df: pd.DataFrame,
                                target_time: datetime.datetime,
                                time_window_minutes: int = 5,
                                min_altitude: float = 8000,
                                max_altitude: float = 12000):
    """
    Filter flights by timestamp and altitude.
    
    Args:
        flights_df: Flight data
        target_time: Target timestamp
        time_window_minutes: Time window in minutes (±)
        min_altitude: Minimum altitude in meters
        max_altitude: Maximum altitude in meters
        
    Returns:
        Filtered DataFrame
    """
    if 'timestamp' not in flights_df.columns:
        print("⚠ No timestamp column in flight data")
        return pd.DataFrame()
    
    # Make target_time timezone-aware (UTC)
    if target_time.tzinfo is None:
        target_time = pytz.UTC.localize(target_time)
    
    print(f"Filtering flights for timestamp: {target_time}")
    
    # Filter by time window
    time_window = datetime.timedelta(minutes=time_window_minutes)
    start_time = target_time - time_window
    end_time = target_time + time_window
    
    flights_filtered = flights_df[
        (flights_df['timestamp'] >= start_time) & 
        (flights_df['timestamp'] <= end_time)
    ].copy()
    
    print(f"  Found {len(flights_filtered)} flights in time window")
    
    # Filter by altitude
    if 'altitude_m' in flights_filtered.columns:
        flights_filtered = flights_filtered[
            (flights_filtered['altitude_m'] >= min_altitude) & 
            (flights_filtered['altitude_m'] <= max_altitude)
        ]
        print(f"  {len(flights_filtered)} flights at cruising altitude "
              f"({min_altitude/1000:.0f}-{max_altitude/1000:.0f}km)")
    
    return flights_filtered


def project_flights_to_pixels(flights_df: pd.DataFrame, 
                              aircraft_projector,
                              projection_settings):
    """
    Project flight coordinates to pixel coordinates.
    
    Args:
        flights_df: Flight data with lat/lon/altitude
        aircraft_projector: AircraftProjector instance
        projection_settings: Projection settings
        
    Returns:
        DataFrame with added pixel_x, pixel_y, visible columns
    """
    if len(flights_df) == 0:
        return flights_df
    
    print("\nProjecting flights to pixels...")
    
    # Extract coordinates
    lons = flights_df['longitude'].values
    lats = flights_df['latitude'].values
    alts = flights_df['altitude_m'].values
    
    # Vectorized projection
    px, py = aircraft_projector.lonlat_to_pixels(lons, lats, alts)
    
    # Add to dataframe
    flights_df = flights_df.copy()
    flights_df['pixel_x'] = px
    flights_df['pixel_y'] = py
    
    # Mark visible flights
    resolution = projection_settings.resolution
    flights_df['visible'] = (
        ~np.isnan(px) & ~np.isnan(py) &
        (px >= 0) & (px < resolution) &
        (py >= 0) & (py < resolution)
    )
    
    visible_count = flights_df['visible'].sum()
    print(f"✓ Projection complete")
    print(f"  Total flights: {len(flights_df)}")
    print(f"  Visible in camera: {visible_count}")
    if len(flights_df) > 0:
        print(f"  Visibility rate: {100*visible_count/len(flights_df):.1f}%")
    
    return flights_df
