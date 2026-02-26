"""
Process flight data from parquet files and filter for flights visible from camera locations.

This script extracts ADS-B flight data and filters for aircraft that pass through
the field of view of the SIRTA (IPSL) and Orsay (ECTL) cameras.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

# Camera positions (from params.csv)
CAMERA_POSITIONS = {
    'SIRTA': {  # IPSL camera
        'latitude': 48.713,
        'longitude': 2.207,
        'height_m': 177.5,
        'name': 'SIRTA (IPSL)'
    },
    'ECTL': {  # ECTL camera  
        'latitude': 48.600518087374105,
        'longitude':2.3467954996250784,
        'height_m': 90,
        'name': 'Bretigny (ECTL)'
    }
}

# Calculate center point between cameras
CENTER_LAT = (CAMERA_POSITIONS['SIRTA']['latitude'] + CAMERA_POSITIONS['ECTL']['latitude']) / 2
CENTER_LON = (CAMERA_POSITIONS['SIRTA']['longitude'] + CAMERA_POSITIONS['ECTL']['longitude']) / 2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def load_flight_data(parquet_path: str) -> pd.DataFrame:
    """
    Load flight data from parquet file.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        DataFrame with flight data
    """
    df = pd.read_parquet(parquet_path)  
    print(f"Loaded {len(df):,} flight records from {parquet_path}")
    print(f"Unique flights: {df['flight_id'].nunique():,}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def filter_flights_by_region(
    df: pd.DataFrame,
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    radius_km: float = 100.0,
    min_altitude_ft: float = 20000.0,
    max_altitude_ft: float = 45000.0
) -> pd.DataFrame:
    """
    Filter flights to those within a circular region around the cameras.
    
    Args:
        df: DataFrame with flight data
        center_lat: Center latitude of the region
        center_lon: Center longitude of the region
        radius_km: Radius of the circular region in km
        min_altitude_ft: Minimum altitude to include (feet)
        max_altitude_ft: Maximum altitude to include (feet)
        
    Returns:
        Filtered DataFrame
    """
    print(f"\nFiltering flights within {radius_km}km of center ({center_lat:.4f}, {center_lon:.4f})")
    print(f"Altitude range: {min_altitude_ft:,.0f} - {max_altitude_ft:,.0f} ft")
    
    # Calculate distance from center for each point
    distances = haversine_distance(
        df['latitude'].values,
        df['longitude'].values,
        center_lat,
        center_lon
    )
    
    # Apply filters
    mask = (
        (distances <= radius_km) &
        (df['altitude'] >= min_altitude_ft) &
        (df['altitude'] <= max_altitude_ft) &
        (df['latitude'].notna()) &
        (df['longitude'].notna())
    )
    
    df_filtered = df[mask].copy()
    df_filtered['distance_from_center_km'] = distances[mask]
    
    print(f"Records after filtering: {len(df_filtered):,} ({100*len(df_filtered)/len(df):.1f}%)")
    print(f"Unique flights in region: {df_filtered['flight_id'].nunique():,}")
    
    return df_filtered


def filter_flights_by_zenith(
    df: pd.DataFrame,
    max_zenith_deg: float = 80.0
) -> pd.DataFrame:
    """
    Filter flights by zenith angle (angle from vertical).
    Lower zenith = closer to directly overhead.
    
    Args:
        df: DataFrame with flight data
        max_zenith_deg: Maximum zenith angle in degrees (90 = horizon)
        
    Returns:
        Filtered DataFrame
    """
    max_zenith_rad = np.radians(max_zenith_deg)
    
    mask = df['zenith'] <= max_zenith_rad
    df_filtered = df[mask].copy()
    
    print(f"\nFiltering by zenith angle <= {max_zenith_deg}°")
    print(f"Records after zenith filter: {len(df):,}")
    print(f"Unique flights: {df['flight_id'].nunique():,}")
    
    return df


def filter_flights_by_time(
    df: pd.DataFrame,
    start_hour: int = 4,
    end_hour: int = 18
) -> pd.DataFrame:
    """
    Filter flights by time of day (UTC).
    
    Args:
        df: DataFrame with flight data
        start_hour: Start hour (UTC)
        end_hour: End hour (UTC)
        
    Returns:
        Filtered DataFrame
    """
    # Extract hour from timestamp
    hours = df['timestamp'].dt.hour
    
    mask = (hours >= start_hour) & (hours < end_hour)
    df_filtered = df[mask].copy()
    
    print(f"\nFiltering by time: {start_hour}:00 - {end_hour}:00 UTC")
    print(f"Records after time filter: {len(df_filtered):,}")
    print(f"Unique flights: {df_filtered['flight_id'].nunique():,}")
    
    return df_filtered


def get_flight_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each unique flight.
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        Summary DataFrame with one row per flight
    """
    summary = df.groupby('flight_id').agg({
        'timestamp': ['min', 'max'],
        'icao24': 'first',
        'registration': 'first',
        'typecode': 'first',
        'callsign': 'first',
        'latitude': ['min', 'max', 'mean'],
        'longitude': ['min', 'max', 'mean'],
        'altitude': ['min', 'max', 'mean'],
        'groundspeed': 'mean',
        'zenith': 'min',
        'azimuth': ['min', 'max'],
    }).reset_index()
    
    # Flatten column names
    summary.columns = [
        'flight_id', 'first_seen', 'last_seen', 'icao24', 'registration',
        'typecode', 'callsign', 'lat_min', 'lat_max', 'lat_mean',
        'lon_min', 'lon_max', 'lon_mean', 'alt_min', 'alt_max', 'alt_mean',
        'avg_groundspeed', 'min_zenith', 'azimuth_min', 'azimuth_max'
    ]
    
    # Calculate duration
    summary['duration_sec'] = (summary['last_seen'] - summary['first_seen']).dt.total_seconds()
    
    # Convert zenith to degrees
    summary['min_zenith_deg'] = np.degrees(summary['min_zenith'])
    
    return summary


def main():
    """Main processing function."""
    # File paths
    flights_dir = Path('/data/common/STEREOSTUDYIPSL/Flights')
    parquet_file = flights_dir / '2025-04-06.parquet'
    
    # Load data
    df = load_flight_data(parquet_file)
    
    # Print camera positions
    print("\n" + "="*60)
    print("CAMERA POSITIONS")
    print("="*60)
    for name, pos in CAMERA_POSITIONS.items():
        print(f"  {pos['name']}: ({pos['latitude']:.6f}, {pos['longitude']:.6f})")
    print(f"  Center point: ({CENTER_LAT:.6f}, {CENTER_LON:.6f})")
    
    # Apply filters
    print("\n" + "="*60)
    print("FILTERING FLIGHTS")
    print("="*60)
    
    # Filter by region (50km radius around cameras)
    df_region = filter_flights_by_region(
        df,
        radius_km=50.0,
        min_altitude_ft=25000.0,  # Contrails typically form above 25,000 ft
        max_altitude_ft=45000.0
    )
    
    # Filter by zenith angle (within camera field of view)
    df_zenith = filter_flights_by_zenith(df_region, max_zenith_deg=70.0)
    
    # Filter by time (5am to 6pm)
    df_time = filter_flights_by_time(df_zenith, start_hour=4, end_hour=18)
    
    # Get flight summary
    print("\n" + "="*60)
    print("FLIGHT SUMMARY")
    print("="*60)

    output_file = flights_dir / '2025-04-06_final.csv'
    df_time.to_csv(output_file, index=False)
    print(f"\n✓ Saved filtered data to {output_file}")
    
    if len(df_time) > 0:
        summary = get_flight_summary(df_time)
        
        # Sort by minimum zenith (closest to overhead first)
        summary = summary.sort_values('min_zenith_deg')
        
        print(f"\nTop 20 flights closest to overhead:")
        print("-"*100)
        cols = ['flight_id', 'callsign', 'typecode', 'first_seen', 'min_zenith_deg', 'alt_mean', 'duration_sec']
        print(summary[cols].head(20).to_string(index=False))
        
        # Save results
        output_file = flights_dir / '2025-04-06_filtered.parquet'
        df_time.to_parquet(output_file)
        print(f"\n✓ Saved filtered data to {output_file}")

        # output_file = flights_dir / '2025-04-06_filtered.csv'
        # df_time.to_csv(output_file, index=False)
        # print(f"\n✓ Saved filtered data to {output_file}")
        
        summary_file = flights_dir / '2025-04-06_summary.csv'
        summary.to_csv(summary_file, index=False)
        print(f"✓ Saved flight summary to {summary_file}")
        
        return df_time, summary
    else:
        print("No flights found matching criteria!")
        return None, None


if __name__ == '__main__':
    df_filtered, summary = main()
