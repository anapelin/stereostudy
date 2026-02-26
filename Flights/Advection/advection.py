#%% 
import pathlib
import sys
# sys.path.append('/data/common/dataiku/config/projects/MASK2FORMER_1/lib/python/contrai/')
import matplotlib.pyplot as plt
import pandas as pd

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.physics import units

from IPython.display import Image, display
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime, timedelta
from pathlib import Path

#%%



def perform_dry_advection(flight_data_file):
    # Advect and plot 
    plt.rcParams["figure.figsize"] = (10, 6)

    # Load flight data first to determine time bounds dynamically
    if flight_data_file.endswith('.parquet'):
        df = pd.read_parquet(flight_data_file)
    else:
        df = pd.read_csv(flight_data_file)
    df["time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    # Drop rows with invalid times
    df = df.dropna(subset=["time"])
    
    # Calculate time bounds from actual flight data with buffer for advection
    # Floor to hour and add 10 hour buffer after max time for advection
    time_min = pd.to_datetime(df["time"].min()).floor("h")
    time_max = pd.to_datetime(df["time"].max()).ceil("h") + pd.Timedelta(hours=10)
    
    # Convert to timezone-naive for ERA5 (ERA5 expects UTC without timezone info)
    if hasattr(time_min, 'tz_localize'):
        time_min = time_min.tz_localize(None) if time_min.tz is not None else time_min
    if hasattr(time_max, 'tz_localize'):
        time_max = time_max.tz_localize(None) if time_max.tz is not None else time_max
        
    time_bounds = (time_min, time_max)
    
    # More pressure levels to cover typical cruise altitudes (like reference implementation)
    pressure_levels = [400, 350, 300, 250, 225, 200, 175, 150]

    era5pl = ERA5(
        time=time_bounds,
        variables=Cocip.met_variables + Cocip.optional_met_variables,
        pressure_levels=pressure_levels,
    )
    era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables)

    met = era5pl.open_metdataset()
    rad = era5sl.open_metdataset()
    
    # Convert altitude from feet to meters (pycontrails requires meters)
    df["altitude"] = units.ft_to_m(df["altitude"])
    
    # Optional: Clip at 36,000 ft (10,972 m) to focus on cruise altitudes
    df["altitude"] = df["altitude"].clip(upper=units.ft_to_m(36000))
    
    # Convert time to timezone-naive (pycontrails expects timezone-naive UTC)
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)

    # Run DryAdvection parameters
    dt_integration = pd.Timedelta(minutes=2)
    max_age = pd.Timedelta(hours=4)

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 50.0,  # initial plume depth, [m]
        "width": 40.0,  # initial plume width, [m]
        "downselect_met": True,  # Optimize met data for each flight
        "include_source_in_output": True,  # Include original waypoints
    }

    # Process all flights
    all_results = []
    all_original_data = []  # Store original flight data for merging
    flight_ids = df["flight_id"].unique()
    
    print(f"Processing {len(flight_ids)} flights...")
    print(f"Met data time range: {time_bounds[0]} to {time_bounds[1]}")
    print(f"Met data pressure levels: {pressure_levels} hPa")
    
    errors_by_type = {}
    
    for idx, flight_id in enumerate(flight_ids):
        try:
            # Filter data for this flight
            flight_df = df[df["flight_id"] == flight_id].copy()
            
            # Add waypoint index to original data for merging later
            flight_df['waypoint'] = range(len(flight_df))
            
            # Ensure we have valid time data
            if flight_df.empty or len(flight_df) < 2:
                raise ValueError(f"Not enough data points")
            
            # Check for valid times
            if flight_df["time"].isna().any():
                raise ValueError(f"Contains invalid times")
            
            # Get aircraft type and callsign from the data
            aircraft_type = flight_df["typecode"].iloc[0] if "typecode" in flight_df.columns else "B738"
            callsign = flight_df["callsign"].iloc[0] if "callsign" in flight_df.columns else flight_id
            
            # Create a flight instance
            fl = Flight(flight_df, aircraft_type=aircraft_type, flight_id=flight_id, drop_duplicated_times=True)
            
            # CRITICAL: Create NEW DryAdvection instance per flight (like reference implementation)
            # Reusing the same instance can cause state issues between flights
            dry_adv = DryAdvection(met, params)
            
            # Run DryAdvection
            dry_adv_result = dry_adv.eval(fl)
            if dry_adv_result is not None:
                dry_adv_df = dry_adv_result.dataframe
                all_results.append(dry_adv_df)
                # Store original data for this flight
                all_original_data.append(flight_df)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(flight_ids)} flights - {len(all_results)} successful")
                
        except Exception as e:
            error_msg = str(e)
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            if idx < 10:  # Print first 10 errors for debugging
                print(f"Error processing flight {flight_id}: {e}")
            continue
    
    # Combine all results
    print(f"\n=== Processing Summary ===")
    print(f"Total flights: {len(flight_ids)}")
    print(f"Successfully processed: {len(all_results)}")
    print(f"Failed: {len(flight_ids) - len(all_results)}")

    # Show the format of the final dataframe - show the summary
    if all_results:
        sample_df = all_results[0]
        print(f"\nSample advection result columns: {sample_df.columns.tolist()}")
        print(f"Sample advection result preview:")
        print(sample_df.head())
    
    if errors_by_type:
        print(f"\nError breakdown:")
        for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    
    if all_results:
        # Combine all advection results
        combined_advection = pd.concat(all_results, ignore_index=True)
        
        # Combine all original flight data
        combined_original = pd.concat(all_original_data, ignore_index=True)
        
        # Merge original data with advection results on flight_id and waypoint
        # This creates one row for each advection point, with original flight data columns
        merged_df = pd.merge(
            combined_original,
            combined_advection,
            on=['flight_id', 'waypoint'],
            how='right',  # Keep all advection results
            suffixes=('_original', '_advection')
        )
        
        # Reorder columns: original columns first, then advection-specific columns
        original_cols = combined_original.columns.tolist()
        advection_only_cols = [col for col in combined_advection.columns if col not in original_cols]
        
        # Handle duplicate columns from merge (time, latitude, longitude, etc.)
        final_cols = []
        for col in original_cols:
            if col + '_original' in merged_df.columns:
                # Keep original version, rename back
                merged_df[col] = merged_df[col + '_original']
                final_cols.append(col)
            elif col in merged_df.columns:
                final_cols.append(col)
        
        # Update time to be initial waypoint time + contrail age
        if 'time' in merged_df.columns and 'age' in merged_df.columns:
            # Ensure time is datetime
            if not pd.api.types.is_datetime64_any_dtype(merged_df['time']):
                merged_df['time'] = pd.to_datetime(merged_df['time'])
            
            # Add age to the initial waypoint time
            merged_df['time'] = merged_df['time'] + merged_df['age']
            print(f"Updated time column: initial waypoint time + contrail age")
        
        # Add advection-specific columns
        for col in advection_only_cols:
            if col in merged_df.columns:
                final_cols.append(col)
        
        # Also add any _advection suffixed columns that might be useful
        for col in merged_df.columns:
            if col.endswith('_advection') and col not in final_cols:
                final_cols.append(col)
        
        # Select and reorder columns
        merged_df = merged_df[final_cols]
        
        print(f"\nMerged data shape: {merged_df.shape}")
        print(f"Final columns: {merged_df.columns.tolist()[:10]}... (showing first 10)")
        
        return merged_df, df
    else:
        print("\nNo flights were successfully processed")
        return None, df


def save_advection_results(dry_adv_df, flight_df, out_file, plot_file):
    '''
    The dry advection results contain the following: 'azimuth', 'width', 'depth', 'level', 'waypoint', 'area_eff', 'age','vertical_velocity', 
    'latitude', 'time', 'air_temperature', 'v_wind', 'flight_id', 'longitude', 'u_wind', 'air_pressure', 'sigma_yz'
    '''
    if dry_adv_df is None:
        print("No advection results to save")
        return
        
    # Check the columns in the dry advection results
    print(f"Advection results columns: {dry_adv_df.columns.tolist()}")
    print(f"Total advection points: {len(dry_adv_df)}")
    print(f"Unique flights in results: {dry_adv_df['flight_id'].nunique()}")

    # Save the dry advection results to a CSV file
    dry_adv_df.to_csv(out_file, index=False)
    print(f"Saved advection results to {out_file}")

    # Create visualization with all flights
    plt.figure(figsize=(12, 8))
    plt.scatter(flight_df["longitude"], flight_df["latitude"], s=1, color="red", alpha=0.5, label="Flight paths")
    plt.scatter(
        dry_adv_df["longitude"], dry_adv_df["latitude"], s=0.05, color="purple", alpha=0.3, label="Plume evolution"
    )
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Flight paths and plume evolution under dry advection\n{dry_adv_df['flight_id'].nunique()} flights processed")

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.show()

#%%

if __name__ == "__main__":
    date = datetime(2025, 4, 6)  

    # Define output directory
    output_dir = Path("/data/common/STEREOSTUDYIPSL/Flights/Advection/all")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    flight_data_file = "/data/common/STEREOSTUDYIPSL/Codebase/notebooks/2025-04-06_12-16.parquet"
    
    # Define search window
    date_time_start = datetime(2025, 4, 6, 4, 0)
    date_time_end = datetime(2025, 4, 6, 5, 0)

    # Perform dry advection on all flights
    print("Starting dry advection for all flights...")
    dry_adv_df, flight_df = perform_dry_advection(flight_data_file)

    # Save advection results 
    out_file = output_dir / "12-16_advected.csv"
    plot_file = output_dir / "12-16_advected.png"
    save_advection_results(dry_adv_df, flight_df, out_file, plot_file)

# %%
