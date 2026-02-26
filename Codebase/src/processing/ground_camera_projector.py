"""
File Name: ground_camera_projector.py
Author: Gabriel JARRY, Valentin TORDJMAN--LEVAVASSEUR, Philippe VERY
Contact Information: {gabriel.jarry, philippe.very}@eurocontrol.int
Date Created: July 2023
Last Modified : March 2024
Description : 
"""



import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import skimage 

import math
from geographiclib import geodesic
from typing import Optional, Tuple
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, griddata


class GroundCameraProjector:
    
    CAT_INFRARED = "infrarouge"
    CAT_VISIBLE = "visible"
    CALIBRATION = "calibration"
    JP2 = "jp2"
    NPY = "npy"
    AZIMUTH = "azimuth"
    ZENITH = "zenith"
    LON_LAT_DIST = "lon_lat_dist"


    
    # Constants representing the location of ECTL
    ECTL_LONGITUDE = 2.3467954996250784
    ECTL_LATITUDE = 48.600518087374105
    ECTL_HEIGHT = 90
    EARTH_RADIUS = 6371
    
    # Default values for initialization
    DEFAULT_MAX_ZENITH_ANGLE = 80
    DEFAULT_CLOUD_HEIGHT = 10000
    DEFAULT_RESOLUTION = 1024
    DEFAULT_SQUARE_SIZE = 75000

    def __init__(self, 
                 raw_dir: str= "/datasave/",
                 category: str= CAT_VISIBLE,
                 resolution: int = DEFAULT_RESOLUTION,
                 longitude: float = ECTL_LONGITUDE, 
                 latitude: float = ECTL_LATITUDE, 
                 height_above_ground: float = ECTL_HEIGHT, 
                 cloud_height: float = DEFAULT_CLOUD_HEIGHT,
                 square_size :int = DEFAULT_SQUARE_SIZE,
                 max_zenith_angle :int = DEFAULT_MAX_ZENITH_ANGLE,
                 #libs = {
                 #    "utils":"image_processing.utils"
                 #}
                ) -> None:
        
        # Store initialization parameters
        self.raw_dir = raw_dir
        self.category = category
        self.resolution = resolution
        self.longitude = longitude
        self.latitude = latitude
        self.height_above_ground = height_above_ground
        self.cloud_height = cloud_height
        self.square_size = square_size
        self.max_zenith_angle = max_zenith_angle
        
        # Load camera calibration data
        self.init_camera_calibration_data()
    
    
    def calculate_azimuth_zenith(self, latitude, longitude, altitude_m):
        # Define the WGS84 ellipsoid
        wgs84 = geodesic.Geodesic.WGS84

        # Compute the geodesic inverse problem between the two points
        inverse_coords = wgs84.Inverse(self.latitude, self.longitude, latitude, longitude)

        # Azimuth from the first point to the second point
        azimuth = inverse_coords['azi1']

        # Distance between points on the surface
        distance_on_surface = inverse_coords['s12']

        # Difference in altitude
        delta_altitude_m = altitude_m - self.height_above_ground

        # Calculate the straight-line distance using Pythagorean theorem
        straight_distance = math.sqrt(distance_on_surface**2 + delta_altitude_m**2)

        # Elevation angle
        elevation_angle = math.degrees(math.asin(delta_altitude_m / straight_distance))

        # Zenith angle is 90 degrees minus the elevation angle
        zenith = 90 - elevation_angle

        return azimuth, zenith

    def init_camera_calibration_data(self) -> None:
        """
        Initializes camera calibration data by loading it and setting up grid interpolation.
        """
        # Load camera calibration data
        self.load_camera_calibration_data()
        
        # Initialize grid interpolation with loaded data
        self.init_grid_interpolation()
        
        
    def load_camera_calibration_data(self) -> None:
        """
        Loads the camera calibration data for azimuth and zenith angles from specific files,
        applies necessary transformations, and initializes a restriction array.
        """
        # Load azimuth array from the calibration directory
        azimuth_filepath = os.path.join(self.raw_dir, self.CALIBRATION, ".".join([self.AZIMUTH + "_" + self.category, self.JP2]))
        # Convert azimuth values from degrees to radians and adjust range
        self.azimuth_array = 360 * np.pi / 180 * cv2.imread(azimuth_filepath, cv2.IMREAD_UNCHANGED) / 64000 - np.pi

        # Load zenith array from the calibration directory
        zenith_filepath = os.path.join(self.raw_dir, self.CALIBRATION, ".".join([self.ZENITH + "_" + self.category, self.JP2]))
        # Convert zenith values from degrees to radians
        self.zenith_array = 90 * np.pi / 180 * cv2.imread(zenith_filepath, cv2.IMREAD_UNCHANGED) / 64000

        # Store the size of the zenith array as image size
        self.image_size = self.zenith_array.shape

        # Create a restriction array to filter out angles greater than the maximum zenith angle
        self.restriction_array = np.where(self.zenith_array > self.max_zenith_angle * np.pi / 180, np.nan, 1)

        
    def init_grid_interpolation(self) -> None:
        """
        Initializes grid interpolation by creating a grid based on the square size and resolution,
        calculates zenith and azimuth angles for the interpolation grid, and sets up a LinearNDInterpolator
        to map azimuth and zenith angles to pixel coordinates.
        """
        # Calculate half of the square size and the step based on the resolution
        half_size = self.square_size / 2
        step = self.square_size / (self.resolution - 1)
        
        # Create arrays of x and y coordinates
        x = np.arange(-half_size, half_size + step, step)
        y = np.arange(-half_size, half_size + step, step)
        # Generate a meshgrid for x and y coordinates
        self.interpolation_grid_xy = np.meshgrid(x, y)
            
        # Calculate the radial distance from the center
        r = np.sqrt(self.interpolation_grid_xy[0]**2 + self.interpolation_grid_xy[1]**2)
        # Calculate zenith angle based on the radial distance
        self.interpolation_zenith = np.arctan(r / self.cloud_height)
        # Calculate azimuth angle from x and y coordinates
        self.interpolation_azimuth = np.arctan2(self.interpolation_grid_xy[1], self.interpolation_grid_xy[0])
        # Adjust azimuth angle to the correct range
        self.interpolation_azimuth = (self.interpolation_azimuth - 3*np.pi/2) % (2*np.pi) - np.pi
        # Combine azimuth and zenith angles into a single array
        self.azimuth_zenith_grid = np.stack([self.interpolation_azimuth, self.interpolation_zenith], axis=-1)    
        
        coordinates = np.array([ [(i, self.resolution-1-j) for i in range(self.resolution)] for j in range(self.resolution)])
        
        # Initialize the LinearNDInterpolator to map from azimuth-zenith to pixel coordinates in the projected image
        self.azimuth_zenith_to_pixel_proj = LinearNDInterpolator(
            self.azimuth_zenith_grid.reshape((self.resolution * self.resolution, 2)), 
            coordinates.reshape((self.resolution * self.resolution, 2))
        )
        
        self.pixel_proj_to_azimuth_zenith = LinearNDInterpolator(
            coordinates.reshape((self.resolution * self.resolution, 2)), 
            self.azimuth_zenith_grid.reshape((self.resolution * self.resolution, 2)))
        
        # Flatten and apply restriction array to azimuth and zenith arrays
        flattened_azimuth = (self.restriction_array * self.azimuth_array).flatten()
        flattened_zenith = (self.restriction_array * self.zenith_array).flatten()
        
        # Identify NaN values and create a combined mask
        mask_nan_azimuth = np.isnan(flattened_azimuth)
        mask_nan_zenith = np.isnan(flattened_zenith)
        self.mask_combined_nan = mask_nan_azimuth | mask_nan_zenith
        
        # Filter out NaN values from azimuth and zenith arrays
        filtered_azimuth = flattened_azimuth[~self.mask_combined_nan]
        filtered_zenith = flattened_zenith[~self.mask_combined_nan]
        
        # Combine filtered azimuth and zenith angles
        self.azimuth_zenith = np.stack([filtered_azimuth, filtered_zenith], axis=-1) 
        
        # Create coordinate grid and filter based on NaN mask
        x_r = np.arange(self.image_size[0])
        y_r = np.arange(self.image_size[1])
        grid = np.meshgrid(x_r, y_r)
        self.coordinates = np.stack([grid[0].T.flatten()[~self.mask_combined_nan], grid[1].T.flatten()[~self.mask_combined_nan]], axis=-1)

        # Initialize the LinearNDInterpolator to map from azimuth-zenith to pixel coordinates in the raw image
        self.azimuth_zenith_to_pixel_raw = LinearNDInterpolator(self.azimuth_zenith, self.coordinates)
        

    def calculate_latitude_longitude(self, azimuth, zenith, target_altitude_m):
        
        wgs84 = geodesic.Geodesic.WGS84

        # Convert zenith to elevation angle
        elevation_angle_deg = 90 - zenith

        # Convert to radians for trigonometric functions
        elevation_angle_rad = math.radians(elevation_angle_deg)

        # Altitude difference
        delta_altitude_m = target_altitude_m - self.height_above_ground

        # Distance on the surface
        # distance_on_surface = delta_altitude_m * cot(elevation_angle)
        # cot(x) = 1 / tan(x)
        distance_on_surface = delta_altitude_m / math.tan(elevation_angle_rad)

        # Solve the forward geodesic problem
        direct_result = wgs84.Direct(self.latitude, self.longitude, azimuth, distance_on_surface)

        target_lat = direct_result['lat2']
        target_lon = direct_result['lon2']

        return target_lat, target_lon
        
    def project_image(self, img_array: np.ndarray, uint8: bool =True) -> np.ndarray:
        """
        Projects an input image onto the azimuth-zenith grid using interpolation,
        effectively re-mapping the image based on the camera's calibration data.

        Parameters:
        - img_array (np.ndarray): The input image array to be projected.

        Returns:
        - np.ndarray: The projected image as a numpy array.
        """
        # Generate arrays of x and y coordinates based on the input image dimensions
        x_r = np.arange(img_array.shape[0])
        y_r = np.arange(img_array.shape[1])
        # Initialize RegularGridInterpolator with the input image to interpolate RGB values
        self.pixel_to_rgb = RegularGridInterpolator((x_r, y_r), img_array, bounds_error=False, fill_value=0)
        
        # Use the azimuth-zenith to pixel mapping to generate the projected grid coordinates
        self.projected_grid = self.azimuth_zenith_to_pixel_raw(self.azimuth_zenith_grid)
        # Interpolate RGB values for the projected grid and convert to unsigned integers
        projected_image = self.pixel_to_rgb(self.projected_grid)
        if uint8:
            return projected_image.astype(np.uint8)
        else:
            return projected_image