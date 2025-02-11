"""Radiative transfer functions for ASTR 5830.

This module provides functions for calculating radiative transfer through
planetary atmospheres, including transmission along arbitrary paths and
cloud scattering calculations.
"""

import numpy as np
from radiation_fundamentals import RadiationFundamentals
# Initialize radiation class
rad = RadiationFundamentals()

# Planck function
planck_function = rad.planck_function

def calc_path_segment_curved(z1: float, 
                           z2: float, 
                           impact_parameter: float,
                           planet_radius: float) -> float:
    """Calculate path length through a spherically symmetric atmospheric layer.
    
    Most basic geometric calculation - path length between two altitudes z1 and z2
    for a given impact parameter (closest approach distance from planet center).
    
    Args:
        z1, z2: Altitudes of segment endpoints in meters
        impact_parameter: Closest approach distance from planet center in meters
        planet_radius: Planet radius in meters
    
    Returns:
        Path length through segment in meters
    """
    r1 = planet_radius + z1
    r2 = planet_radius + z2
    
    # Handle case where path doesn't intersect layer
    if impact_parameter > min(r1, r2):
        return 0.0
        
    # Use law of cosines to find path length
    cos_theta1 = np.sqrt(1 - (impact_parameter/r1)**2)
    cos_theta2 = np.sqrt(1 - (impact_parameter/r2)**2)
    
    return abs(r1*cos_theta1 - r2*cos_theta2)

def integrate_opacity_nadir(opacity_profile: np.ndarray,
                          altitude_grid: np.ndarray,
                          z1: float,
                          z2: float,
                          viewing_angle: float,
                          planet_radius: float) -> float:
    """Calculate integrated opacity along nadir/off-nadir path.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        z1: Start altitude of path (m)
        z2: End altitude of path (m)
        viewing_angle: Angle from nadir (radians)
        planet_radius: Planet radius (m)
        
    Returns:
        Integrated optical depth along path
    """
    # Convert viewing angle to local path angle accounting for curvature
    r1 = planet_radius + z1
    r2 = planet_radius + z2
    
    # Use Snell's law in spherical geometry
    sin_alpha = (r1/r2) * np.sin(viewing_angle)
    alpha = np.arcsin(sin_alpha)
    
    # Calculate path length
    dz = z2 - z1
    ds = dz / np.cos(alpha)
    
    # Interpolate opacity to midpoint
    z_mid = (z1 + z2) / 2
    opacity = np.interp(z_mid, altitude_grid, opacity_profile)
    
    # Return optical depth increment
    return opacity * ds

def cloud_thermal_brightness(surface_temperature: float,
                           wavelength: float, 
                           optical_depth: float,
                           single_scatter_albedo: float,
                           emission_angle: float,
                           cloud_temperature: float,
                           cloud_altitude: float,
                           planet_radius: float) -> float:
    """Calculate cloud brightness at thermal wavelengths.
    
    [previous docstring content]
    """
    # Calculate surface and cloud emission using Planck function
    surface_radiance = planck_function(wavelength, surface_temperature)
    cloud_radiance = planck_function(wavelength, cloud_temperature)
    
    # Set up opacity profile for the cloud layer
    dz = 100  # meters
    altitude_grid = np.array([0, cloud_altitude-dz/2, cloud_altitude+dz/2])
    opacity_profile = np.array([0, optical_depth/dz, 0])
    
    # Get slant path optical depth through cloud
    tau = integrate_opacity_nadir(opacity_profile, altitude_grid,
                                cloud_altitude-dz/2, cloud_altitude+dz/2,
                                emission_angle, planet_radius)
    trans = np.exp(-tau)
    
    # Calculate cloud emissivity correctly
    cloud_emissivity = tau * (1 - single_scatter_albedo)
    
    # Components of thermal radiance:
    # 1. Surface emission transmitted through cloud
    surface_component = surface_radiance * trans
    
    # 2. Cloud thermal emission 
    emission_component = cloud_radiance * cloud_emissivity
    
    # 3. Surface emission scattered by cloud
    scatter_component = surface_radiance * tau * single_scatter_albedo
    
    # Total radiance
    return surface_component + emission_component + scatter_component

#------------------------------------------
# Functions to be implemented by students:
#------------------------------------------

def integrate_opacity_limb(opacity_profile: np.ndarray,
                         altitude_grid: np.ndarray,
                         z1: float,
                         z2: float,
                         impact_parameter: float,
                         planet_radius: float) -> float:
    """Written by Claude.
    Calculate integrated opacity along a limb path.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        z1: Start altitude of path (m)
        z2: End altitude of path (m) 
        impact_parameter: Closest approach distance from planet center (m)
        planet_radius: Planet radius (m)
    
    Returns:
        Integrated optical depth along limb path
        
    Notes:
        Use calc_path_segment_curved to compute geometric path lengths
    """

    # Initialize total optical depth
    total_tau = 0.0
    
    # Ensure z1 is lower than z2 for consistent processing
    if z1 > z2:
        z1, z2 = z2, z1
        
    # Find relevant altitude grid indices
    idx_start = np.searchsorted(altitude_grid, z1, side='right') - 1
    idx_end = np.searchsorted(altitude_grid, z2, side='right')
    
    # Ensure indices are within bounds
    idx_start = max(0, idx_start)
    idx_end = min(len(altitude_grid), idx_end)
    
    # Process each layer in the altitude grid between z1 and z2
    for i in range(idx_start, idx_end):
        # Define layer boundaries
        layer_bottom = max(altitude_grid[i], z1)
        layer_top = min(altitude_grid[i + 1] if i + 1 < len(altitude_grid) else z2, z2)
        
        # Calculate path length through this layer
        path_length = calc_path_segment_curved(
            layer_bottom,
            layer_top,
            impact_parameter,
            planet_radius
        )
        
        # Use average opacity for the layer
        layer_opacity = opacity_profile[i]
        
        # Add contribution from this layer
        total_tau += path_length * layer_opacity
        
    return total_tau

def surf_transmission(opacity_profile: np.ndarray,
                     angle: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray,
                     direction: str = 'from') -> float:
    """Written by Claude.
    Calculate transmission along path to/from surface.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        angle: Surface incidence/emission angle (radians)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        direction: Either 'to' (downward) or 'from' (upward) surface
    
    Returns:
        Total transmission along the path
        
    Raises:
        ValueError: If angle >= pi/2 or direction not 'to'/'from'
        
    Notes:
        Use integrate_opacity_nadir for calculations
    """
    # Validate inputs
    if angle >= np.pi/2:
        raise ValueError("Angle must be less than π/2")
    
    if direction not in ['to', 'from']:
        raise ValueError("Direction must be 'to' or 'from'")
    
    # Surface altitude is always zero
    surface_alt = 0.0
    
    # Determine path depending on direction
    if direction == 'to':
        # Path from spacecraft down to surface
        start_alt = spacecraft_alt
        end_alt = surface_alt
    else:  # 'from'
        # Path from surface up to spacecraft
        start_alt = surface_alt
        end_alt = spacecraft_alt
    
    # Calculate integrated opacity along the path
    tau = integrate_opacity_nadir(
        opacity_profile,
        altitude_grid,
        start_alt,
        end_alt,
        angle,
        planet_radius
    )
    
    # Return transmission using Beer-Lambert law
    return np.exp(-tau)

def limb_transmission(opacity_profile: np.ndarray,
                     tangent_alt: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray) -> float:
    """Calculate transmission along limb path through curved atmosphere.
    
    Uses integrate_opacity_limb to compute the total transmission along the
    limb viewing path, integrating from the spacecraft through the tangent
    point and back up to spacecraft altitude on the other side.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        tangent_alt: Tangent altitude of line of sight (m)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
    
    Returns:
        Total transmission along the limb path
        
    Raises:
        ValueError: If tangent_alt > spacecraft_alt
        
    Notes:
        Path is integrated symmetrically on both sides of tangent point.
        Use impact_parameter = planet_radius + tangent_alt for the limb
        geometry calculation.
    """
    #verifying the tangent altitude is above that of the spacecraft
    if tangent_alt > spacecraft_alt:
        raise ValueError("Tangent altitude must be smaller than the sc altitude")
    
    #calculating the impact parameter
    impact_parameter = planet_radius + tangent_alt

    #integrating tau for the limb
    tau = integrate_opacity_limb(opacity_profile,
                         altitude_grid,
                         spacecraft_alt,
                         tangent_alt,
                         impact_parameter,
                         planet_radius)
    
    #multiplying by 2 for the symmetric transmission
    total_tau = 2 * tau

    #returning the total transmission
    return np.exp(-total_tau)
    
def cloud_visible_brightness(surface_albedo: float,
                           solar_flux: float,
                           solar_zenith: float,
                           emission_angle: float, 
                           azimuth_angle: float,
                           optical_depth: float,
                           single_scatter_albedo: float,
                           asymmetry_parameter: float) -> float:
    """Written by Claude
    Calculate cloud brightness at visible wavelengths using single-scattering.
    
    Implements single-scattering approximation for cloud brightness where
    sunlight is scattered by cloud particles. Includes both direct scattered
    sunlight and surface-reflected light scattered by cloud.
    
    The radiance calculation should implement:
    L = F₀μ₀ϖ₀P(g)/4π × [1 - exp(-τ/μ₀ - τ/μ)] + 
        A_sF₀μ₀/π × exp(-τ/μ₀) × exp(-τ/μ)
    
    Args:
        surface_albedo: Surface Lambert albedo (0-1)
        solar_flux: Incident solar flux at top of atmosphere (W/m2)
        solar_zenith: Solar zenith angle (radians)
        emission_angle: Viewing angle from nadir (radians)
        azimuth_angle: Relative azimuth between sun and viewing direction (radians)
        optical_depth: Cloud optical depth
        single_scatter_albedo: Single scattering albedo (0-1) 
        asymmetry_parameter: Asymmetry parameter g (-1 to 1)
    
    Returns:
        Cloud brightness in W/m2/sr
        
    Notes:
        Use the Henyey-Greenstein phase function:
        P(g) = (1 - g^2)/(1 + g^2 - 2g*cos(theta))^(3/2)
        where theta is the scattering angle
    """
    
    # Calculate cosines of angles for convenience
    mu_0 = np.cos(solar_zenith)      # cosine of solar zenith angle
    mu = np.cos(emission_angle)       # cosine of emission angle
    
    # Calculate scattering angle using spherical trigonometry
    cos_scatter = (mu_0 * mu + 
                  np.sqrt(1 - mu_0**2) * np.sqrt(1 - mu**2) * 
                  np.cos(azimuth_angle))
    
    # Calculate Henyey-Greenstein phase function
    g = asymmetry_parameter
    phase_func = ((1 - g**2) / 
                 (1 + g**2 - 2*g*cos_scatter)**(3/2))
    
    # Calculate transmittances
    trans_sun = np.exp(-optical_depth/mu_0)    # solar beam transmission
    trans_view = np.exp(-optical_depth/mu)      # viewing path transmission
    
    # Calculate the direct scattered sunlight component
    # L = F₀μ₀ϖ₀P(g)/4π × [1 - exp(-τ/μ₀ - τ/μ)]
    direct_scatter = (solar_flux * mu_0 * single_scatter_albedo * phase_func / 
                     (4 * np.pi) * (1 - np.exp(-optical_depth/mu_0 - optical_depth/mu)))
    
    # Calculate the surface-reflected component
    # A_sF₀μ₀/π × exp(-τ/μ₀) × exp(-τ/μ)
    surface_component = (surface_albedo * solar_flux * mu_0 / np.pi * 
                        trans_sun * trans_view)
    
    # Return total brightness
    return direct_scatter + surface_component