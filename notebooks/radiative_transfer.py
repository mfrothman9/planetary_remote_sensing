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
    # Create fine grid for integration
    n_layers = 100  # Number of integration layers
    z_edges = np.linspace(z1, z2, n_layers+1)
    z_mid = (z_edges[1:] + z_edges[:-1])/2
    
    # Calculate impact parameter for the off-nadir path
    r1 = planet_radius + z1
    impact_parameter = r1 * np.sin(viewing_angle)
    
    # Initialize total optical depth
    total_tau = 0.0
    
    # Integrate through layers
    for i in range(n_layers):
        # Calculate path length through this layer using curved geometry
        ds = calc_path_segment_curved(z_edges[i], z_edges[i+1], 
                                    impact_parameter, planet_radius)
        
        # Get opacity at layer midpoint by interpolation
        opacity = np.interp(z_mid[i], altitude_grid, opacity_profile)
        
        # Add contribution to total optical depth
        total_tau += opacity * ds
        
    return total_tau

def cloud_thermal_brightness(surface_temperature: float,
                           wavelength: float, 
                           optical_depth: float,
                           single_scatter_albedo: float,
                           emission_angle: float,
                           cloud_temperature: float,
                           cloud_altitude: float,
                           planet_radius: float) -> float:
    """Calculate cloud brightness at thermal wavelengths.
    
    Calculates thermal emission from a thin cloud layer accounting for:
    1. Surface emission transmitted through cloud
    2. Thermal emission by cloud (proportional to tau*(1-omega))
    3. Surface emission scattered by cloud
    
    The curved geometry of the atmosphere is handled using integrate_opacity_nadir
    for calculating slant paths through the cloud layer.
    
    Args:
        surface_temperature: Surface temperature in Kelvin
        wavelength: Wavelength in meters
        optical_depth: Nadir optical depth of cloud
        single_scatter_albedo: Single scattering albedo (0-1)
        emission_angle: Viewing angle from nadir in radians
        cloud_temperature: Temperature of cloud layer in Kelvin
        cloud_altitude: Height of cloud layer above surface in meters
        planet_radius: Planet radius in meters
    
    Returns:
        Cloud brightness in W/m²/sr/m
        
    Notes:
        Key assumptions:
        - Optically thin cloud (tau << 1)
        - Cloud layer is isothermal
        - No atmospheric absorption above/below cloud
        - Cloud layer thickness << atmospheric scale height
        - Valid for both nadir and off-nadir viewing angles
        
        For optically thick clouds or cases violating these
        assumptions, a full multiple-scattering radiative 
        transfer calculation is required.
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
    """Calculate integrated opacity along a limb path.
    
    Integrates opacity along a path through a spherically symmetric atmosphere 
    characterized by an arbitrary vertical opacity profile. The path is defined 
    by its impact parameter (closest approach distance from planet center) and 
    start/end altitudes.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        z1: Start altitude of path (m)
        z2: End altitude of path (m) 
        impact_parameter: Closest approach distance from planet center (m)
        planet_radius: Planet radius (m)
    
    Returns:
        float: Integrated optical depth along limb path
        
    Notes:
        - Uses numerical integration with 100 layers
        - For each layer:
            1. Calculates geometric path length using calc_path_segment_curved
            2. Gets opacity at layer midpoint by interpolation
            3. Multiplies path length × opacity to get optical depth increment
        - Sums contributions from all layers
    """
    # Create fine grid for integration
    n_layers = 100
    z_edges = np.linspace(z1, z2, n_layers+1)
    z_mid = (z_edges[1:] + z_edges[:-1])/2
    
    # Initialize total optical depth
    total_tau = 0.0
    
    # Integrate through layers
    for i in range(n_layers):
        # Calculate path length through this layer
        ds = calc_path_segment_curved(z_edges[i], z_edges[i+1], 
                                    impact_parameter, planet_radius)
        
        # Get opacity at layer midpoint by interpolation
        opacity = np.interp(z_mid[i], altitude_grid, opacity_profile)
        
        # Add contribution to total optical depth
        total_tau += opacity * ds
        
    return total_tau

def surf_transmission(opacity_profile: np.ndarray,
                     angle: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray,
                     direction: str = 'from') -> float:
    """Calculate transmission along path to/from surface.
    
    Calculates the total transmission along a path between the surface and 
    spacecraft, accounting for atmospheric curvature. Can handle both upward
    and downward paths.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        angle: Surface incidence/emission angle (radians)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
        direction: Either 'to' (downward) or 'from' (upward) surface
    
    Returns:
        float: Total transmission along the path
        
    Raises:
        ValueError: If angle >= pi/2 or direction not 'to'/'from'
        
    Notes:
        - Uses integrate_opacity_nadir for path calculation
        - Returns exp(-tau) where tau is the total optical depth
        - Accounts for atmospheric curvature in path length calculation
    """
    if angle >= np.pi/2:
        raise ValueError("Angle must be less than pi/2 radians")
    if direction not in ['to', 'from']:
        raise ValueError("Direction must be 'to' or 'from'")
        
    # Calculate total optical depth along path
    tau = integrate_opacity_nadir(opacity_profile, altitude_grid,
                                0, spacecraft_alt, angle, planet_radius)
    
    # Return transmission
    return np.exp(-tau)

def limb_transmission(opacity_profile: np.ndarray,
                     tangent_alt: float,
                     spacecraft_alt: float,
                     planet_radius: float,
                     altitude_grid: np.ndarray) -> float:
    """Calculate transmission along limb path through curved atmosphere.
    
    Calculates total transmission along a limb-viewing path through a 
    spherically symmetric atmosphere. The path is characterized by its 
    tangent height (altitude of closest approach to planet surface) and
    extends from the spacecraft through the tangent point and back up
    to spacecraft altitude.
    
    Args:
        opacity_profile: Vertical profile of opacity per unit length (1/m)
        tangent_alt: Tangent altitude of line of sight (m)
        spacecraft_alt: Altitude of spacecraft (m)
        planet_radius: Planet radius (m)
        altitude_grid: Altitudes corresponding to opacity profile (m)
    
    Returns:
        float: Total transmission along the limb path
        
    Raises:
        ValueError: If tangent_alt > spacecraft_alt
        
    Notes:
        - Uses integrate_opacity_limb for path calculations
        - Takes advantage of path symmetry around tangent point
        - Impact parameter = planet_radius + tangent_alt
        - Returns exp(-tau) where tau is the total optical depth
    """
    if tangent_alt > spacecraft_alt:
        raise ValueError("Tangent altitude must be below spacecraft")
        
    # Calculate impact parameter
    impact_parameter = planet_radius + tangent_alt
    
    # Get optical depth by integrating from tangent point to spacecraft
    # Note: Due to symmetry, only integrate one side and multiply by 2
    tau = 2.0 * integrate_opacity_limb(opacity_profile, altitude_grid,
                                     tangent_alt, spacecraft_alt,
                                     impact_parameter, planet_radius)
    
    # Return transmission
    return np.exp(-tau)

def cloud_visible_brightness(surface_albedo: float,
                           solar_flux: float,
                           solar_zenith: float,
                           emission_angle: float, 
                           azimuth_angle: float,
                           optical_depth: float,
                           single_scatter_albedo: float,
                           asymmetry_parameter: float) -> float:
    """Calculate cloud brightness at visible wavelengths using single-scattering.
    
    Implements single-scattering approximation for cloud brightness where
    sunlight is scattered by cloud particles. Includes both direct scattered
    sunlight and surface-reflected light scattered by cloud.
    
    The radiance calculation implements:
    I = F₀μ₀ϖ₀P(g)/4π × [1 - exp(-τ/μ₀ - τ/μ)] + 
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
        float: Cloud brightness in W/m2/sr
        
    Notes:
        - Uses Henyey-Greenstein phase function:
          P(g) = (1 - g^2)/(1 + g^2 - 2g*cos(theta))^(3/2)
          where theta is the scattering angle
        - Scattering angle calculated using spherical trigonometry:
          cos(theta) = -μμ₀ + sqrt(1-μ²)sqrt(1-μ₀²)cos(ψ)
        - Includes two components:
          1. Direct scattered sunlight
          2. Surface-reflected light transmitted through cloud
    """
    # Calculate cosines of angles
    mu0 = np.cos(solar_zenith)    # Incident cosine
    mu = np.cos(emission_angle)    # Emission cosine
    
    # Calculate scattering angle using spherical trigonometry
    cos_theta = (-mu * mu0 + 
                np.sqrt(1 - mu**2) * np.sqrt(1 - mu0**2) * 
                np.cos(azimuth_angle))
    
    # Calculate Henyey-Greenstein phase function
    g = asymmetry_parameter
    phase = henyey_greenstein(cos_theta, g)
    
    # Calculate path optical depths
    tau_sun = optical_depth/mu0  # Solar path
    tau_view = optical_depth/mu  # Viewing path
    
    # Direct scattered sunlight
    direct_term = (solar_flux * mu0 * single_scatter_albedo * phase / 
                  (4 * np.pi) * (1 - np.exp(-tau_sun - tau_view)))
    
    # Surface-reflected light transmitted through cloud
    surface_term = (surface_albedo * solar_flux * mu0 / np.pi * 
                   np.exp(-tau_sun) * np.exp(-tau_view))
    
    return direct_term + surface_term

def henyey_greenstein(cos_theta, g):
    """Calculate Henyey-Greenstein phase function.
    
    Args:
        cos_theta: Cosine of scattering angle
        g: Asymmetry parameter (-1 to 1)
            g > 0: forward scattering
            g = 0: isotropic
            g < 0: back scattering
    """
    return (1 - g**2) / (1 + g**2 - 2*g*cos_theta)**(3/2)