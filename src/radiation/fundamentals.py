import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar
from scipy import optimize

def planck_function(self, wavelength, temperature):
    """ 
    Written by Claude.
    Calculate spectral radiance from Planck's law.
    Args:
        wavelength (float or array): Wavelength in meters
        temperature (float): Temperature in Kelvin

    Returns:
    float or array: Spectral radiance in W.m-2.sr-1.um-1
    """

    wavelength = np.asarray(wavelength)

    # Physical constants
    h = 6.62607015e-34  # Planck constant (J*s)
    c = 299792458  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (J/K)

    # Planck's law calculation
    numerator = (2 * h * c**2) / (wavelength**5)
    denominator = np.exp((h * c) / (wavelength * k * temperature)) - 1

    # Spectral radiance (W.m-2.sr-1.um-1)
    spectral_radiance = numerator / denominator

    #convert to W.m^-2.sr^-1.micron
    spectral_radiance *= 1e-6

    return spectral_radiance

def planck_inv(self, wavelength, spectral_radiance):
    """ 
    Written by Claude.
    Calculate temperature from spectral radiance using Planck's law.

    Args:
        wavelength (float): Wavelength in meters
        spectral_radiance (float): Spectral radiance in W.m-2.sr-1.um-1

    Returns:
        float: Temperature in Kelvin
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J*s)
    c = 299792458       # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)
    
    # Objective function to minimize
    def objective(T):
        numerator = (2 * h * c**2) / (wavelength**5)
        exponent = (h * c) / (wavelength * k * T)
        denominator = np.exp(exponent) - 1
        calculated_radiance = (numerator / denominator) * 1e-6
        
        return abs(calculated_radiance - spectral_radiance)
    
    # Solve using bounded optimization
    result = optimize.minimize_scalar(objective, bounds=(1, 10000), method='bounded')
    
    return result.x

def brightness_temperature(self, radiance, band_center, band_width):
    """ 
    Written by Claude.

    Calculate brightness temperature for a given rectangular bandpass.

    Implements numerical integration over a rectangular bandpass defined
    by its center wavelength and width. The bandpass is assumed to have
    unity transmission within its bounds and zero outside.

    Args:
        radiance (float): Observed radiance in W.m-2.sr-1.um-1
        band_center (float): Center wavelength of bandpass in meters
        band_width (float): Width of bandpass in meters

    Returns:
        float: Brightness temperature in Kelvin

    Raises:
        ValueError: If band_width <= 0 or band_center <= band_width /2
    """

    # Validate input
    if band_width <= 0:
        raise ValueError("Band width must be positive")
    if band_center <= band_width / 2:
        raise ValueError("Invalid band center and width")

    # Physical constants
    h = 6.626e-34  # Planck's constant (J·s)
    c = 3e8        # Speed of light (m/s)
    k = 1.380e-23  # Boltzmann constant (J/K)

    # Define integration bounds
    lower_wavelength = band_center - band_width / 2
    upper_wavelength = band_center + band_width / 2

    # Objective function for integration
    def brightness_temp_integrand(wavelength, temperature):
        numerator = (2 * h * c**2) / (wavelength**5)
        exponent = (h * c) / (wavelength * k * temperature)
        denominator = np.exp(exponent) - 1
        return (numerator / denominator) * 1e-6

    # Objective function to minimize
    def objective(T):
        integrated_radiance, _ = integrate.quad(
            brightness_temp_integrand, 
            lower_wavelength, 
            upper_wavelength, 
            args=(T,)
        )
        return abs(integrated_radiance - radiance)

    # Solve using bounded optimization
    result = minimize_scalar(objective, bounds=(1, 10000), method='bounded')
    
    return result.x

def radiance(temperature, band_center, band_width):
    """ 
    Written by Claude.
    Calculate band-integrated radiance for a given temperature and rectangular bandpass.

    Args:
        temperature (float): Temperature in Kelvin
        band_center (float): Center wavelength of bandpass in meters
        band_width (float): Width of bandpass in meters

    Returns:
        float: Band-integrated radiance in W.m-2.sr-1
    """
    # Validate input
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    if band_width <= 0:
        raise ValueError("Band width must be positive")
    if band_center <= band_width / 2:
        raise ValueError("Invalid band center and width")
    
    # Define integration bounds
    lower_wavelength = band_center - band_width / 2
    upper_wavelength = band_center + band_width / 2

    # Physical constants
    h = 6.62607015e-34  # Planck constant (J*s)
    c = 299792458       # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)

    # Integration function
    def planck_integrand(wavelength):
        numerator = (2 * h * c**2) / (wavelength**5)
        denominator = np.exp((h * c) / (wavelength * k * temperature)) - 1
        return (numerator / denominator) * 1e-6

    # Perform numerical integration
    integrated_radiance, _ = integrate.quad(
        planck_integrand, 
        lower_wavelength, 
        upper_wavelength
    )
    
    return integrated_radiance

def calculate_NEDT(self, temperature, NER, band_center, band_width):
    """ 
        Written by Claude.
        Calculate the noise-equivalent differential temperature (NEDT)
        for given scene temperature and noise-equivalent radiance (NER).

        Uses numerical derivative of band-integrated radiance with respect
        to temperature to determine the temperature uncertainty corresponding
        to the NER.

    Args:
        temperature (float): Scene temperature in Kelvin
        NER (float): Noise-equivalent radiance in W.m-2.sr-1
        band_center (float): Center wavelength of bandpass in meters
        band_width (float): Width of bandpass in meters

    Returns:
        float: NEDT in Kelvin

    Raises:
        ValueError: If temperature <= 0, NER <= 0, band_width <= 0,
        or band_center <= band_width/2
    """
    # Validate input parameters
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if NER <= 0:
        raise ValueError("Noise-equivalent radiance must be positive")
    if band_width <= 0:
        raise ValueError("Band width must be positive")
    if band_center <= band_width / 2:
        raise ValueError("Invalid band center and width")

    # Small temperature perturbation
    delta_T = 0.1  # Kelvin

    # Calculate radiance at current and perturbed temperatures
    R0 = self.radiance(temperature, band_center, band_width)
    R1 = self.radiance(temperature + delta_T, band_center, band_width)

    # Numerical derivative (radiance change per Kelvin)
    dR_dT = (R1 - R0) / delta_T

    # NEDT calculation
    NEDT = NER / abs(dR_dT)

    return NEDT




