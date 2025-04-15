import numpy as np
from scipy import constants as const
from scipy import integrate

class RadiationFundamentals:
    def __init__(self):
        # Physical constants
        self.h = const.h  # Planck constant
        self.c = const.c  # Speed of light
        self.k = const.k  # Boltzmann constant
        
    def planck_function(self, wavelength, temperature):
        """Calculate spectral radiance from Planck's law.
        
        Args:
            wavelength (float or array): Wavelength in meters
            temperature (float): Temperature in Kelvin
            
        Returns:
            float or array: Spectral radiance in W⋅m⁻²⋅sr⁻¹⋅m⁻¹
        """
        if np.any(wavelength <= 0):
            raise ValueError("Wavelength must be positive")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        # Calculate terms to avoid overflow
        c1 = 2.0 * self.h * self.c**2
        c2 = self.h * self.c / (wavelength * self.k * temperature)
        
        # Return spectral radiance
        return c1 / (wavelength**5 * (np.exp(c2) - 1.0)) / np.pi
    
    def planck_inv(self, wavelength, spectral_radiance):
        """Calculate temperature from spectral radiance using Planck's law.
        
        Args:
            wavelength (float): Wavelength in meters
            spectral_radiance (float): Spectral radiance in W⋅m⁻²⋅sr⁻¹⋅m⁻¹
            
        Returns:
            float: Temperature in Kelvin
        """
        if wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if spectral_radiance <= 0:
            raise ValueError("Spectral radiance must be positive")
            
        # Constants for Planck function
        c1 = 2.0 * self.h * self.c**2
        c2 = self.h * self.c / (wavelength * self.k)
        
        # Solve for temperature
        return c2 / np.log(c1 / (wavelength**5 * np.pi * spectral_radiance) + 1.0)
    
    def brightness_temperature(self, radiance, band_center, band_width):
        """Calculate brightness temperature for a given rectangular bandpass.
        
        Implements numerical integration over a rectangular bandpass defined
        by its center wavelength and width. The bandpass is assumed to have
        unity transmission within its bounds and zero outside.
        
        Args:
            radiance (float): Observed radiance in W⋅m⁻²⋅sr⁻¹
            band_center (float): Center wavelength of bandpass in meters
            band_width (float): Width of bandpass in meters
            
        Returns:
            float: Brightness temperature in Kelvin
            
        Raises:
            ValueError: If band_width <= 0 or band_center <= band_width/2
        """
        if band_width <= 0:
            raise ValueError("Band width must be positive")
        if band_center <= band_width/2:
            raise ValueError("Band center must be > band_width/2")
        if radiance <= 0:
            raise ValueError("Radiance must be positive")
            
        # Initial guess using Wien's displacement law
        # Wien's constant = hc/4.965k
        b_wien = (self.h * self.c)/(4.965 * self.k)
        T_guess = b_wien / band_center
        
        # Define function to find root of (measured - calculated radiance)
        def radiance_difference(T):
            return self.radiance(T, band_center, band_width) - radiance
        
        # Find brightness temperature using root finding
        from scipy.optimize import root_scalar
        result = root_scalar(radiance_difference, bracket=[1.0, 10000.0])
        
        if not result.converged:
            raise RuntimeError("Brightness temperature calculation did not converge")
            
        return result.root
    
    def radiance(self, temperature, band_center, band_width):
        """Calculate band-integrated radiance for a given temperature and
        rectangular bandpass.
        
        Integrates Planck function over a rectangular bandpass defined
        by its center wavelength and width. The bandpass is assumed to
        have unity transmission within its bounds and zero outside.
        
        Args:
            temperature (float): Temperature in Kelvin
            band_center (float): Center wavelength of bandpass in meters
            band_width (float): Width of bandpass in meters
            
        Returns:
            float: Band-integrated radiance in W⋅m⁻²⋅sr⁻¹
            
        Raises:
            ValueError: If temperature <= 0, band_width <= 0, or
                       band_center <= band_width/2
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if band_width <= 0:
            raise ValueError("Band width must be positive")
        if band_center <= band_width/2:
            raise ValueError("Band center must be > band_width/2")
            
        # Integration limits
        lambda_min = band_center - band_width/2
        lambda_max = band_center + band_width/2
        
        # Define integrand
        def planck_integrand(wavelength):
            return self.planck_function(wavelength, temperature)
        
        # Perform integration
        result = integrate.quad(planck_integrand, lambda_min, lambda_max)
        
        return result[0]
    
    def calculate_NEDT(self, temperature, NER, band_center, band_width):
        """Calculate the noise-equivalent differential temperature (NEDT)
        for given scene temperature and noise-equivalent radiance (NER).
        
        Uses numerical derivative of band-integrated radiance with respect 
        to temperature to determine the temperature uncertainty corresponding
        to the NER.
        
        Args:
            temperature (float): Scene temperature in Kelvin
            NER (float): Noise-equivalent radiance in W⋅m⁻²⋅sr⁻¹
            band_center (float): Center wavelength of bandpass in meters
            band_width (float): Width of bandpass in meters
            
        Returns:
            float: NEDT in Kelvin
            
        Raises:
            ValueError: If temperature <= 0, NER <= 0, band_width <= 0,
                       or band_center <= band_width/2
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if NER <= 0:
            raise ValueError("NER must be positive")
        if band_width <= 0:
            raise ValueError("Band width must be positive")
        if band_center <= band_width/2:
            raise ValueError("Band center must be > band_width/2")
            
        # Calculate derivative dL/dT using central difference
        delta_T = temperature * 0.001  # Small temperature difference
        
        L1 = self.radiance(temperature - delta_T, band_center, band_width)
        L2 = self.radiance(temperature + delta_T, band_center, band_width)
        
        dL_dT = (L2 - L1) / (2 * delta_T)
        
        # Calculate NEDT
        return NER / dL_dT