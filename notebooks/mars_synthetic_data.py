"""Generate synthetic Mars observations for remote sensing retrieval.

This module provides a class to generate synthetic radiance data for Mars
atmospheric dust observations in both nadir and limb viewing geometries.
All radiances are band-integrated over instrument spectral response.

Author: Paul O. Hayne
Affiliation: CU Boulder / LASP
"""

import numpy as np
import matplotlib.pyplot as plt
from radiative_transfer import (cloud_visible_brightness, integrate_opacity_nadir,
                              integrate_opacity_limb, calc_path_segment_curved,
                              henyey_greenstein)
from radiation_fundamentals import RadiationFundamentals

# Physical constants
AU = 1.496e11  # meters
class MarsSyntheticData:
    """Generate synthetic Mars observations in both nadir and limb geometry."""
    def __init__(self, 
             # Dust optical properties  
             Q_ext_vis: float = 2.7,      # Extinction efficiency at visible wavelength
             Q_ext_ir: float = 0.8,       # Extinction efficiency at IR wavelength
             single_scatter_albedo_vis: float = 0.97,  # Visible single scattering albedo
             single_scatter_albedo_ir: float = 0.5,   # IR single scattering albedo
             asymmetry_parameter_vis: float = 0.7,     # Visible asymmetry parameter g
             asymmetry_parameter_ir: float = 0.5,     # IR asymmetry parameter g
             
             # Instrument properties
             vis_center: float = 0.7e-6,  # Visible channel center wavelength (m)
             vis_width: float = 0.2e-6,   # Visible channel bandwidth (m) 
             ir_center: float = 12.0e-6,  # IR channel center wavelength (m)
             ir_width: float = 2.0e-6,    # IR channel bandwidth (m)
             
             # Atmospheric properties
             H_dust: float = 10e3,       # Dust scale height (m)
             z_max: float = 50e3,        # Max altitude in meters
             n_layers: int = 51,          # Number of atmospheric layers
             spacecraft_alt: float = 300e3,  # Spacecraft altitude in meters
             planet_radius: float = 3396e3,    # Mars radius in meters
             
             # Solar parameters
             mars_distance: float = 1.52*AU,  # Mars orbital distance
             solar_zenith: float = 0.0,        # Local noon by default
             
             # Temperature profile parameters
             surface_temp: float = 240.0,  # Surface temperature in K
             lapse_rate: float = -2.5e-3,  # Temperature lapse rate in K/m
             min_temp: float = 150.0       # Minimum temperature in K
             ):
        """Initialize atmospheric grid and basic properties."""
        # Initialize radiation calculator
        self.rad = RadiationFundamentals()
        
        # Set up atmospheric grid
        self.z_grid = np.linspace(0, z_max, n_layers)
        self.spacecraft_alt = spacecraft_alt
        self.planet_radius = planet_radius
        
        # Store dust properties
        self.H_dust = H_dust
        self.Q_ext_vis = Q_ext_vis
        self.Q_ext_ir = Q_ext_ir
        self.vis_to_ir = self.Q_ext_ir/self.Q_ext_vis
        
        # Store scattering properties separately for visible and IR
        self.single_scatter_albedo_vis = single_scatter_albedo_vis
        self.single_scatter_albedo_ir = single_scatter_albedo_ir
        self.asymmetry_parameter_vis = asymmetry_parameter_vis
        self.asymmetry_parameter_ir = asymmetry_parameter_ir
        
        # Store instrument properties
        self.vis_center = vis_center
        self.vis_width = vis_width
        self.ir_center = ir_center
        self.ir_width = ir_width
        
        # Calculate solar spectral flux
        solar_spectral_radiance = self.rad.radiance(5778.0,  # Sun's temperature
                                                   self.vis_center,
                                                   self.vis_width)
        R_sun = 6.957e8  # Solar radius in meters
        omega_sun = np.pi * (R_sun/mars_distance)**2  # Solid angle of Sun from Mars
        self.solar_spectral_flux = solar_spectral_radiance * omega_sun
        
        # Store solar geometry
        self.solar_zenith = solar_zenith
        
        # Calculate temperature profile
        self.T_profile = surface_temp + lapse_rate * self.z_grid
        self.T_profile[self.T_profile < min_temp] = min_temp
        
    def generate_nadir_obs(self,
                        dust_tau_true: float,
                        surface_albedo_true: float,
                        surface_temp: float,
                        emission_angles: np.ndarray,
                        noise_visible: float = 0.05,
                        noise_ir: float = 0.05,
                        n_samples: int = 100) -> tuple:
        """Generate synthetic nadir-viewing observations."""
        
        # Calculate dust opacity profile (m^-1)
        self.opacity_profile = (dust_tau_true/self.H_dust) * np.exp(-self.z_grid/self.H_dust)
        dust_opacity_ir = self.opacity_profile * self.vis_to_ir

        # Initialize arrays
        vis_radiance = np.zeros((n_samples, len(emission_angles)))
        ir_radiance = np.zeros((n_samples, len(emission_angles)))
        
        for i, angle in enumerate(emission_angles):
            # Visible calculation - use visible scattering properties
            vis_radiance[:,i] = cloud_visible_brightness(
                surface_albedo_true,
                solar_flux=self.solar_spectral_flux,
                solar_zenith=self.solar_zenith,
                emission_angle=angle,
                azimuth_angle=0.0,
                optical_depth=dust_tau_true,
                single_scatter_albedo=self.single_scatter_albedo_vis,  # Use visible properties
                asymmetry_parameter=self.asymmetry_parameter_vis
            )
            
            # IR calculation layer by layer
            for j in range(len(self.z_grid)-1):
                # Layer boundaries
                z_bot = self.z_grid[j]
                z_top = self.z_grid[j+1]
                z_mid = (z_bot + z_top)/2
                T_layer = self.T_profile[j]
                
                # Calculate layer optical depth
                tau_layer = integrate_opacity_nadir(dust_opacity_ir, self.z_grid,
                                                z_bot, z_top, angle, 
                                                self.planet_radius)
                
                # Calculate transmission from layer to space
                tau_above = integrate_opacity_nadir(dust_opacity_ir, self.z_grid,
                                                z_top, self.z_grid[-1], 
                                                angle, self.planet_radius)
                trans_above = np.exp(-tau_above)
                
                # Add layer contribution using IR scattering properties
                layer_emission = self.rad.radiance(T_layer, self.ir_center, self.ir_width)
                emissivity = (1 - self.single_scatter_albedo_ir) * (1 - np.exp(-tau_layer))
                layer_radiance = layer_emission * emissivity * trans_above
                
                # Add scattering contribution
                phase = henyey_greenstein(np.cos(np.pi - angle), self.asymmetry_parameter_ir)
                scatter_radiance = layer_emission * self.single_scatter_albedo_ir * phase/(4*np.pi) * tau_layer * trans_above
                
                ir_radiance[:,i] += layer_radiance + scatter_radiance

            # Calculate surface contribution with IR properties
            tau_total = integrate_opacity_nadir(dust_opacity_ir, self.z_grid,
                                            0, self.z_grid[-1], angle,
                                            self.planet_radius)
            surface_radiance = self.rad.radiance(surface_temp, self.ir_center, self.ir_width)
            surface_contribution = surface_radiance * np.exp(-tau_total)
            ir_radiance[:,i] += surface_contribution
        
        # Add noise
        vis_noise = np.random.normal(0, noise_visible, (n_samples, len(emission_angles)))
        ir_noise = np.random.normal(0, noise_ir, (n_samples, len(emission_angles)))
        
        vis_radiance_noisy = vis_radiance * (1 + vis_noise)
        ir_radiance_noisy = ir_radiance * (1 + ir_noise)
        
        return vis_radiance_noisy, ir_radiance_noisy, emission_angles, {}

    def generate_limb_obs(self,
                        dust_tau_true: float,
                        tangent_heights: np.ndarray,
                        surface_temp: float,
                        noise_visible: float = 0.05,
                        noise_ir: float = 0.05,
                        n_samples: int = 100) -> tuple:
        """Generate synthetic limb-viewing observations."""
        
        # Calculate dust opacity profile (m^-1)
        self.opacity_profile = (dust_tau_true/self.H_dust) * np.exp(-self.z_grid/self.H_dust)
        dust_opacity_ir = self.opacity_profile * self.vis_to_ir
        
        # Initialize arrays
        vis_radiance = np.zeros((n_samples, len(tangent_heights)))
        ir_radiance = np.zeros((n_samples, len(tangent_heights)))
        
        for i, h_tan in enumerate(tangent_heights):
            # Calculate impact parameter
            b = self.planet_radius + h_tan
            
            # Calculate IR emission first
            tau_ir = integrate_opacity_limb(dust_opacity_ir, self.z_grid,
                                        h_tan, self.spacecraft_alt,
                                        b, self.planet_radius)
            trans_ir = np.exp(-tau_ir)
            
            # Layer emission at tangent point using IR properties
            T_layer = np.interp(h_tan, self.z_grid, self.T_profile)
            layer_emission = self.rad.radiance(T_layer, self.ir_center, self.ir_width)
            emissivity = (1 - self.single_scatter_albedo_ir) * (1 - trans_ir)
            ir_radiance[:,i] = layer_emission * emissivity
            
            # Add scattering contribution - integrate along path
            r_tan = b  # Radius at tangent point
            r_sc = self.planet_radius + self.spacecraft_alt
            r_edges = np.linspace(r_tan, r_sc, 100)
            dr = r_edges[1] - r_edges[0]
            
            for j, r in enumerate(r_edges[:-1]):
                z = r - self.planet_radius
                local_opacity = np.interp(z, self.z_grid, self.opacity_profile)
                
                # Calculate scattering angle
                sin_beta = b/r
                if sin_beta > 1.0:
                    sin_beta = 1.0
                beta = np.arcsin(sin_beta)
                scatter_angle = np.pi - beta
                
                # Use appropriate scattering properties for each wavelength
                ds = calc_path_segment_curved(z, z + dr, b, self.planet_radius)
                
                # Visible scattering using visible properties
                phase_vis = henyey_greenstein(np.cos(scatter_angle), self.asymmetry_parameter_vis)
                vis_contribution = (self.solar_spectral_flux * 
                                self.single_scatter_albedo_vis * phase_vis/(4*np.pi) * 
                                local_opacity * ds)
                
                # IR scattering using IR properties
                phase_ir = henyey_greenstein(np.cos(scatter_angle), self.asymmetry_parameter_ir)
                ir_contribution = (layer_emission * 
                                self.single_scatter_albedo_ir * phase_ir/(4*np.pi) * 
                                local_opacity * self.vis_to_ir * ds)
                
                vis_radiance[:,i] += vis_contribution
                ir_radiance[:,i] += ir_contribution
        
        # Add noise
        vis_noise = np.random.normal(0, noise_visible, (n_samples, len(tangent_heights)))
        ir_noise = np.random.normal(0, noise_ir, (n_samples, len(tangent_heights)))
        
        vis_radiance_noisy = vis_radiance * (1 + vis_noise)
        ir_radiance_noisy = ir_radiance * (1 + ir_noise)
        
        return vis_radiance_noisy, ir_radiance_noisy, tangent_heights, {}
    
# Example usage
if __name__ == '__main__':
    # Initialize synthetic data generator
    mars_obs = MarsSyntheticData()
    
    # Generate nadir observations
    angles = np.radians([0, 30, 45, 60])
    vis_rad, ir_rad, angles, meta_nadir = mars_obs.generate_nadir_obs(
        dust_tau_true=0.5,
        surface_albedo_true=0.25,
        surface_temp=250,
        emission_angles=angles
    )
    
    # Generate limb observations
    heights = np.linspace(0, 40000, 20)
    vis_limb, ir_limb, heights, meta_limb = mars_obs.generate_limb_obs(
        dust_tau_true=0.5,
        tangent_heights=heights,
        surface_temp=250
    )
    
    print(f"Generated {len(vis_rad)} nadir samples at {len(angles)} angles")
    print(f"Generated {len(vis_limb)} limb samples at {len(heights)} heights")

# Define common parameters used throughout
angles = np.radians(np.arange(0,89,5))  # Emission angles for nadir
heights = np.arange(0, 40e3, 1e3)        # Tangent heights for limb
surface_temp = 240                         # Surface temperature in K

def plot_comparison(obs1, obs2, label1, label2, title):
    """Helper function to plot and compare two sets of observations
    
    Args:
        obs1, obs2: MarsSyntheticData instances with different parameters
        label1, label2: Plot labels for each case
        title: Overall plot title
    """
    try:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title)

        # Noise levels
        noise_visible = 0.02
        noise_ir = 0.02
        
        # Generate observations with noise
        vis_rad1, ir_rad1, angles_out, _ = obs1.generate_nadir_obs(
            dust_tau_true=0.4,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        vis_rad2, ir_rad2, angles_out, _ = obs2.generate_nadir_obs(
            dust_tau_true=0.4,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        vis_limb1, ir_limb1, heights_out, _ = obs1.generate_limb_obs(
            dust_tau_true=0.4,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        vis_limb2, ir_limb2, heights_out, _ = obs2.generate_limb_obs(
            dust_tau_true=0.4,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )

        # Normalize each sample to its nadir value
        vis_rad1_norm = vis_rad1 / vis_rad1[:,[0]]  # Broadcasting for normalization
        vis_rad2_norm = vis_rad2 / vis_rad2[:,[0]]
        ir_rad1_norm = ir_rad1 / ir_rad1[:,[0]]
        ir_rad2_norm = ir_rad2 / ir_rad2[:,[0]]

        # Calculate means and standard deviations
        vis_rad1_mean = np.mean(vis_rad1_norm, axis=0)
        vis_rad1_std = np.std(vis_rad1_norm, axis=0)
        vis_rad2_mean = np.mean(vis_rad2_norm, axis=0)
        vis_rad2_std = np.std(vis_rad2_norm, axis=0)

        ir_rad1_mean = np.mean(ir_rad1_norm, axis=0)
        ir_rad1_std = np.std(ir_rad1_norm, axis=0)
        ir_rad2_mean = np.mean(ir_rad2_norm, axis=0)
        ir_rad2_std = np.std(ir_rad2_norm, axis=0)

        # Plot with error bars
        ax[0,0].errorbar(np.degrees(angles_out), vis_rad1_mean, yerr=vis_rad1_std, 
                        fmt='o-', label=label1, capsize=3)
        ax[0,0].errorbar(np.degrees(angles_out), vis_rad2_mean, yerr=vis_rad2_std, 
                        fmt='s--', label=label2, capsize=3)
        ax[0,0].set_xlabel('Emission Angle (degrees)')
        ax[0,0].set_ylabel('Normalized Radiance')
        ax[0,0].legend()
        ax[0,0].set_title('Nadir Visible Radiance')

        ax[0,1].errorbar(np.degrees(angles_out), ir_rad1_mean, yerr=ir_rad1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[0,1].errorbar(np.degrees(angles_out), ir_rad2_mean, yerr=ir_rad2_std,
                        fmt='s--', label=label2, capsize=3)
        ax[0,1].set_xlabel('Emission Angle (degrees)')
        ax[0,1].set_ylabel('Normalized Radiance')
        ax[0,1].legend()
        ax[0,1].set_title('Nadir IR Radiance')

        # Normalize limb observations
        vis_limb1_norm = vis_limb1 / vis_limb1[:,[0]]
        vis_limb2_norm = vis_limb2 / vis_limb2[:,[0]]
        ir_limb1_norm = ir_limb1 / ir_limb1[:,[0]]
        ir_limb2_norm = ir_limb2 / ir_limb2[:,[0]]

        # Calculate limb statistics
        vis_limb1_mean = np.mean(vis_limb1_norm, axis=0)
        vis_limb1_std = np.std(vis_limb1_norm, axis=0)
        vis_limb2_mean = np.mean(vis_limb2_norm, axis=0)
        vis_limb2_std = np.std(vis_limb2_norm, axis=0)

        # Plot with error bars (note xerr for horizontal error bars)
        ax[1,0].errorbar(vis_limb1_mean, heights_out/1e3, xerr=vis_limb1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[1,0].errorbar(vis_limb2_mean, heights_out/1e3, xerr=vis_limb2_std,
                        fmt='s--', label=label2, capsize=3)
        ax[1,0].set_xlabel('Normalized Radiance')
        ax[1,0].set_ylabel('Tangent Height (km)')
        ax[1,0].legend()
        ax[1,0].set_title('Limb Visible Radiance')
        
        ir_limb1_mean = np.mean(ir_limb1_norm, axis=0)
        ir_limb1_std = np.std(ir_limb1_norm, axis=0)
        ir_limb2_mean = np.mean(ir_limb2_norm, axis=0)
        ir_limb2_std = np.std(ir_limb2_norm, axis=0)

        ax[1,1].errorbar(ir_limb1_mean, heights_out/1e3, xerr=ir_limb1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[1,1].errorbar(ir_limb2_mean, heights_out/1e3, xerr=ir_limb2_std,
                        fmt='s--', label=label2, capsize=3)
        ax[1,1].set_xlabel('Normalized Radiance')
        ax[1,1].set_ylabel('Tangent Height (km)')
        ax[1,1].legend()
        ax[1,1].set_title('Limb IR Radiance')


        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in plot_comparison: {str(e)}")
        raise

def plot_brightness_temps(obs1, obs2, label1, label2, title):
    """Helper function to plot brightness temperature comparisons with error bars"""
    try:
        rad = RadiationFundamentals()

        # Noise levels
        noise_visible = 0.02
        noise_ir = 0.02
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title)
        
        # Generate observations
        _, ir_rad1, angles_out, _ = obs1.generate_nadir_obs(
            dust_tau_true=0.4,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        _, ir_rad2, angles_out, _ = obs2.generate_nadir_obs(
            dust_tau_true=0.4,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        # Calculate brightness temperatures for each sample
        T_b1 = np.zeros_like(ir_rad1)
        T_b2 = np.zeros_like(ir_rad2)
        
        for i in range(ir_rad1.shape[0]):  # Loop over samples
            for j in range(ir_rad1.shape[1]):  # Loop over angles
                T_b1[i,j] = rad.brightness_temperature(ir_rad1[i,j], 
                                                     obs1.ir_center, 
                                                     obs1.ir_width)
                T_b2[i,j] = rad.brightness_temperature(ir_rad2[i,j],
                                                     obs2.ir_center,
                                                     obs2.ir_width)
        
        # Calculate statistics
        T_b1_mean = np.mean(T_b1, axis=0)
        T_b1_std = np.std(T_b1, axis=0)
        T_b2_mean = np.mean(T_b2, axis=0)
        T_b2_std = np.std(T_b2, axis=0)
        
        # Plot nadir brightness temperatures
        ax[0].errorbar(np.degrees(angles_out), T_b1_mean, yerr=T_b1_std,
                      fmt='o-', label=label1, capsize=3)
        ax[0].errorbar(np.degrees(angles_out), T_b2_mean, yerr=T_b2_std,
                      fmt='s--', label=label2, capsize=3)
        ax[0].set_xlabel('Emission Angle (degrees)')
        ax[0].set_ylabel('Brightness Temperature (K)')
        ax[0].legend()
        ax[0].set_title('Nadir IR Brightness Temperature')
        
        # Generate limb observations
        _, ir_limb1, heights_out, _ = obs1.generate_limb_obs(
            dust_tau_true=0.4,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        _, ir_limb2, heights_out, _ = obs2.generate_limb_obs(
            dust_tau_true=0.4,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        # Calculate limb brightness temperatures
        T_b_limb1 = np.zeros_like(ir_limb1)
        T_b_limb2 = np.zeros_like(ir_limb2)
        
        for i in range(ir_limb1.shape[0]):
            for j in range(ir_limb1.shape[1]):
                T_b_limb1[i,j] = rad.brightness_temperature(ir_limb1[i,j],
                                                          obs1.ir_center,
                                                          obs1.ir_width)
                T_b_limb2[i,j] = rad.brightness_temperature(ir_limb2[i,j],
                                                          obs2.ir_center,
                                                          obs2.ir_width)
        
        # Calculate limb statistics
        T_b_limb1_mean = np.mean(T_b_limb1, axis=0)
        T_b_limb1_std = np.std(T_b_limb1, axis=0)
        T_b_limb2_mean = np.mean(T_b_limb2, axis=0)
        T_b_limb2_std = np.std(T_b_limb2, axis=0)
        
        ax[1].errorbar(T_b_limb1_mean, heights_out/1e3, xerr=T_b_limb1_std,
                      fmt='o-', label=label1, capsize=3)
        ax[1].errorbar(T_b_limb2_mean, heights_out/1e3, xerr=T_b_limb2_std,
                      fmt='s--', label=label2, capsize=3)
        ax[1].set_xlabel('Brightness Temperature (K)')
        ax[1].set_ylabel('Tangent Height (km)')
        ax[1].legend()
        ax[1].set_title('Limb IR Brightness Temperature')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in plot_brightness_temps: {str(e)}")
        raise

def plot_comparison_optical_depth(obs1, obs2, obs3, tau1, tau2, tau3, label1, label2, label3, title):
    """Helper function to plot and compare two sets of observations
    
    Args:
        obs1, obs2: MarsSyntheticData instances with different parameters
        label1, label2: Plot labels for each case
        title: Overall plot title
    """
    try:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title)

        # Noise levels
        noise_visible = 0.02
        noise_ir = 0.02
        
        # Generate observations with noise
        vis_rad1, ir_rad1, angles_out, _ = obs1.generate_nadir_obs(
            dust_tau_true=tau1,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        vis_rad2, ir_rad2, angles_out, _ = obs2.generate_nadir_obs(
            dust_tau_true=tau2,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
#new
        vis_rad3, ir_rad3, angles_out, _ = obs3.generate_nadir_obs(
            dust_tau_true=tau3,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        

        vis_limb1, ir_limb1, heights_out, _ = obs1.generate_limb_obs(
            dust_tau_true=tau1,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        vis_limb2, ir_limb2, heights_out, _ = obs2.generate_limb_obs(
            dust_tau_true=tau2,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
#new
        vis_limb3, ir_limb3, heights_out, _ = obs3.generate_limb_obs(
            dust_tau_true=tau3,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )

        # Normalize each sample to its nadir value
        vis_rad1_norm = vis_rad1 / vis_rad1[:,[0]]  # Broadcasting for normalization
        vis_rad2_norm = vis_rad2 / vis_rad2[:,[0]]
#new
        vis_rad3_norm = vis_rad3 / vis_rad3[:,[0]]
        ir_rad1_norm = ir_rad1 / ir_rad1[:,[0]]
        ir_rad2_norm = ir_rad2 / ir_rad2[:,[0]]
#new
        ir_rad3_norm = ir_rad3 / ir_rad3[:,[0]]

        # Calculate means and standard deviations
        vis_rad1_mean = np.mean(vis_rad1_norm, axis=0)
        vis_rad1_std = np.std(vis_rad1_norm, axis=0)
        vis_rad2_mean = np.mean(vis_rad2_norm, axis=0)
        vis_rad2_std = np.std(vis_rad2_norm, axis=0)
#new
        vis_rad3_mean = np.mean(vis_rad3_norm, axis=0)
        vis_rad3_std = np.std(vis_rad3_norm, axis=0)

        ir_rad1_mean = np.mean(ir_rad1_norm, axis=0)
        ir_rad1_std = np.std(ir_rad1_norm, axis=0)
        ir_rad2_mean = np.mean(ir_rad2_norm, axis=0)
        ir_rad2_std = np.std(ir_rad2_norm, axis=0)
#new
        ir_rad3_mean = np.mean(ir_rad3_norm, axis=0)
        ir_rad3_std = np.std(ir_rad3_norm, axis=0)

        # Plot with error bars
        ax[0,0].errorbar(np.degrees(angles_out), vis_rad1_mean, yerr=vis_rad1_std, 
                        fmt='o-', label=label1, capsize=3)
        ax[0,0].errorbar(np.degrees(angles_out), vis_rad2_mean, yerr=vis_rad2_std, 
                        fmt='s--', label=label2, capsize=3)
#new
        ax[0,0].errorbar(np.degrees(angles_out), vis_rad3_mean, yerr=vis_rad3_std, 
                        fmt='s--', label=label3, capsize=3)
        ax[0,0].set_xlabel('Emission Angle (degrees)')
        ax[0,0].set_ylabel('Normalized Radiance')
        ax[0,0].legend()
        ax[0,0].set_title('Nadir Visible Radiance')

        ax[0,1].errorbar(np.degrees(angles_out), ir_rad1_mean, yerr=ir_rad1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[0,1].errorbar(np.degrees(angles_out), ir_rad2_mean, yerr=ir_rad2_std,
                        fmt='s--', label=label2, capsize=3)
#new
        ax[0,1].errorbar(np.degrees(angles_out), ir_rad3_mean, yerr=ir_rad3_std,
                        fmt='s--', label=label3, capsize=3)
        ax[0,1].set_xlabel('Emission Angle (degrees)')
        ax[0,1].set_ylabel('Normalized Radiance')
        ax[0,1].legend()
        ax[0,1].set_title('Nadir IR Radiance')

        # Normalize limb observations
        vis_limb1_norm = vis_limb1 / vis_limb1[:,[0]]
        vis_limb2_norm = vis_limb2 / vis_limb2[:,[0]]
#new
        vis_limb3_norm = vis_limb3 / vis_limb3[:,[0]]
        ir_limb1_norm = ir_limb1 / ir_limb1[:,[0]]
        ir_limb2_norm = ir_limb2 / ir_limb2[:,[0]]
#new
        ir_limb3_norm = ir_limb3 / ir_limb3[:,[0]]

        # Calculate limb statistics
        vis_limb1_mean = np.mean(vis_limb1_norm, axis=0)
        vis_limb1_std = np.std(vis_limb1_norm, axis=0)
        vis_limb2_mean = np.mean(vis_limb2_norm, axis=0)
        vis_limb2_std = np.std(vis_limb2_norm, axis=0)
#new
        vis_limb3_mean = np.mean(vis_limb3_norm, axis=0)
        vis_limb3_std = np.std(vis_limb3_norm, axis=0)

        # Plot with error bars (note xerr for horizontal error bars)
        ax[1,0].errorbar(vis_limb1_mean, heights_out/1e3, xerr=vis_limb1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[1,0].errorbar(vis_limb2_mean, heights_out/1e3, xerr=vis_limb2_std,
                        fmt='s--', label=label2, capsize=3)
#new
        ax[1,0].errorbar(vis_limb3_mean, heights_out/1e3, xerr=vis_limb3_std,
                        fmt='s--', label=label3, capsize=3)
        ax[1,0].set_xlabel('Normalized Radiance')
        ax[1,0].set_ylabel('Tangent Height (km)')
        ax[1,0].legend()
        ax[1,0].set_title('Limb Visible Radiance')
        
        ir_limb1_mean = np.mean(ir_limb1_norm, axis=0)
        ir_limb1_std = np.std(ir_limb1_norm, axis=0)
        ir_limb2_mean = np.mean(ir_limb2_norm, axis=0)
        ir_limb2_std = np.std(ir_limb2_norm, axis=0)
#new
        ir_limb3_mean = np.mean(ir_limb3_norm, axis=0)
        ir_limb3_std = np.std(ir_limb3_norm, axis=0)

        ax[1,1].errorbar(ir_limb1_mean, heights_out/1e3, xerr=ir_limb1_std,
                        fmt='o-', label=label1, capsize=3)
        ax[1,1].errorbar(ir_limb2_mean, heights_out/1e3, xerr=ir_limb2_std,
                        fmt='s--', label=label2, capsize=3)
        ax[1,1].errorbar(ir_limb3_mean, heights_out/1e3, xerr=ir_limb3_std,
                        fmt='s--', label=label3, capsize=3)
        ax[1,1].set_xlabel('Normalized Radiance')
        ax[1,1].set_ylabel('Tangent Height (km)')
        ax[1,1].legend()
        ax[1,1].set_title('Limb IR Radiance')


        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in plot_comparison: {str(e)}")
        raise

def plot_brightness_temps_optical_depth(obs1, obs2, obs3, tau1, tau2, tau3, label1, label2, label3, title):
    """Helper function to plot brightness temperature comparisons with error bars"""
    try:
        rad = RadiationFundamentals()

        # Noise levels
        noise_visible = 0.02
        noise_ir = 0.02
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title)
        
        # Generate observations
        _, ir_rad1, angles_out, _ = obs1.generate_nadir_obs(
            dust_tau_true=tau1,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        _, ir_rad2, angles_out, _ = obs2.generate_nadir_obs(
            dust_tau_true=tau2,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
#new
        _, ir_rad3, angles_out, _ = obs3.generate_nadir_obs(
            dust_tau_true=tau3,
            surface_albedo_true=0.25,
            surface_temp=surface_temp,
            emission_angles=angles,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        # Calculate brightness temperatures for each sample
        T_b1 = np.zeros_like(ir_rad1)
        T_b2 = np.zeros_like(ir_rad2)
#new
        T_b3 = np.zeros_like(ir_rad3)  
        
        for i in range(ir_rad1.shape[0]):  # Loop over samples
            for j in range(ir_rad1.shape[1]):  # Loop over angles
                T_b1[i,j] = rad.brightness_temperature(ir_rad1[i,j], 
                                                     obs1.ir_center, 
                                                     obs1.ir_width)
                T_b2[i,j] = rad.brightness_temperature(ir_rad2[i,j],
                                                     obs2.ir_center,
                                                     obs2.ir_width)
#new
                T_b3[i,j] = rad.brightness_temperature(ir_rad3[i,j],
                                                     obs3.ir_center,
                                                     obs3.ir_width)
        
        # Calculate statistics
        T_b1_mean = np.mean(T_b1, axis=0)
        T_b1_std = np.std(T_b1, axis=0)
        T_b2_mean = np.mean(T_b2, axis=0)
        T_b2_std = np.std(T_b2, axis=0)
#new
        T_b3_mean = np.mean(T_b3, axis=0)
        T_b3_std = np.std(T_b3, axis=0)
        
        # Plot nadir brightness temperatures
        ax[0].errorbar(np.degrees(angles_out), T_b1_mean, yerr=T_b1_std,
                      fmt='o-', label=label1, capsize=3)
        ax[0].errorbar(np.degrees(angles_out), T_b2_mean, yerr=T_b2_std,
                      fmt='s--', label=label2, capsize=3)
#new
        ax[0].errorbar(np.degrees(angles_out), T_b3_mean, yerr=T_b3_std,
                      fmt='s--', label=label3, capsize=3)
        ax[0].set_xlabel('Emission Angle (degrees)')
        ax[0].set_ylabel('Brightness Temperature (K)')
        ax[0].legend()
        ax[0].set_title('Nadir IR Brightness Temperature')
        
        # Generate limb observations
        _, ir_limb1, heights_out, _ = obs1.generate_limb_obs(
            dust_tau_true=tau1,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        _, ir_limb2, heights_out, _ = obs2.generate_limb_obs(
            dust_tau_true=tau2,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
#new
        _, ir_limb3, heights_out, _ = obs3.generate_limb_obs(
            dust_tau_true=tau3,
            tangent_heights=heights,
            surface_temp=surface_temp,
            noise_visible=noise_visible,
            noise_ir=noise_ir
        )
        
        # Calculate limb brightness temperatures
        T_b_limb1 = np.zeros_like(ir_limb1)
        T_b_limb2 = np.zeros_like(ir_limb2)
#new
        T_b_limb3 = np.zeros_like(ir_limb3)
        
        for i in range(ir_limb1.shape[0]):
            for j in range(ir_limb1.shape[1]):
                T_b_limb1[i,j] = rad.brightness_temperature(ir_limb1[i,j],
                                                          obs1.ir_center,
                                                          obs1.ir_width)
                T_b_limb2[i,j] = rad.brightness_temperature(ir_limb2[i,j],
                                                          obs2.ir_center,
                                                          obs2.ir_width)
#new
                T_b_limb3[i,j] = rad.brightness_temperature(ir_limb3[i,j],
                                                          obs3.ir_center,
                                                          obs3.ir_width)
        
        # Calculate limb statistics
        T_b_limb1_mean = np.mean(T_b_limb1, axis=0)
        T_b_limb1_std = np.std(T_b_limb1, axis=0)
        T_b_limb2_mean = np.mean(T_b_limb2, axis=0)
        T_b_limb2_std = np.std(T_b_limb2, axis=0)
#new
        T_b_limb3_mean = np.mean(T_b_limb3, axis=0)
        T_b_limb3_std = np.std(T_b_limb3, axis=0)
        
        ax[1].errorbar(T_b_limb1_mean, heights_out/1e3, xerr=T_b_limb1_std,
                      fmt='o-', label=label1, capsize=3)
        ax[1].errorbar(T_b_limb2_mean, heights_out/1e3, xerr=T_b_limb2_std,
                      fmt='s--', label=label2, capsize=3)
#new
        ax[1].errorbar(T_b_limb3_mean, heights_out/1e3, xerr=T_b_limb3_std,
                      fmt='s--', label=label3, capsize=3)
        ax[1].set_xlabel('Brightness Temperature (K)')
        ax[1].set_ylabel('Tangent Height (km)')
        ax[1].legend()
        ax[1].set_title('Limb IR Brightness Temperature')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error in plot_brightness_temps: {str(e)}")
        raise