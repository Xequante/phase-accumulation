import numpy as np
import matplotlib.pyplot as plt
from src.propagation.np_propagate import free_space_propagate


def run_simulations():

    # Simulation parameters
    NAs = [0.9, 0.95]
    wavelengths_nm = [750, 1000, 1500]
    # NAs = [0.9]
    # wavelengths_nm = [750]
    f = 5e-4
    N = 21
    resolution = 4000
    distances = np.linspace(0.5 * f, 1.5 * f, N)

    # Prepare plots
    fig, axs = plt.subplots(len(NAs), len(wavelengths_nm))

    # Run simulation for each parameter
    data = []
    for i, na in enumerate(NAs):
        for j, wavelength in enumerate(wavelengths_nm):

            # Print current run
            print(na, wavelength)

            # Convert wavelength to nm
            wavelength *= 1e-9

            # Calculate the field when focused by a microscope objective
            x, field_propagations = microscope_calculation(
                distances=distances, f=f, na=na,
                wavelength=wavelength, resolution=2 * resolution + 1
            )

            # Check the phase profiles
            for z in range(N):
                field_profile = field_propagations[z]
                phase = np.angle(field_profile)

            # Take cross sections of the data
            intensity_cross_section = np.zeros((2 * resolution + 1, N))
            energies = np.zeros(N)
            for z in range(N):
                profile = np.abs(field_propagations[z]) ** 2
                energies[z] = np.sum(profile)
                intensity_cross_section[:, z] = profile[:, resolution + 1]
            data.append(intensity_cross_section)
            print(np.max(energies) - np.min(energies))

            # Select axis for plotting
            if len(NAs) > 1 or len(wavelengths_nm) > 1:
                ax = axs[i, j]
            else:
                ax = axs

            # Create the plot
            extent = [distances[0], distances[-1], x[0], x[-1]]
            ax.imshow(intensity_cross_section, extent=extent,
                      origin='lower', aspect='auto')
            ax.set_xlabel("z (m)")
            ax.set_ylabel("x (m)")

            # Calculate the expected vs. simulated FWHM
            expected_fwhm = 0.51 * wavelength / na

            # Identify the focus
            z_focus_idx = np.argmax(
                [np.max(np.abs(f) ** 2) for f in field_propagations])
            focal_length = distances[z_focus_idx]
            print(f'Focal length = {focal_length}')

            # Simulated FWHM
            focus_cross_section = intensity_cross_section[:, z_focus_idx]
            simulated_fwhm = calculate_fwhm(focus_cross_section, x)

            # Print the comparison
            print(expected_fwhm, simulated_fwhm)

    # Display the plot
    plt.show()


def microscope_calculation(distances, f: float = 5e-3, n: float = 1,
                           na: float = 0.9, wavelength: float = 750e-9,
                           resolution: int = 1001):

    # Calculate the necessary radius of the lens
    theta_max = np.arcsin(na / n)

    # Aperture radius determined by geometry: r_max = f * tan(theta_max)
    r_max = f * np.tan(theta_max)

    # Select spatial coordinates for the system
    spatial_coords = np.linspace(-2 * r_max, 2 * r_max, resolution)
    x, y = np.meshgrid(spatial_coords, spatial_coords)

    # Calculate the gaussian input field
    input_field, z_R = gaussian_beam_field(x, y, r_max, wavelength=wavelength)

    # Try a input plane wave
    input_field = np.ones(x.shape)

    # Apply lens aperature
    input_field[x**2 + y**2 > r_max**2] = 0

    # Calculate the phase profile of the lens
    phase, _ = lens_phase_profile(x, y, na, f, n=n, wavelength=wavelength)

    # Plotting test
    # plt.imshow
    # plt.imshow(phase)
    # plt.show()

    # Apply the phase profile to the lens
    modulated_field = input_field * np.exp(1j * phase)

    # Free space propagate at each distance
    field_profiles = []
    for d in distances:
        field_profiles.append(free_space_propagate(
            modulated_field, x=x, y=y, wavelength=wavelength, distance=d
        ))

    # Return the field profiles
    return spatial_coords, field_profiles


def gaussian_beam_field(x, y, w0, wavelength=1.0):
    """
    Compute the complex field of a collimated Gaussian beam at Rayleigh range.

    Parameters:
    x, y       : 2D numpy arrays representing spatial coordinates
    w0         : Beam waist radius at focus (z = 0)
    wavelength : Wavelength of the beam (default = 1.0 for normalized units)

    Returns:
    field      : 2D numpy array of the complex electric field E(x, y)
    z_R        : Rayleigh range
    """

    # Calculate the Rayleigh range
    k = 2 * np.pi / wavelength
    z_R = np.pi * w0**2 / wavelength  # Rayleigh range

    # Compute the parameters of the beam
    r_sq = x**2 + y**2
    R_z = z_R  # At z = z_R, R(z) = 2 * z_R
    w_z = w0 * np.sqrt(2)  # Beam radius at Rayleigh range

    # Compute amplitude and phase of the beam profile
    amplitude = (w0 / w_z) * np.exp(-r_sq / w_z**2)
    phase = -k * r_sq / (2 * R_z) + np.pi / 4  # Gouy phase at z_R = π/4

    # Calculate the field
    field = amplitude * np.exp(1j * phase)
    return field, z_R


def lens_phase_profile(x, y, na, f, n=1.0, wavelength=1.0):
    """
    Generate the phase profile of a lens.

    Parameters:
    x, y       : 2D numpy arrays representing spatial coordinates
    NA         : Numerical aperture of the lens (NA = n * sin(theta_max))
    f          : Focal length of the lens
    n          : Refractive index of the surrounding medium (default = 1.0)
    wavelength : Wavelength of light (default = 1.0 for normalized units)

    Returns:
    phase      : 2D numpy array with phase profile in radians
    r_max      : Maximum radius (aperture radius) defined by NA
    """

    # Parameters (Wavenumber and Radius)
    k = 2 * np.pi / wavelength
    r = np.sqrt(x**2 + y**2)

    # Compute theta_max from NA = n * sin(theta_max)
    theta_max = np.arcsin(na / n)

    # Aperture radius determined by geometry: r_max = f * tan(theta_max)
    r_max = f * np.tan(theta_max)

    # Phase profile of ideal thin lens: φ(x,y) = -k (sqrt(x^2 + y^2 + f^2) - f)
    phase = -k * (np.sqrt(r**2 + f**2) - f)
    # phase = - k * ((f ** 2 + r ** 2) - f)

    r_max = f * np.tan(np.arcsin(na))
    delta_phi = 2 * np.pi / wavelength * (np.sqrt(r_max ** 2 + f ** 2) - f)
    print(f"Max phase swing: {delta_phi:.2f} rad")

    return phase, r_max


def calculate_fwhm(intensity, x=None):
    """
    Calculate the full width at half maximum (FWHM) of a 1D intensity profile.

    Parameters:
    intensity : 1D np.array
        Intensity values (not field amplitude).
    x : 1D np.array or None
        Corresponding spatial coordinates. If None, index positions are used.

    Returns:
    fwhm : float
        Full width at half maximum.
    """

    # Normalize intensity
    intensity = np.array(intensity)
    intensity -= np.min(intensity)
    intensity /= np.max(intensity)

    # Half maximum
    half_max = 0.5

    # Find where intensity crosses half maximum
    indices = np.where(intensity >= half_max)[0]
    if len(indices) < 2:
        return 0.0  # No proper FWHM

    left_idx = indices[0]
    right_idx = indices[-1]

    if x is None:
        fwhm = right_idx - left_idx
    else:
        fwhm = x[right_idx] - x[left_idx]

    return fwhm