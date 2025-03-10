import numpy as np



def max_velocity(particle_data):
    '''
    Returns the maximum velocity magnitude of particles.

    Parameters:
    - particle_data: (5, n) array, where rows 2,3,4 contain velocity components.

    Returns:
    - max_velocity: Maximum velocity magnitude.
    '''
    v = particle_data[2:, :]  # Extract velocity components (vz, vr, vf)
    v_magnitude = np.sqrt(np.sum(v**2, axis=0))  # Compute speed
    return np.max(v_magnitude)

def max_value_in_array(arr):
    '''
    Returns the maximum value and its index in a 2D numpy array.

    Parameters:
    - arr: 2D numpy array.

    Returns:
    - max_value: Maximum value.
    - max_index: Tuple (i, j) of its location.
    '''
    max_value = np.max(arr)
    max_index = np.unravel_index(np.argmax(arr), arr.shape)  # Get 2D index
    return max_value, max_index

def debye_length(ne, Te):
    '''
    Computes the Debye length in a cell.

    Parameters:
    - ne: Electron number density (m⁻³).
    - Te: Electron temperature (K).

    Returns:
    - debye_length: Debye length (m).
    '''
    e0 = 8.85418782e-12  # Vacuum permittivity (F/m)
    kb = 1.38064852e-23  # Boltzmann constant (J/K)
    qe = 1.60217662e-19  # Elementary charge (C)

    return np.sqrt(e0 * kb * Te / (ne * qe**2))

def larmor_radius(v_perp, B, q, m):
    '''
    Computes the minimum Larmor radius.

    Parameters:
    - v_perp: Perpendicular velocity of particles (m/s).
    - B: Magnetic field strength (T).
    - q: Charge of particle (C).
    - m: Mass of particle (kg).

    Returns:
    - min_radius: Minimum Larmor radius.
    '''
    r_L = m * v_perp / (q * B)  # Larmor radius formula
    return np.min(r_L)

def plasma_frequency(ne, q, m):
    '''
    Computes the plasma frequency in a cell.

    Parameters:
    - ne: Electron number density (m⁻³).
    - q: Charge of particle (C).
    - m: Mass of particle (kg).

    Returns:
    - omega_p: Plasma frequency (rad/s).
    '''
    e0 = 8.85418782e-12  # Vacuum permittivity
    return np.sqrt(ne * q**2 / (e0 * m))

def cyclotron_frequency(B, q, m):
    '''
    Computes the maximum cyclotron frequency.

    Parameters:
    - B: Magnetic field strength (T).
    - q: Charge of particle (C).
    - m: Mass of particle (kg).

    Returns:
    - omega_c: Cyclotron frequency (rad/s).
    '''
    return np.max(q * B / m)

def collision_frequency(n, v, sigma):
    '''
    Computes the maximum collision frequency.

    Parameters:
    - n: Particle number density (m⁻³).
    - v: Relative velocity (m/s).
    - sigma: Collision cross-section (m²).

    Returns:
    - max_nu: Maximum collision frequency.
    '''
    return np.max(n * v * sigma)


if __name__ == '__main__':
    pass