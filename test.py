import h5py
import numpy as np

def read_particles_as_matrix(file_path: str, particle_name: str) -> np.ndarray:
    """Reads particle data from an HDF5 file and returns it as an N × 5 NumPy matrix.
    
    Args:
        file_path (str): Path to the HDF5 file.
        particle_name (str): The key under which particle data is stored.

    Returns:
        np.ndarray: An N × 5 matrix containing all particle data.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if particle_name not in f:
                print(f"Warning: '{particle_name}' not found in {file_path}")
                return np.empty((0, 5))  # Return an empty N×5 array if not found
            
            data_list = []  # List to store N×5 matrices
            
            for group_name in f[particle_name].keys():
                group = f[particle_name][group_name]
                if 'ptcls' in group:
                    data = group['ptcls'][()]  # Read as NumPy array
                    print(data.shape, data.shape[0], data.shape[1])
                    if data.shape[1] == 5:  # Ensure correct format
                        data_list.append(data)  # Keep as is (N × 5)
                    else:
                        print(f"Warning: Skipping {group_name} due to incorrect shape {data.shape}")
            
            if data_list:
                return np.vstack(data_list)  # Concatenate along rows to get N × 5
            else:
                return np.empty((0, 5))  # Return empty matrix if no valid data found
    except (OSError, KeyError, ValueError) as e:
        print(f"Error reading {file_path}: {e}")
        return np.empty((0, 5))  # Return empty matrix in case of errors


ions = read_particles_as_matrix("dump/2.h5", "ions")
print(ions.shape)

'''
with h5py.File("dump/2.h5", "r") as f:
    ions = f['ions']
    group = ions['pGrp0']
    particles = group['ptcls']
    np_particles = particles[()]  # Read all data as a NumPy array
    print(np_particles.shape)
'''


'''







'''