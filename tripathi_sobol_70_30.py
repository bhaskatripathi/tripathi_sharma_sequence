import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.stats.qmc import discrepancy

print(f"NumPy version: {np.__version__}")

def sobol_sequence(dim, n):
    """Generate a standard Sobol sequence."""
    sobol_engine = qmc.Sobol(d=dim, scramble=True)
    return sobol_engine.random(n)

def halton_sequence(dim, n):
    """Generate a standard Halton sequence."""
    halton_engine = qmc.Halton(d=dim)
    return halton_engine.random(n)

def tripathi_sharma_sequence(dim, n, optimization_param=0.7):
    """
    Generate a hybrid Tripathi-Sharma sequence using a grid of sunflower patterns
    with Sobol sequence points filling the gaps between clusters.
    
    Parameters:
    -----------
    dim : int
        Number of dimensions
    n : int
        Number of points to generate
    optimization_param : float
        Parameter controlling the optimization strength (0.0 to 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Hybrid Tripathi-Sharma sequence with shape (n, dim)
    """
    # Ensure parameters are valid
    optimization_param = float(np.clip(optimization_param, 0.0, 1.0))
    dim = int(dim)
    n = int(n)
    
    # Create the output array
    sequence = np.zeros((n, dim))
    
    # Define the Golden Angle in radians (137.5 degrees)
    golden_angle = 137.5 * (np.pi / 180)
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    # Determine grid size for sunflower patterns
    points_per_sunflower = 40
    grid_size = max(3, int(np.sqrt(n / points_per_sunflower)))
    
    # Create centers for the sunflower patterns
    sunflower_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (j + 0.5) / grid_size
            center_y = (i + 0.5) / grid_size
            sunflower_centers.append((center_x, center_y))
    
    # Calculate sunflower coverage - use 70% of points for sunflowers, 30% for Sobol infill
    sunflower_points = int(n * 0.7)
    sobol_points = n - sunflower_points
    
    # Calculate points per sunflower pattern
    points_per_pattern = sunflower_points // len(sunflower_centers)
    remaining_points = sunflower_points % len(sunflower_centers)
    
    # Keep track of occupied regions to avoid overlap
    occupied_mask = np.zeros((100, 100), dtype=bool)  # Fine grid to track occupancy
    
    # Generate points for each sunflower pattern
    point_index = 0
    
    for i, (center_x, center_y) in enumerate(sunflower_centers):
        # Determine how many points for this pattern
        this_pattern_points = points_per_pattern
        if i < remaining_points:
            this_pattern_points += 1
        
        # Skip if no points assigned
        if this_pattern_points == 0:
            continue
        
        # Determine if this is a forward or reverse spiral
        is_reverse = (i % 2 == 1)
        
        # Generate sunflower pattern
        for j in range(this_pattern_points):
            # Adjust angle based on forward/reverse setting
            if is_reverse:
                angle = -j * golden_angle
            else:
                angle = j * golden_angle
            
            # Calculate radius - scale based on grid size but slightly smaller
            # to leave some gaps between clusters
            max_radius = 0.35 / grid_size
            radius = max_radius * np.sqrt(j / this_pattern_points)
            
            # Convert to Cartesian coordinates and position at center
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Ensure point is in domain
            if 0 <= x <= 1 and 0 <= y <= 1:
                sequence[point_index, 0] = x
                sequence[point_index, 1] = y
                
                # Mark this region as occupied
                grid_x = int(x * 99)
                grid_y = int(y * 99)
                occupied_mask[grid_y, grid_x] = True
                
                # Move to next point
                point_index += 1
                if point_index >= sunflower_points:
                    break
        
        if point_index >= sunflower_points:
            break
    
    # Fill the remaining spaces with Sobol sequence
    if sobol_points > 0:
        # Generate a Sobol sequence with more points than needed
        sobol_engine = qmc.Sobol(d=dim, scramble=True)
        # Generate 3x the needed points to have plenty to filter
        extra_sobol = sobol_engine.random(sobol_points * 3)
        
        # Find points that are in empty regions
        sobol_index = 0
        added_sobol = 0
        
        while added_sobol < sobol_points and sobol_index < len(extra_sobol):
            x = extra_sobol[sobol_index, 0]
            y = extra_sobol[sobol_index, 1]
            
            # Check if this region is already occupied
            grid_x = int(x * 99)
            grid_y = int(y * 99)
            
            # Check 3x3 neighborhood to ensure minimum distance
            neighborhood_occupied = False
            for nx in range(max(0, grid_x-1), min(99, grid_x+2)):
                for ny in range(max(0, grid_y-1), min(99, grid_y+2)):
                    if occupied_mask[ny, nx]:
                        neighborhood_occupied = True
                        break
                if neighborhood_occupied:
                    break
            
            # If not occupied, add this Sobol point
            if not neighborhood_occupied:
                sequence[point_index, :] = extra_sobol[sobol_index, :]
                occupied_mask[grid_y, grid_x] = True
                point_index += 1
                added_sobol += 1
            
            sobol_index += 1
            
            # If we're running out of candidate points, relax the neighborhood constraint
            if sobol_index > len(extra_sobol) - sobol_points + added_sobol:
                # Just fill remaining points with Sobol
                remaining_to_add = sobol_points - added_sobol
                sequence[point_index:point_index+remaining_to_add] = extra_sobol[sobol_index:sobol_index+remaining_to_add]
                break
    
    # For dimensions beyond 2, use standard Sobol for better coverage
    if dim > 2:
        sobol_engine = qmc.Sobol(d=dim-2, scramble=True)
        higher_dims = sobol_engine.random(n)
        sequence[:, 2:] = higher_dims
    
    # Apply light stratification to improve distribution
    if optimization_param > 0:
        for d in range(dim):
            # Sort along this dimension
            indices = np.argsort(sequence[:, d])
            sorted_values = sequence[indices, d].copy()
            
            # Create stratified values
            stratified = np.linspace(0, 1, n)
            
            # Move toward stratified distribution (light touch)
            move_factor = 0.1 * optimization_param
            sequence[indices, d] = (1 - move_factor) * sorted_values + move_factor * stratified
    
    # Ensure all values stay in [0,1] range
    sequence = np.clip(sequence, 0, 1)
    
    return sequence

def safe_discrepancy(seq, method):
    """Safely compute discrepancy with error handling."""
    try:
        result = discrepancy(seq, method=method)
        # Convert to Python float
        if isinstance(result, np.ndarray):
            if result.size == 1:
                return float(result.item())
            else:
                return float(np.mean(result))
        return float(result)
    except Exception as e:
        print(f"Error computing {method} discrepancy: {str(e)}")
        return float('nan')

def compute_discrepancies(dim, n, alpha=0.5):
    """Compute discrepancies for different sequences."""
    # Generate sequences
    sobol_seq = sobol_sequence(dim, n)
    halton_seq = halton_sequence(dim, n)
    tripathi_sharma_seq = tripathi_sharma_sequence(dim, n, alpha)
    
    # Compute discrepancies
    print("Computing L2-star discrepancies...")
    sobol_l2 = safe_discrepancy(sobol_seq, 'L2-star')
    halton_l2 = safe_discrepancy(halton_seq, 'L2-star')
    ts_l2 = safe_discrepancy(tripathi_sharma_seq, 'L2-star')
    
    print("Computing CD discrepancies...")
    sobol_cd = safe_discrepancy(sobol_seq, 'CD')
    halton_cd = safe_discrepancy(halton_seq, 'CD')
    ts_cd = safe_discrepancy(tripathi_sharma_seq, 'CD')
    
    print("Computing MD discrepancies...")
    sobol_md = safe_discrepancy(sobol_seq, 'MD')
    halton_md = safe_discrepancy(halton_seq, 'MD')
    ts_md = safe_discrepancy(tripathi_sharma_seq, 'MD')
    
    # Store results directly in Python lists
    sequence_names = ["Sobol", "Halton", "Tripathi-Sharma"]
    l2_values = [sobol_l2, halton_l2, ts_l2]
    cd_values = [sobol_cd, halton_cd, ts_cd]
    md_values = [sobol_md, halton_md, ts_md]
    
    return sequence_names, l2_values, cd_values, md_values, sobol_seq, halton_seq, tripathi_sharma_seq

def plot_sequences(sobol_seq, halton_seq, tripathi_sharma_seq, save_path=None):
    """Plot the different sequences for visual comparison."""
    if sobol_seq.shape[1] < 2:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        indices = np.arange(len(sobol_seq))
        axs[0].scatter(indices, sobol_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence')
        axs[1].scatter(indices, halton_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence')
        axs[2].scatter(indices, tripathi_sharma_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[2].set_title('Tripathi-Sharma Sequence')
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence')
        axs[1].scatter(halton_seq[:, 0], halton_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence')
        axs[2].scatter(tripathi_sharma_seq[:, 0], tripathi_sharma_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[2].set_title('Tripathi-Sharma Sequence')
    
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    # Parameters
    dim = 50
    n = 1024  # Use a power of 2 to avoid Sobol warning
    alpha = 0.1
    
    print(f"Computing discrepancies for {n} points in {dim} dimensions with alpha={alpha}")
    
    # Compute discrepancies
    sequence_names, l2_values, cd_values, md_values, sobol_seq, halton_seq, tripathi_sharma_seq = compute_discrepancies(dim, n, alpha)
    
    # Print results without using pandas
    print("\nDiscrepancy Results:")
    print(f"{'Sequence':<20} {'L2-star':<15} {'CD':<15} {'MD':<15}")
    print("-" * 65)
    for i, name in enumerate(sequence_names):
        print(f"{name:<20} {l2_values[i]:<15.10f} {cd_values[i]:<15.10f} {md_values[i]:<15.10f}")
    
    # Find the best sequence for each metric
    min_l2_idx = l2_values.index(min(l2_values))
    min_cd_idx = cd_values.index(min(cd_values))
    min_md_idx = md_values.index(min(md_values))
    
    print("\nBest Sequences:")
    print(f"L2-star: {sequence_names[min_l2_idx]} ({l2_values[min_l2_idx]:.10f})")
    print(f"CD: {sequence_names[min_cd_idx]} ({cd_values[min_cd_idx]:.10f})")
    print(f"MD: {sequence_names[min_md_idx]} ({md_values[min_md_idx]:.10f})")
    
    # Plot sequences
    plot_sequences(sobol_seq, halton_seq, tripathi_sharma_seq, "sequence_comparison.png")
    
    print("\nSequence visualization saved to 'sequence_comparison.png'")

if __name__ == "__main__":
    main()
