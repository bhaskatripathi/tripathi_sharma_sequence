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

def golden_spiral_sequence(dim, n, optimization_param=0.7, spiral_density=None, growth_rate=None, 
                           min_grid_size=None, spiral_coverage=None, spiral_direction_alternation=True):
    """
    Generate a sequence based on true Golden Logarithmic Spirals with fully adaptive parameters.
    
    Parameters:
    -----------
    dim : int
        Number of dimensions
    n : int
        Number of points to generate
    optimization_param : float
        Parameter controlling the optimization strength (0.0 to 1.0)
    spiral_density : float or None
        Controls how many points are in each spiral. If None, calculated automatically.
        Lower values create more spirals with fewer points each.
    growth_rate : float or None
        Controls how quickly spirals expand. If None, based on golden ratio.
        Higher values create tighter spirals.
    min_grid_size : int or None
        Minimum size of spiral grid. If None, calculated based on point count.
    spiral_coverage : float or None
        Fraction of cell size that spirals should cover (0.0-1.0). If None, adaptive.
    spiral_direction_alternation : bool
        Whether to alternate spiral directions (clockwise/counterclockwise)
        
    Returns:
    --------
    numpy.ndarray
        Golden Spiral sequence with shape (n, dim)
    """
    # Ensure parameters are valid
    optimization_param = float(np.clip(optimization_param, 0.0, 1.0))
    dim = int(dim)
    n = int(n)
    
    # Create the output array
    sequence = np.zeros((n, dim))
    
    # Golden Ratio and related constants
    phi = (1 + np.sqrt(5)) / 2
    
    # Default growth rate based on golden ratio if not specified
    if growth_rate is None:
        # Natural logarithmic spiral growth based on golden ratio
        growth_rate = 5 * np.pi  # Default denominator for theta in exponential
        # Allows for approximately 5π radians to double in size, which is ~2.5 turns
    
    # Calculate adaptive spiral density if not specified
    if spiral_density is None:
        # Scale with sqrt(n) to balance between too many small spirals and too few large ones
        # Lower bound ensures enough points for a nice spiral, upper bound prevents giant spirals
        points_per_spiral_factor = 2.0  # Controls overall spiral granularity
        spiral_density = np.sqrt(n) * points_per_spiral_factor
        # Cap values to reasonable ranges
        spiral_density = max(15, min(120, spiral_density))
    
    # Calculate adaptive grid size based on point count and spiral density
    if min_grid_size is None:
        # Minimum size based on dimension to ensure adequate coverage in higher dimensions
        min_grid_size = max(2, int(dim ** 0.25))
    
    # Calculate grid size based on point count and spiral density
    grid_size = max(min_grid_size, int(np.sqrt(n / spiral_density)))
    
    # Determine spiral coverage ratio if not specified
    if spiral_coverage is None:
        # Adaptive coverage - smaller for higher dimensions to prevent overlap
        # Range is typically 0.4-0.9
        spiral_coverage = 0.9 / (1 + 0.1 * np.sqrt(dim))
    
    # Log the calculated parameters
    print(f"Dynamic parameters: grid_size={grid_size}×{grid_size}, spiral_density~{spiral_density:.1f}")
    print(f"Coverage: {spiral_coverage:.2f}, Growth rate: {growth_rate:.1f}")
    
    # Create spiral centers
    spiral_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (j + 0.5) / grid_size
            center_y = (i + 0.5) / grid_size
            spiral_centers.append((center_x, center_y))
    
    # Determine points per spiral
    total_spirals = len(spiral_centers)
    points_per_spiral_exact = n / total_spirals
    
    # Track current point index
    point_index = 0
    
    # Generate points for each spiral
    for spiral_idx, (center_x, center_y) in enumerate(spiral_centers):
        # Calculate exact number of points for this spiral
        # This ensures we get exactly n points in total
        start_point = int(spiral_idx * points_per_spiral_exact)
        end_point = int((spiral_idx + 1) * points_per_spiral_exact)
        num_points = end_point - start_point
        
        if num_points <= 0:
            continue
            
        # Calculate the maximum radius for this spiral
        # Spiral coverage controls how much of each grid cell is covered
        max_radius = (spiral_coverage / 2) / grid_size
        
        # Determine spiral direction
        if spiral_direction_alternation:
            clockwise = (spiral_idx % 2 == 0)
        else:
            clockwise = True  # All clockwise if alternation disabled
        
        # Calculate number of revolutions based on point count and optimization parameter
        # More points = more revolutions needed for smooth coverage
        base_revolutions = 1.0
        point_based_revolutions = np.log2(1 + num_points / 10) * 0.5
        optimization_revolutions = optimization_param * 2
        num_revolutions = base_revolutions + point_based_revolutions + optimization_revolutions
        
        # Calculate minimum radius as a fraction of maximum
        # This prevents points from clustering too tightly in the center
        min_radius_ratio = max(0.01, 0.05 / np.sqrt(num_points))
        min_radius = max_radius * min_radius_ratio
        
        # For each point in this spiral
        for i in range(num_points):
            # Calculate normalized position along the spiral (0 to 1)
            t = i / max(1, num_points - 1)
            
            # Use sqrt(t) for more uniform point spacing along the spiral
            # This compensates for the exponential growth of the spiral
            theta = num_revolutions * 2 * np.pi * np.sqrt(t)
            if not clockwise:
                theta = -theta
                
            # Calculate radius using the logarithmic spiral formula
            # The radius grows exponentially with the angle
            radius = min_radius * np.exp(theta / growth_rate)
            
            # Ensure radius doesn't exceed max_radius
            radius = min(radius, max_radius * 0.95)
            
            # Convert to Cartesian coordinates
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            
            # Clamp to the unit square
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            # Store the point
            sequence[point_index, 0] = x
            sequence[point_index, 1] = y
            point_index += 1
            
            # Safety check
            if point_index >= n:
                break
                
        if point_index >= n:
            break
    
    # If we need to fill more dimensions beyond 2D
    if dim > 2:
        # Select algorithm based on dimensionality
        # Halton generally better for medium dimensions, Sobol for higher
        if dim <= 10:
            # Halton sequence for dimensions 3-10
            halton_engine = qmc.Halton(d=dim-2)
            higher_dims = halton_engine.random(n)
        else:
            # Sobol sequence better for very high dimensions
            sobol_engine = qmc.Sobol(d=dim-2, scramble=True)
            higher_dims = sobol_engine.random(n)
            
        sequence[:, 2:] = higher_dims
    
    # Apply adaptive stratification to improve discrepancy metrics
    if optimization_param > 0:
        # Scale stratification strength with dimension
        # Higher dimensions benefit from stronger stratification
        strat_factor = 0.1 * (1 + 0.05 * np.sqrt(dim))
        move_factor = strat_factor * optimization_param
        
        for d in range(dim):
            indices = np.argsort(sequence[:, d])
            sorted_values = sequence[indices, d].copy()
            
            # Create stratified values
            stratified = np.linspace(0, 1, n)
            
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
    golden_spiral_seq = golden_spiral_sequence(dim, n, alpha)
    
    # Compute discrepancies
    print("Computing L2-star discrepancies...")
    sobol_l2 = safe_discrepancy(sobol_seq, 'L2-star')
    halton_l2 = safe_discrepancy(halton_seq, 'L2-star')
    gs_l2 = safe_discrepancy(golden_spiral_seq, 'L2-star')
    
    print("Computing CD discrepancies...")
    sobol_cd = safe_discrepancy(sobol_seq, 'CD')
    halton_cd = safe_discrepancy(halton_seq, 'CD')
    gs_cd = safe_discrepancy(golden_spiral_seq, 'CD')
    
    print("Computing MD discrepancies...")
    sobol_md = safe_discrepancy(sobol_seq, 'MD')
    halton_md = safe_discrepancy(halton_seq, 'MD')
    gs_md = safe_discrepancy(golden_spiral_seq, 'MD')
    
    # Store results directly in Python lists
    sequence_names = ["Sobol", "Halton", "Golden Spiral"]
    l2_values = [sobol_l2, halton_l2, gs_l2]
    cd_values = [sobol_cd, halton_cd, gs_cd]
    md_values = [sobol_md, halton_md, gs_md]
    
    return sequence_names, l2_values, cd_values, md_values, sobol_seq, halton_seq, golden_spiral_seq

def plot_sequences(sobol_seq, halton_seq, golden_spiral_seq, save_path=None):
    """Plot the different sequences for visual comparison."""
    if sobol_seq.shape[1] < 2:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        indices = np.arange(len(sobol_seq))
        axs[0].scatter(indices, sobol_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence')
        axs[1].scatter(indices, halton_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence')
        axs[2].scatter(indices, golden_spiral_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[2].set_title('Golden Spiral Sequence')
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence')
        axs[1].scatter(halton_seq[:, 0], halton_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence')
        axs[2].scatter(golden_spiral_seq[:, 0], golden_spiral_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[2].set_title('Golden Spiral Sequence')
    
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
    dim = 20
    n = 1024  # Use a power of 2 to avoid Sobol warning
    alpha = 0.1
    
    print(f"Computing discrepancies for {n} points in {dim} dimensions with alpha={alpha}")
    
    # Compute discrepancies
    sequence_names, l2_values, cd_values, md_values, sobol_seq, halton_seq, golden_spiral_seq = compute_discrepancies(dim, n, alpha)
    
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
    plot_sequences(sobol_seq, halton_seq, golden_spiral_seq, "sequence_comparison.png")
    
    print("\nSequence visualization saved to 'sequence_comparison.png'")

if __name__ == "__main__":
    main()
