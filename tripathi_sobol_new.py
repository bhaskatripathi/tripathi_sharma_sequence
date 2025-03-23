"""
Main motivation is this blog post: https://etereaestudios.com/works/nature-by-numbers/

"""
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

# Helper function to compute Fibonacci numbers
def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b

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
    
    # Compute standard discrepancies
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
    
    print("Computing WD discrepancies (wrap-around)...")
    sobol_wd = safe_discrepancy(sobol_seq, 'WD')
    halton_wd = safe_discrepancy(halton_seq, 'WD')
    ts_wd = safe_discrepancy(tripathi_sharma_seq, 'WD')
    
    print("Computing custom symmetric discrepancy...")
    sobol_sym = compute_symmetric_discrepancy(sobol_seq)
    halton_sym = compute_symmetric_discrepancy(halton_seq)
    ts_sym = compute_symmetric_discrepancy(tripathi_sharma_seq)
    
    print("Computing centered variation metric...")
    sobol_center = compute_centered_variations(sobol_seq)
    halton_center = compute_centered_variations(halton_seq)
    ts_center = compute_centered_variations(tripathi_sharma_seq)
    
    print("Computing first integration test error...")
    sobol_int = integration_test_error(sobol_seq)
    halton_int = integration_test_error(halton_seq)
    ts_int = integration_test_error(tripathi_sharma_seq)
    
    print("Computing second integration test error...")
    sobol_int2 = integration_test_error2(sobol_seq)
    halton_int2 = integration_test_error2(halton_seq)
    ts_int2 = integration_test_error2(tripathi_sharma_seq)
    
    # Store results directly in Python lists
    sequence_names = ["Sobol", "Halton", "Tripathi-Sharma"]
    l2_values = [sobol_l2, halton_l2, ts_l2]
    cd_values = [sobol_cd, halton_cd, ts_cd]
    md_values = [sobol_md, halton_md, ts_md]
    wd_values = [sobol_wd, halton_wd, ts_wd]
    sym_values = [sobol_sym, halton_sym, ts_sym]
    center_values = [sobol_center, halton_center, ts_center]
    int_values = [sobol_int, halton_int, ts_int]
    int2_values = [sobol_int2, halton_int2, ts_int2]
    
    return (sequence_names, l2_values, cd_values, md_values, wd_values, 
            sym_values, center_values, int_values, int2_values,
            sobol_seq, halton_seq, tripathi_sharma_seq)

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

def compute_star_discrepancy(points):
    """Approximate the star discrepancy using WD method (as close alternative)"""
    try:
        # Since 'star' is not available, we'll use WD (Wrap-around Discrepancy) as an alternative
        from scipy.stats.qmc import discrepancy
        return safe_discrepancy(points, 'WD')
    except Exception as e:
        print(f"Error computing star discrepancy alternative: {str(e)}")
        return float('nan')

def compute_wraparound_discrepancy(points):
    """Compute the wrap-around discrepancy using the WD method"""
    try:
        # Use the WD method which is available
        from scipy.stats.qmc import discrepancy
        return safe_discrepancy(points, 'WD')  # WD is wrap-around discrepancy
    except Exception as e:
        print(f"Error computing wrap-around discrepancy: {str(e)}")
        return float('nan')

def compute_symmetric_discrepancy(points):
    """Custom implementation of symmetric discrepancy since not available in scipy"""
    try:
        # Simple approximation of symmetric discrepancy
        # This is a simplified version that won't match the theoretical definition
        # but provides a measure that is somewhat symmetrically invariant
        n, d = points.shape
        
        # Compute distances between points
        distances = []
        for i in range(min(1000, n)):  # Limit to 1000 points for computational efficiency
            for j in range(i+1, min(1000, n)):
                # Compute symmetric distance (min of direct and reflected)
                dist = 0
                for k in range(d):
                    # Regular distance
                    direct_dist = abs(points[i, k] - points[j, k])
                    # Wrapped distance (simulate reflection)
                    wrapped_dist = min(direct_dist, 1 - direct_dist)
                    dist += wrapped_dist**2
                distances.append(np.sqrt(dist))
        
        # Return mean distance as a measure (smaller is better for uniformity)
        return float(np.mean(distances))
    except Exception as e:
        print(f"Error computing custom symmetric discrepancy: {str(e)}")
        return float('nan')

def compute_centered_variations(points):
    """Compute alternative to centered discrepancy using L2-star and manual centering"""
    try:
        # Since L2-center isn't available, we'll use a manual approach
        n, d = points.shape
        
        # Center the points around 0.5
        centered_points = np.abs(points - 0.5)
        
        # Use L2-star on these centered points
        from scipy.stats.qmc import discrepancy
        return safe_discrepancy(centered_points, 'L2-star')
    except Exception as e:
        print(f"Error computing centered variation: {str(e)}")
        return float('nan')

def integration_test_error(points):
    """Compute integration error on a suite of test functions"""
    try:
        # Example test function (Genz corner peak function)
        def corner_peak(x, a=1.0):
            return 1.0 / (1.0 + sum(a * x))
        
        # True value (analytically determined)
        true_value = 0.6931471805599453  # ln(2) for a=1.0
        
        # Estimated value
        estimate = np.mean([corner_peak(point) for point in points])
        
        # Absolute error
        return abs(estimate - true_value)
    
    except Exception as e:
        print(f"Error computing integration test error: {str(e)}")
        return float('nan')

# Add a second integration test function to make the evaluation more robust
def integration_test_error2(points):
    """Compute integration error on a second test function (Gaussian peak)"""
    try:
        # Define sigma first
        sigma = 0.1
        
        # Gaussian peak function
        def gaussian_peak(x):
            return np.exp(-sum(((xi-0.5)/sigma)**2 for xi in x)/2)
        
        # For d dimensions, true value requires computing a d-dimensional integral
        d = points.shape[1]
        true_value = (sigma * np.sqrt(2 * np.pi))**d  # This is approximate
        
        # Estimated value
        estimate = np.mean([gaussian_peak(point) for point in points])
        
        # Absolute error
        return abs(estimate - true_value)
    except Exception as e:
        print(f"Error computing second integration test: {str(e)}")
        return float('nan')



def tripathi_sharma_sequence1(dim, n, optimization_param=0.7):
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
    points_per_sunflower = 40 #dim*n
    grid_size = max(3, int(np.sqrt(n / points_per_sunflower)))
    
    # Create centers for the sunflower patterns
    sunflower_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (j + 0.5) / grid_size
            center_y = (i + 0.5) / grid_size
            sunflower_centers.append((center_x, center_y))
    
    # Calculate sunflower coverage - use 70% of points for sunflowers, 30% for Sobol infill
    sunflower_points = int(n * 0.4)
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

def tripathi_sharma_sequence(dim, n, optimization_param=0.7):
    """
    Generate a hybrid Tripathi-Sharma sequence using a grid of square sunflower patterns
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
    
    # Determine grid size for square sunflower patterns
    points_per_sunflower = 4
    grid_size = max(3, int(np.sqrt(n / points_per_sunflower)))
    
    # Create centers for the square sunflower patterns
    sunflower_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = (j + 0.5) / grid_size
            center_y = (i + 0.5) / grid_size
            sunflower_centers.append((center_x, center_y))
    
    # Calculate sunflower coverage - use 70% of points for sunflowers, 30% for Sobol infill
    sunflower_points = int(n * 0.4)
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
        
        # Calculate cell size - the area of each grid cell
        cell_size = 1.0 / grid_size
        # Make pattern slightly smaller to leave gaps between clusters
        pattern_size = 0.85 * cell_size
        
        # Generate square sunflower pattern
        for j in range(this_pattern_points):
            # Adjust angle based on forward/reverse setting
            if is_reverse:
                angle = -j * golden_angle
            else:
                angle = j * golden_angle
            
            # Calculate position using square mapping instead of circular
            # For a square distribution, we'll use a different radius calculation
            # and then map from polar to square coordinates
            
            # First get the radius as a fraction of the pattern size (0 to 1)
            radius_fraction = np.sqrt(j / this_pattern_points)
            
            # Convert polar coordinates to unit square using a square mapping
            # This spreads points more evenly in the corners
            theta = angle
            
            # Map radius and angle to square coordinates
            # Method 1: Use max projection for better corner filling
            if abs(np.cos(theta)) >= abs(np.sin(theta)):
                # Along x-axis dominant direction
                x_offset = np.sign(np.cos(theta)) * radius_fraction
                y_offset = np.tan(theta) * x_offset
            else:
                # Along y-axis dominant direction
                y_offset = np.sign(np.sin(theta)) * radius_fraction
                x_offset = y_offset / np.tan(theta) if np.tan(theta) != 0 else 0
            
            # Scale by pattern size and position relative to center
            x = center_x + x_offset * (pattern_size/2)
            y = center_y + y_offset * (pattern_size/2)
            
            # Ensure point is in domain
            if 0 <= x <= 1 and 0 <= y <= 1:
                sequence[point_index, 0] = x
                sequence[point_index, 1] = y
                
                # Mark this region as occupied
                grid_x = int(x * 99)
                grid_y = int(y * 99)
                if 0 <= grid_x < 100 and 0 <= grid_y < 100:
                    occupied_mask[grid_y, grid_x] = True
                
                # Move to next point
                point_index += 1
                if point_index >= sunflower_points:
                    break
        
        if point_index >= sunflower_points:
            break
    
    # Fill the remaining spaces with Sobol sequence (keep the rest of the function the same)
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


def main():
    # Parameters
    dim = 5
    n = 1024  # Use a power of 2 to avoid Sobol warning
    alpha = 0.5
    
    print(f"Computing discrepancies for {n} points in {dim} dimensions with alpha={alpha}")
    
    # Compute discrepancies with all the new measures
    (sequence_names, l2_values, cd_values, md_values, wd_values, 
     sym_values, center_values, int_values, int2_values,
     sobol_seq, halton_seq, tripathi_sharma_seq) = compute_discrepancies(dim, n, alpha)
    
    # Print results without using pandas
    print("\nDiscrepancy Results:")
    headers = ['Sequence', 'L2-star', 'CD', 'MD', 'WD (Wrap)', 
               'Symmetric', 'Center Var', 'Int. Error 1', 'Int. Error 2']
    header_str = " ".join([f"{h:<15}" for h in headers])
    print(header_str)
    print("-" * (15 * len(headers)))
    
    # Print all metric values for each sequence
    for i, name in enumerate(sequence_names):
        values = [
            name, 
            l2_values[i], 
            cd_values[i], 
            md_values[i],
            wd_values[i],
            sym_values[i],
            center_values[i],
            int_values[i],
            int2_values[i]
        ]
        # Format each value with appropriate precision - increase to 12 decimal places
        formatted_values = [f"{values[0]:<15}"] + [f"{v:<15.12f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"{'N/A':<15}" for v in values[1:]]
        print(" ".join(formatted_values))
    
    # Find the best sequence for each metric
    metrics = [
        ("L2-star", l2_values),
        ("CD", cd_values),
        ("MD", md_values),
        ("WD (Wrap)", wd_values),
        ("Symmetric", sym_values),
        ("Center Var", center_values),
        ("Int. Error 1", int_values),
        ("Int. Error 2", int2_values)
    ]
    
    print("\nBest Sequences:")
    for metric_name, metric_values in metrics:
        # Handle potential NaN values when finding minimum
        valid_indices = [i for i, v in enumerate(metric_values) if not (isinstance(v, float) and np.isnan(v))]
        if valid_indices:
            min_idx = valid_indices[0]
            for i in valid_indices:
                if metric_values[i] < metric_values[min_idx]:
                    min_idx = i
            print(f"{metric_name}: {sequence_names[min_idx]} ({metric_values[min_idx]:.12f})")
        else:
            print(f"{metric_name}: No valid measurements")
    
    # Plot sequences
    plot_sequences(sobol_seq, halton_seq, tripathi_sharma_seq, "sequence_comparison.png")
    
    print("\nSequence visualization saved to 'sequence_comparison.png'")
    headers = ['Sequence', 'L2-star', 'CD', 'MD', 'WD (Wrap)', 
               'Symmetric', 'Center Var', 'Int. Error 1', 'Int. Error 2']
    header_str = " ".join([f"{h:<15}" for h in headers])
    print(header_str)
    print("-" * (15 * len(headers)))
    
    # Print all metric values for each sequence
    for i, name in enumerate(sequence_names):
        values = [
            name, 
            l2_values[i], 
            cd_values[i], 
            md_values[i],
            wd_values[i],
            sym_values[i],
            center_values[i],
            int_values[i],
            int2_values[i]
        ]
        # Format each value with appropriate precision - increase to 12 decimal places
        formatted_values = [f"{values[0]:<15}"] + [f"{v:<15.12f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"{'N/A':<15}" for v in values[1:]]
        print(" ".join(formatted_values))
    
    # Find the best sequence for each metric
    metrics = [
        ("L2-star", l2_values),
        ("CD", cd_values),
        ("MD", md_values),
        ("WD (Wrap)", wd_values),
        ("Symmetric", sym_values),
        ("Center Var", center_values),
        ("Int. Error 1", int_values),
        ("Int. Error 2", int2_values)
    ]
    
    print("\nBest Sequences:")
    for metric_name, metric_values in metrics:
        # Handle potential NaN values when finding minimum
        valid_indices = [i for i, v in enumerate(metric_values) if not (isinstance(v, float) and np.isnan(v))]
        if valid_indices:
            min_idx = valid_indices[0]
            for i in valid_indices:
                if metric_values[i] < metric_values[min_idx]:
                    min_idx = i
            print(f"{metric_name}: {sequence_names[min_idx]} ({metric_values[min_idx]:.12f})")
        else:
            print(f"{metric_name}: No valid measurements")
    
    # Plot sequences
    plot_sequences(sobol_seq, halton_seq, tripathi_sharma_seq, "sequence_comparison.png")
    
    print("\nSequence visualization saved to 'sequence_comparison.png'")

if __name__ == "__main__":
    main()
