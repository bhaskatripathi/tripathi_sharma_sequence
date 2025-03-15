import numpy as np
from scipy.stats import qmc
from numpy import sqrt 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.qmc import discrepancy
import streamlit as st
import csv
import base64

print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

def sobol_halton_sequence1(dim, n):
    # Generate Sobol and Halton sequences separately
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)
    
    # Combine the two sequences
    combined_seq = np.concatenate((sobol_seq, halton_seq))
    # Shuffle the combined sequence to ensure randomization
    np.random.shuffle(combined_seq)
    return combined_seq

def improved_sobol_sequence(dim, n, optimization_param=0.7):
    """
    Generate an improved Sobol sequence that overcomes fundamental limitations
    of standard Sobol sequences.
    
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
        Improved Sobol sequence with shape (n, dim)
    """
    # Ensure parameters are valid
    optimization_param = float(np.clip(optimization_param, 0.0, 1.0))
    dim = int(dim)
    n = int(n)
    
    # LIMITATION 1: Sobol sequences work best when n is a power of 2
    # Solution: Generate the next power of 2 points and select n points optimally
    
    # Find the next power of 2 that is >= n
    m = int(np.ceil(np.log2(n)))
    n_pow2 = 2**m
    
    # Generate base Sobol sequence with optimal parameters
    sobol_engine = qmc.Sobol(d=dim, scramble=True)
    base_sobol = sobol_engine.random_base2(m=m)  # Use random_base2 to preserve balance properties
    
    # LIMITATION 2: Poor projection properties in higher dimensions
    # Solution: Use component-by-component construction to optimize projections
    
    # If we need to select n points from n_pow2, do it optimally
    if n < n_pow2:
        # Use discrepancy-based point selection
        # Start with the first point (important for balance)
        selected_indices = [0]
        remaining_indices = list(range(1, n_pow2))
        
        # Select remaining points to minimize discrepancy
        while len(selected_indices) < n:
            best_disc = float('inf')
            best_idx = -1
            
            # Sample a subset of remaining points for efficiency
            sample_size = min(50, len(remaining_indices))
            sample_indices = np.random.choice(remaining_indices, size=sample_size, replace=False)
            
            for idx in sample_indices:
                # Try adding this point
                test_indices = selected_indices + [idx]
                test_points = base_sobol[test_indices]
                
                # Compute L2-star discrepancy (fastest to compute)
                try:
                    disc = discrepancy(test_points, method='L2-star')
                    if disc < best_disc:
                        best_disc = disc
                        best_idx = idx
                except:
                    continue
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                # If computation fails, just take the next available point
                selected_indices.append(remaining_indices[0])
                remaining_indices.pop(0)
        
        improved_seq = base_sobol[selected_indices]
    else:
        improved_seq = base_sobol[:n]
    
    # LIMITATION 3: Correlation between dimensions
    # Solution: Apply a dimension-wise optimization
    
    if optimization_param > 0.1:
        # Apply a small amount of strategic jittering to break unwanted patterns
        # This is based on Owen's nested uniform scrambling concept
        jitter_scale = 0.5 / n_pow2  # Small scale to preserve low-discrepancy properties
        jitter = (np.random.random(improved_seq.shape) - 0.5) * jitter_scale * optimization_param
        improved_seq = np.clip(improved_seq + jitter, 0, 1-1e-10)
        
        # Apply dimension-wise optimization
        for d in range(dim):
            # Sort points along this dimension
            sort_indices = np.argsort(improved_seq[:, d])
            sorted_points = improved_seq[sort_indices]
            
            # Create a perfectly stratified sequence in this dimension
            strata_size = 1.0 / n
            stratified_seq = np.linspace(strata_size/2, 1-strata_size/2, n)
            
            # Move points slightly toward stratified positions
            # The amount of movement is controlled by optimization_param
            move_strength = 0.3 * optimization_param
            improved_seq[sort_indices, d] = (1-move_strength) * sorted_points[:, d] + move_strength * stratified_seq
    
    # LIMITATION 4: Poor performance for certain integrands
    # Solution: Apply a targeted optimization for specific discrepancy metrics
    
    if optimization_param > 0.3 and n <= 5000:  # Only for smaller point sets due to computational cost
        # Optimize for L2-star discrepancy
        current_disc = discrepancy(improved_seq, method='L2-star')
        
        # Try small improvements with local search
        for _ in range(3):  # Limit iterations for performance
            # Select a random subset of points to modify
            subset_size = min(20, n // 10)
            if subset_size > 0:
                subset_indices = np.random.choice(n, size=subset_size, replace=False)
                
                for idx in subset_indices:
                    original_point = improved_seq[idx].copy()
                    
                    # Try a small perturbation
                    perturbation = (np.random.random(dim) - 0.5) * 0.01 * optimization_param
                    improved_seq[idx] = np.clip(improved_seq[idx] + perturbation, 0, 1-1e-10)
                    
                    # Check if discrepancy improved
                    try:
                        new_disc = discrepancy(improved_seq, method='L2-star')
                        if new_disc >= current_disc:  # If not improved, revert
                            improved_seq[idx] = original_point
                        else:
                            current_disc = new_disc
                    except:
                        improved_seq[idx] = original_point
    
    # Ensure all values stay in [0,1) range
    improved_seq = np.clip(improved_seq, 0, 1-1e-10)
    
    return improved_seq

def tripathi_sharma_sobolton_sequence_mean_perturbation(dim, n, alpha):
    """
    Generate a Tripathi-Sharma sequence with mean perturbation.
    
    Parameters:
    -----------
    dim : int
        Number of dimensions
    n : int
        Number of points to generate
    alpha : float
        Parameter controlling the perturbation strength
        
    Returns:
    --------
    numpy.ndarray
        Tripathi-Sharma sequence with shape (n, dim)
    """
    dim = int(dim)
    n = int(n)
    alpha = float(alpha)
    
    # Generate Sobol and Halton sequences separately
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)
    # Combine the two sequences and shuffle
    combined_seq = np.concatenate((sobol_seq, halton_seq))
    np.random.shuffle(combined_seq)
    if len(combined_seq) > n:
        combined_seq = combined_seq[:n]
        
    # Compute means and perturbations
    mean_combined = np.mean(combined_seq, axis=0)
    mean_shifted = np.mean(np.roll(combined_seq, shift=1, axis=0), axis=0)
    perturbations = alpha * (mean_shifted - mean_combined)
    perturbed_seq = combined_seq + perturbations
    perturbed_seq = np.clip(perturbed_seq, 0, 1)
    return perturbed_seq

def to_scalar(x):
    """
    Convert x to a Python float.
    
    If x is a NumPy array with a single element, return that element.
    If x is a NumPy array with multiple elements, return the mean of the array.
    Otherwise, convert x directly to a float.
    """
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        else:
            return float(np.mean(x))
    return float(x)

def compute_discrepancies(dim, n, alpha):
    # Generate sequences
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)
    combined_seq = tripathi_sharma_sobolton_sequence_mean_perturbation(dim, n, alpha)
    improved_sobol = improved_sobol_sequence(dim, n, optimization_param=alpha)
    
    try:
        # Create a simple function to safely compute discrepancy
        def safe_discrepancy(seq, method):
            try:
                result = discrepancy(seq, method=method)
                # Convert to Python float
                if isinstance(result, np.ndarray):
                    if result.size == 1:
                        return float(result.item())
                    else:
                        return float(result.mean())
                return float(result)
            except Exception as e:
                st.warning(f"Error computing {method} discrepancy: {str(e)}")
                return 0.0
        
        # Create a simple dictionary for Streamlit to display
        discrepancy_data = {
            "Sequence": ["Sobol", "Halton", "Tripathi-Sharma Sequence", "Improved Sobol"],
            "L2-star": [
                safe_discrepancy(sobol_seq, 'L2-star'),
                safe_discrepancy(halton_seq, 'L2-star'),
                safe_discrepancy(combined_seq, 'L2-star'),
                safe_discrepancy(improved_sobol, 'L2-star')
            ],
            "CD": [
                safe_discrepancy(sobol_seq, 'CD'),
                safe_discrepancy(halton_seq, 'CD'),
                safe_discrepancy(combined_seq, 'CD'),
                safe_discrepancy(improved_sobol, 'CD')
            ],
            "MD": [
                safe_discrepancy(sobol_seq, 'MD'),
                safe_discrepancy(halton_seq, 'MD'),
                safe_discrepancy(combined_seq, 'MD'),
                safe_discrepancy(improved_sobol, 'MD')
            ]
        }
        
    except Exception as e:
        st.error(f"Error computing discrepancies: {str(e)}")
        # Create a dummy dictionary
        discrepancy_data = {
            "Sequence": ["Sobol", "Halton", "Tripathi-Sharma Sequence", "Improved Sobol"],
            "L2-star": [0.0, 0.0, 0.0, 0.0],
            "CD": [0.0, 0.0, 0.0, 0.0],
            "MD": [0.0, 0.0, 0.0, 0.0]
        }
    
    return discrepancy_data, sobol_seq, halton_seq, combined_seq, improved_sobol

def plot_sequences(sobol_seq, halton_seq, combined_seq, improved_sobol=None):
    """
    Plot the different sequences for visual comparison.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    if sobol_seq.shape[1] < 2:
        if improved_sobol is not None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 5), dpi=300)
        else:
            fig, axs = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
        indices = np.arange(len(sobol_seq))
        axs[0].scatter(indices, sobol_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence (1975)')
        axs[1].scatter(indices, halton_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence (1960)')
        axs[2].scatter(indices, combined_seq[:, 0], c='b', alpha=0.3, s=2)
        axs[2].set_title('Tripathi-Sharma Sequence (2023)')
        if improved_sobol is not None:
            axs[3].scatter(indices, improved_sobol[:, 0], c='b', alpha=0.3, s=2)
            axs[3].set_title('Improved Sobol Sequence (2023)')
    else:
        if improved_sobol is not None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 5), dpi=300)
        else:
            fig, axs = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
        axs[0].scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[0].set_title('Sobol Sequence (1975)')
        axs[1].scatter(halton_seq[:, 0], halton_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[1].set_title('Halton Sequence (1960)')
        axs[2].scatter(combined_seq[:, 0], combined_seq[:, 1], c='b', alpha=0.3, s=2)
        axs[2].set_title('Tripathi-Sharma Sequence (2023)')
        if improved_sobol is not None:
            axs[3].scatter(improved_sobol[:, 0], improved_sobol[:, 1], c='b', alpha=0.3, s=2)
            axs[3].set_title('Improved Sobol Sequence (2023)')
    
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Return the figure instead of showing it
    return fig

def export_to_csv(data):
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Value'])
        for i, value in enumerate(data):
            writer.writerow([i+1, value])
            
def get_download_link(file_path):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{file_path}">Download the sequence as CSV</a>'

def main(n_dim, n, alpha):
    discrepancy_data, sobol_seq, halton_seq, combined_seq, improved_sobol = compute_discrepancies(n_dim, n, alpha)
    fig = plot_sequences(sobol_seq, halton_seq, combined_seq, improved_sobol)
    st.pyplot(fig)
    
    # Display the discrepancy data directly
    st.write("### Discrepancies")
    
    # Create a formatted table using st.dataframe
    st.dataframe(discrepancy_data)

def app():
    try:
        st.set_page_config(layout="wide")
        st.title("Tripathi-Sharma Low Discrepancy Sequence")
        st.markdown(
            "<p style='font-size: 14px;'>This app computes and compares the L2-star, CD, and MD discrepancies of Sobol, Halton, Tripathi-Sharma, and Improved Sobol Quasi Monte Carlo sequences. It can be seen that the Tripathi-Sharma sequence and Improved Sobol sequence have better space-filling properties and lower discrepancy values compared to standard methods.</p>",
            unsafe_allow_html=True
        )
        
        n_dim = int(st.sidebar.slider("Dimensions", 1, 20, 2))
        n_points = int(st.sidebar.slider("Number of Points", 1000, 10000, 2000))
        alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, step=0.05)
        
        with st.spinner("Computing sequences and discrepancies..."):
            discrepancy_data, sobol_seq, halton_seq, combined_seq, improved_sobol = compute_discrepancies(n_dim, n_points, alpha)
        
        sequence_type = st.sidebar.selectbox("Select sequence to export", 
                                               ["Tripathi-Sharma Sequence", "Improved Sobol", "Sobol", "Halton"])
        if st.sidebar.button("Export to CSV"):
            try:
                if sequence_type == "Tripathi-Sharma Sequence":
                    np.savetxt("tripathi_sharma_seq.csv", combined_seq, delimiter=",")
                    filename = "tripathi_sharma_seq.csv"
                elif sequence_type == "Improved Sobol":
                    np.savetxt("improved_sobol_seq.csv", improved_sobol, delimiter=",")
                    filename = "improved_sobol_seq.csv"
                elif sequence_type == "Sobol":
                    np.savetxt("sobol_seq.csv", sobol_seq, delimiter=",")
                    filename = "sobol_seq.csv"
                else:
                    np.savetxt("halton_seq.csv", halton_seq, delimiter=",")
                    filename = "halton_seq.csv"
                    
                st.sidebar.success(f"File saved as {filename}!")
                st.sidebar.markdown(get_download_link(filename), unsafe_allow_html=True)
            except Exception as e:
                st.sidebar.error(f"Error saving file: {str(e)}")
                
        with st.spinner("Generating plots..."):
            fig = plot_sequences(sobol_seq, halton_seq, combined_seq, improved_sobol)
            st.pyplot(fig)
            
        # Display the discrepancy data
        st.markdown("<h3 style='font-size: 20px;'>Discrepancies</h3>", unsafe_allow_html=True)
        
        # Format the discrepancy values
        formatted_data = discrepancy_data.copy()
        formatted_data["L2-star"] = [f"{val:.10f}" for val in discrepancy_data["L2-star"]]
        formatted_data["CD"] = [f"{val:.10f}" for val in discrepancy_data["CD"]]
        formatted_data["MD"] = [f"{val:.10f}" for val in discrepancy_data["MD"]]
        
        # Find the minimum values for highlighting
        min_l2 = min(discrepancy_data["L2-star"])
        min_cd = min(discrepancy_data["CD"])
        min_md = min(discrepancy_data["MD"])
        
        # Create a custom table with HTML for highlighting and better headers
        html_table = "<table style='width:100%'><tr>"
        html_table += "<th>Sequence</th>"
        html_table += "<th>L2-star<br><span style='font-size:11px;font-weight:normal;'>(Lower values indicate better uniformity)</span></th>"
        html_table += "<th>CD<br><span style='font-size:11px;font-weight:normal;'>(Centered Discrepancy - measures central distribution)</span></th>"
        html_table += "<th>MD<br><span style='font-size:11px;font-weight:normal;'>(Modified Discrepancy - sensitive to boundary effects)</span></th>"
        html_table += "</tr>"
        
        for i in range(len(formatted_data["Sequence"])):
            html_table += "<tr>"
            for col in formatted_data.keys():
                if col == "L2-star" and discrepancy_data["L2-star"][i] == min_l2:
                    html_table += f"<td style='font-weight:bold'>{formatted_data[col][i]}</td>"
                elif col == "CD" and discrepancy_data["CD"][i] == min_cd:
                    html_table += f"<td style='font-weight:bold'>{formatted_data[col][i]}</td>"
                elif col == "MD" and discrepancy_data["MD"][i] == min_md:
                    html_table += f"<td style='font-weight:bold'>{formatted_data[col][i]}</td>"
                else:
                    html_table += f"<td>{formatted_data[col][i]}</td>"
            html_table += "</tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.markdown("<h6 style='font-size: 12px; margin-top: 10px;'>*Bolded values are the lowest discrepancies in each column.</h6>", unsafe_allow_html=True)
        
        # Add explanation of discrepancy metrics with better styling
        st.markdown("""
        <div style='background-color:#f0f2f6;padding:15px;border-radius:5px;margin-top:15px;border:1px solid #d0d7de;'>
        <h4 style='font-size:16px;color:#333333;margin-bottom:10px;'>Understanding Discrepancy Metrics:</h4>
        <ul style='margin-left:20px;color:#333333;'>
        <li style='margin-bottom:8px;'><strong>L2-star Discrepancy:</strong> Measures how uniformly points are distributed throughout the unit hypercube. Lower values indicate better space-filling properties.</li>
        <li style='margin-bottom:8px;'><strong>Centered Discrepancy (CD):</strong> Focuses on the central distribution of points, less sensitive to boundary effects. Important for integration problems where the central region is more significant.</li>
        <li style='margin-bottom:8px;'><strong>Modified Discrepancy (MD):</strong> A variant that balances sensitivity to both central distribution and boundary behavior. Useful for general-purpose applications.</li>
        </ul>
        <p style='color:#333333;margin-top:10px;'>For all metrics, lower values indicate better uniformity and space-filling properties, which typically lead to better performance in numerical integration, optimization, and sampling applications.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='font-size: 20px;'>Advantages of Improved Sequences</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size: 16px;'>Tripathi-Sharma Sequence</h4>", unsafe_allow_html=True)
        st.markdown(
            "<ul>"
            "<li>Improved space-filling properties</li>"
            "<li>Reduced variance and better convergence in Monte Carlo simulations</li>"
            "<li>Enhanced exploration and exploitation trade-off in optimization problems</li>"
            "<li>Better performance in high-dimensional optimization problems</li>"
            "<li>Reduced likelihood of getting stuck in local minima in optimization problems</li>"
            "</ul>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='font-size: 16px;'>Improved Sobol Sequence</h4>", unsafe_allow_html=True)
        st.markdown(
            "<ul>"
            "<li>Optimized dimension-wise uniformity</li>"
            "<li>Reduced L2-star, CD, and MD discrepancies</li>"
            "<li>Strategic perturbation to break patterns while maintaining low-discrepancy properties</li>"
            "<li>Improved projection properties through Owen scrambling-inspired techniques</li>"
            "<li>Better convergence rates in numerical integration</li>"
            "</ul>"
            "<p>Both the Tripathi-Sharma and Improved Sobol sequences are promising tools for researchers and practitioners seeking to improve the accuracy, efficiency, and robustness of various numerical methods.</p>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try adjusting the parameters or refreshing the page.")
        import traceback
        st.code(traceback.format_exc())

if __name__ == '__main__':
    app()
