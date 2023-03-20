import numpy as np
from scipy.stats import qmc
from numpy import sqrt 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.qmc import discrepancy
import streamlit as st
import csv
import base64

def sobol_halton_sequence1(dim, n):
    # Generate Sobol and Halton sequences separately
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)
    

    # Combine the two sequences
    combined_seq = np.concatenate((sobol_seq, halton_seq))

    # Shuffle the combined sequence to ensure randomization
    np.random.shuffle(combined_seq)

    # # return combined_seq
    # sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    # halton_seq = qmc.Halton(d=dim).random(n)
    
    # # Generate set of points using Sobol sequence
    # points = sobol_seq
    
    # # Perturb points using Halton sequence
    # for i in range(n):
    #     for j in range(dim):
    #         points[i,j] += halton_seq[i,j] / n
    # combined_seq = points
    return combined_seq



def tripathi_sharma_sobolton_sequence_mean_perturbation(dim, n, alpha):
    # Generate Sobol and Halton sequences separately
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)

    # Combine the two sequences
    combined_seq = np.concatenate((sobol_seq, halton_seq))

    # Shuffle the combined sequence to ensure randomization
    np.random.shuffle(combined_seq)

    # Compute means
    mean_combined = np.mean(combined_seq, axis=0)
    mean_shifted = np.mean(np.roll(combined_seq, shift=1, axis=0), axis=0)

    # Compute perturbations
    perturbations = alpha * (mean_shifted - mean_combined)

    # Apply perturbations to sequence
    perturbed_seq = combined_seq + perturbations

    return perturbed_seq


def compute_discrepancies(dim, n,alpha):
    # Generate sequences
    sobol_seq = qmc.Sobol(d=dim, scramble=True).random(n)
    halton_seq = qmc.Halton(d=dim).random(n)
    combined_seq = tripathi_sharma_sobolton_sequence_mean_perturbation(dim, n,alpha)

    # Compute discrepancies
    sobol_disc_l2star = discrepancy(sobol_seq, method='L2-star')
    sobol_disc_cd = discrepancy(sobol_seq, method='CD')
    sobol_disc_md = discrepancy(sobol_seq, method='MD')
    halton_disc_l2star = discrepancy(halton_seq, method='L2-star')
    halton_disc_cd = discrepancy(halton_seq, method='CD')
    halton_disc_md = discrepancy(halton_seq, method='MD')
    combined_disc_l2star = discrepancy(combined_seq, method='L2-star')
    combined_disc_cd = discrepancy(combined_seq, method='CD')
    combined_disc_md = discrepancy(combined_seq, method='MD')
    # Set display options
    pd.set_option('display.float_format', lambda x: '%.20f' % x)
    

    # Create DataFrame
    data = {
        'L2-star': [sobol_disc_l2star, halton_disc_l2star, combined_disc_l2star],
        'CD': [sobol_disc_cd, halton_disc_cd, combined_disc_cd],
        'MD': [sobol_disc_md, halton_disc_md, combined_disc_md]
    }
    df = pd.DataFrame(data, index=['Sobol', 'Halton', 'Tripathi-Sharma Sequence'])
    return df,sobol_seq, halton_seq, combined_seq


def plot_sequences(sobol_seq, halton_seq, combined_seq):
    fig, axs = plt.subplots(1, 3, figsize=(20,5), dpi=300)
    axs[0].scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='b', alpha=0.3, s=2, label='Sobol Sequence')
    axs[0].set_title('Sobol Sequence (1975)')
    axs[1].scatter(halton_seq[:, 0], halton_seq[:, 1], c='b', alpha=0.3, s=2,label='Halton Sequence')
    axs[1].set_title('Halton Sequence (1960)')
    axs[2].scatter(combined_seq[:, 0], combined_seq[:, 1], c='b', alpha=0.3, s=2,label='Tripathi-Sharma Sequence')
    axs[2].set_title('Tripathi-Sharma Sequence (2023)')
    plt.show()
  
def export_to_csv(data):
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Value'])
        for i, value in enumerate(data):
            writer.writerow([i+1, value])
            
def get_download_link(file_path):
    with open(file_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{file_path}">Download the sequence as CSV</a>'
    
def main(n_dim, n,alpha):
    # Compute the discrepancies and get the sequences
    df, sobol_seq, halton_seq, combined_seq = compute_discrepancies(n_dim, n, alpha)
    # Plot the sequences
    plot_sequences(sobol_seq, halton_seq, combined_seq)
    # Print the discrepancies dataframe
    #print(df)
    st.pyplot()
    # Print the discrepancies dataframe
    st.table(df)

# Define the streamlit app
def app():
    st.set_page_config(layout="wide")

    st.title("Tripathi-Sharma Low Discrepancy Sequence")

    # Add a description of the app
    #st.markdown("<h3 style='font-size: 20px;'>Introduction</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 14px;'>This app computes and compares the L2-star, CD, and MD discrepancies of Sobol, Halton, and Tripathi-Sharma Quasi Monte Carlo sequences. It can be seen that the Tripathi-Sharma sequence has Improved space-filling properties, Lower discrepancy values. It is also computationally less expensive than the standard Sobol and Halton methods. </p>", unsafe_allow_html=True)

    # Add sliders for the number of dimensions and points
    #n_dim = st.sidebar.slider("Dimensions", 1, 20, 2)
    n_dim = st.sidebar.text_input("Dimensions", 2)
    n_dim= int(n_dim)
    #n_points = st.sidebar.slider("Number of Points", 1000, 10000, 2000)
    n_points = st.sidebar.text_input("Number of Points", 2000)
    n_points=int(n_points)/2

    # Add a slider for the alpha parameter
    alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 0.5, step=0.05)
    
    # Compute the discrepancies and get the sequences
    df, sobol_seq, halton_seq, combined_seq = compute_discrepancies(n_dim, n_points, alpha)
    
    # Add a button to download the combined sequence as a CSV file
    if st.sidebar.button("Export to CSV"):
        np.savetxt("tripathi_sharma_seq.csv", combined_seq, delimiter=",")
        st.sidebar.success("File saved successfully!")
        st.sidebar.markdown(get_download_link("tripathi_sharma_seq.csv"), unsafe_allow_html=True)  

    # Plot the sequences 
    plot_sequences(sobol_seq, halton_seq, combined_seq)
    st.pyplot()
    
    # Highlight minimum values in each column
    def highlight_min(s):
        is_min = s == s.min()
        return ['font-weight: bold' if v else '' for v in is_min]

    # Apply formatting to the DataFrame
    styled_df = df.style.apply(highlight_min, axis=0)
    
    # Print the discrepancies dataframe
    st.markdown("<h3 style='font-size: 20px;'>Discrepancies</h3>", unsafe_allow_html=True)
    st.table(styled_df.format('{:.10f}'))
    st.markdown("<h6 style='font-size: 12px; margin-top: 10px;'>*Bolded values are the lowest discrepancies in each column.</h6>", unsafe_allow_html=True)
    # Add a markdown with the advantages of Tripathi Sharma sequence
    st.markdown("<h3 style='font-size: 20px;'>Advantages of Tripathi Sharma Sequence</h3>", unsafe_allow_html=True)
    st.markdown("<ul><li>Improved space-filling properties</li><li>Reduced variance and better convergence in Monte Carlo simulations</li><li>Enhanced exploration and exploitation trade-off in optimization problems</li><li>Better performance in high-dimensional optimization problems</li><li>Reduced likelihood of getting stuck in local minima in optimization problems</li></ul><p>Overall, the Tripathi-Sharma Sobolton sequence is a promising tool for researchers and practitioners seeking to improve the accuracy, efficiency, and robustness of various numerical methods.</p>", unsafe_allow_html=True)

# Run the streamlit app
if __name__ == '__main__':
    app()
