import numpy as np
from scipy.stats import qmc
from numpy import sqrt 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.qmc import discrepancy
import streamlit as st


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
    

    # Create DataFrame
    data = {
        'L2-star': [sobol_disc_l2star, halton_disc_l2star, combined_disc_l2star],
        'CD': [sobol_disc_cd, halton_disc_cd, combined_disc_cd],
        'MD': [sobol_disc_md, halton_disc_md, combined_disc_md]
    }
    df = pd.DataFrame(data, index=['Sobol', 'Halton', 'Tripathi-Sharma Sobolton Sequence'])
    return df,sobol_seq, halton_seq, combined_seq


def plot_sequences(sobol_seq, halton_seq, combined_seq):
    fig, axs = plt.subplots(1, 3, figsize=(20,5), dpi=300)
    axs[0].scatter(sobol_seq[:, 0], sobol_seq[:, 1], c='b', alpha=0.3, s=2, label='Sobol Sequence')
    axs[0].set_title('Sobol Sequence ')
    axs[1].scatter(halton_seq[:, 0], halton_seq[:, 1], c='b', alpha=0.3, s=2,label='Halton Sequence')
    axs[1].set_title('Halton Sequence ')
    axs[2].scatter(combined_seq[:, 0], combined_seq[:, 1], c='b', alpha=0.3, s=2,label='Tripathi-Sharma Sobolton Sequence')
    axs[2].set_title('Tripathi-Sharma Sobolton Sequence')
    plt.show()

def main(n_dim, n,alpha):
    # Compute the discrepancies and get the sequences
    df, sobol_seq, halton_seq, combined_seq = compute_discrepancies(n_dim, n, alpha)
    # Plot the sequences
    plot_sequences(sobol_seq, halton_seq, combined_seq)
    # Print the discrepancies dataframe
    print(df)

# Define the streamlit app
def app():
    st.title("Sobol and Halton Sequences Demo")
    
    # Define the input parameters using streamlit sliders
    n_dim = st.slider('n_dim', min_value=2, max_value=10, step=1, value=2)
    alpha = st.slider('alpha', min_value=0.0, max_value=2.0, step=0.5, value=2)
    n = st.slider('n', min_value=100, max_value=5000, step=100, value=5000)
    
    # Create a button to call the `main` function
    if st.button("Compute Discrepancies"):
        # Call the `main` function with the input parameters
        main(n_dim, n, alpha)

# Run the streamlit app
if __name__ == '__main__':
    app()
