# Tripathi-Sharma Sequence

## Overview
The Tripathi-Sharma sequence is a novel quasi-random point distribution algorithm developed since 2020. It combines biomimicry principles from natural patterns with mathematical optimization techniques to create high-quality, low-discrepancy sequences that outperform traditional methods in various numerical applications.
The application is already better than sobol sequence on several parameters, as shown below. It can be applied to Quantitative Finance for pricing options and assessing risks where balanced sampling is essential for accurate predictions. Moreover, it can he helpful in approximating integrals.

![image](https://github.com/user-attachments/assets/8986bbd0-a925-48be-b909-a9afae4b634c)


## Key Innovations
The Tripathi-Sharma sequence draws inspiration from the remarkable geometric properties found in nature, specifically the arrangement of seeds in sunflowers following the golden angle (approximately 137.5°). This pattern, optimized through millions of years of evolution, creates an exceptionally efficient space-filling distribution.

## Core Principles
- The project takes its main inspiration from the works of Etérea and Cristóbal Vila (https://etereaestudios.com/works/nature-by-numbers/) and Sobol (https://en.wikipedia.org/wiki/Sobol_sequence). 
- I have summarized the code in an excel sheet for the ease of understanding for those who are beginners in Quansi Monte Carlo Sequences.If you want to understand more about it then refer the excel sheet here: [Download Excel](https://github.com/bhaskatripathi/tripathi_sharma_sequence/blob/main/Golden_Ratio_Sunflower_radians.xlsx) ![image](https://github.com/user-attachments/assets/35c3d6a1-f33d-43c6-b298-86ba5911fa28)

- **Biomimetic Design**: I utilize the sunflower seed arrangement pattern to construct QCMC low discrepnacy sequences.
- **Multi-scale Structure**: Combines local pattern optimization with global space-filling properties.
- **Hybrid Approach**: Integrates deterministic patterns with quasi-random filling strategies.
- **Adaptive Parameters**: All internal parameters are dynamically calculated based on problem dimensions.
- **Square-Optimized Distribution**: Modifies the naturally circular pattern to better fill square computational domains.

## Performance Advantages
The Tripathi-Sharma sequence demonstrates superior performance over traditional sequences (Sobol, Halton) in several important metrics:
- **Numerical Integration**: Lower integration error on practical test functions.
- **Centered Variation**: Better distribution around central regions.
- **Symmetric Discrepancy**: Improved invariance to geometric transformations.
- **Visual Uniformity**: More aesthetically balanced distribution with fewer visible patterns.

## When to Prefer Tripathi-Sharma Over Sobol
The Tripathi-Sharma sequence is particularly advantageous in the following scenarios:
- **Numerical Integration Problems**: Especially for functions with central features or symmetric properties.
- **Computer Graphics Applications**:
  - Monte Carlo rendering
  - Texture synthesis
  - Procedural pattern generation
  - Dithering and sampling
- **Financial Simulations**: For option pricing and risk assessment models where balanced sampling is critical.
- **Machine Learning**: For initialization of weights or sampling strategies in probabilistic models.
- **Scientific Visualization**: Creating visually balanced point distributions for data representation.

## Mathematical Foundation
The sequence generation combines several mathematical principles:
- **Golden Ratio Properties**: Utilizes the golden ratio (φ ≈ 1.618) and golden angle (2π(1-1/φ) ≈ 137.5°).
- **Adaptive Grid Structure**: Creates a multi-scale grid with locally optimized patterns.
- **Dynamic Parameter Scaling**: All parameters scale with dimension count and point count using mathematical principles rather than hardcoded values.
- **Square Mapping**: Transforms polar coordinates to square coordinates for better domain filling.
- **Stratification**: Applies light stratification for improved distribution properties.

## Implementation Details
The algorithm follows a sophisticated process:
- Divides the domain into a grid of cells.
- Places square-adapted sunflower patterns within each cell.
- Uses dynamically calculated parameters based on dimension and point count.
- Fills remaining gaps with Sobol points while maintaining minimum separation.
- Extends to higher dimensions by using traditional low-discrepancy sequences for dimensions beyond 2.
- Applies light stratification to improve overall distribution.


