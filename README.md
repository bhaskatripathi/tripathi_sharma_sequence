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
- **Gradient Free Optimization**: Tripathi-Sharma sequence can be used to initialize the population in particle-based meta-heuristic methods such as PSO, GWO, and Genetic Algorithms to improve their performance.

## Mathematical Foundation
# Mathematical Foundation of the Tripathi-Sharma Sequence

The **Tripathi-Sharma Sequence** draws from natural patterns and mathematical principles to achieve a quasi-random point distribution. The main ideas behind the sequence are:

## 1. **Golden Angle and Golden Ratio**:
The sequence leverages the **golden angle** (137.5°) and **golden ratio** (φ ≈ 1.618) as a key component. These values have been used in nature to create efficient space-filling patterns, such as the arrangement of seeds in a sunflower. The golden angle and ratio help achieve uniform distribution over the unit square or higher-dimensional domains.

- Golden angle: 
  $$\theta = 137.5^\circ = 2\pi(1 - \frac{1}{\phi})$$

- Golden ratio (φ):
  $$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618$$

## 2. **Sunflower Pattern Adaptation**:
The sequence adapts the **sunflower seed arrangement** (which follows the golden angle) into a computational domain, dividing the space into grids where each grid contains a pattern of sunflower-like points. This results in highly efficient local filling.

The points within a sunflower pattern are generated by spiraling outward, with each successive point placed at a specific angle and distance. This spiral pattern ensures that points are distributed with minimal overlap and efficient coverage.

## 3. **Hybrid Sequence Generation**:
The **Tripathi-Sharma sequence** is hybrid in nature, combining deterministic sunflower patterns with **quasi-random** Sobol points to fill the remaining gaps. This hybrid approach helps to achieve both local and global distribution properties:
  - **Sunflower clusters** fill about 70% of the domain, creating dense patterns.
  - **Sobol sequence points** fill the remaining 30%, ensuring low discrepancy and more uniform coverage.

### Hybrid Sequence Approach:
- **Sunflower Patterns**: Placed based on the golden angle and ratio.
- **Sobol Points**: Used to fill gaps between the sunflower clusters, ensuring a well-balanced distribution.

## 4. **Adaptive Parameters**:
All internal parameters are dynamically calculated based on the problem's **dimensions** and **point count**. This ensures that the sequence adapts to different computational problems without hardcoding values. The algorithm adjusts the grid size, number of sunflower centers, and Sobol filling ratio based on the problem's complexity.

### Main Intution:
The pattern for each sunflower point is based on the below equations. Basically we take the distance of each seed from the center and then theta times the golden angle radians for both co-ordinates, then we plot them on a scatter plot of 3X3 grid. It is quite simplistic implementation in that sense:
  
 -  $$x=\text{center}_x + r \cdot \cos(\theta)$$ \
  - $$y=\text{center}_y + r \cdot \sin(\theta)$$ \\
  where $$\( r \)$$ is the radius (calculated based on the point's index) and \( \theta \) is the angle (based on the golden angle).

## 5. **Square-Optimized Distribution**:
To improve the sequence's performance in square domains, the naturally circular sunflower pattern is modified. The points are mapped to square coordinates, improving coverage for non-circular domains. This adjustment ensures better filling of the computational space, especially for problems involving square or rectangular regions.

## 6. **Dynamic Scaling and Stratification**:
- The algorithm applies **light stratification** to ensure the points are evenly distributed across the domain.
- It also scales the points dynamically based on the dimension count and number of points, ensuring the sequence remains optimal across different problem sizes.

### Stratification Process:
- **Sort points** in each dimension.
- Move them toward a **stratified** distribution to further improve uniformity.

### Final Sequence Generation:
After applying sunflower patterns and Sobol points, the sequence is refined by stratification and dynamic scaling to ensure that the points are well-distributed, especially in higher-dimensional spaces.

### Technical Formulation:
- Golden ratio (φ):
  $$ \phi = \frac{1 + \sqrt{5}}{2} $$

- Golden angle (θ):
  $$ \theta = 137.5^\circ $$

- Point placement in sunflower patterns:
  $$ x = \text{center}_x + r \cdot \cos(\theta) $$
  $$ y = \text{center}_y + r \cdot \sin(\theta) $$
  


