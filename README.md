# ZetaReg

ZetaReg is a Julia package designed to implement and evaluate several statistical models for the Zeta (or Zipf) distribution, including QDE, ZetaMeanModel, and ZetaMeanLogModel.


## Installation

To install the package, you can use Julia's package manager:

```julia
using Pkg
Pkg.add("https://github.com/Xunjian-Li/ZetaReg.git")

- [Usage](#usage)
- [Models Overview](#models-overview)
- [Examples](#examples)

## Usage

After installing the package, you can start using it by including it in your Julia code. Below are the steps to load data and call the models.

### Loading the Package and Data

```julia
using ZetaReg  # Load the ZetaReg package

# Access data from the InsuranceData module
Z = InsuranceData.num_insurances  # Insurance data matrix
W = InsuranceData.age  # Age data vector



# Models Overview

The package implements three main models:

1. QDE (Quantile Difference Estimation)

The QDE model focuses on calculating the quantile differences for the given data and estimating the parameters based on the analysis.

2. ZetaMeanModel

This model uses the Zeta distribution to calculate the mean based on the input data.

3. ZetaMeanLogModel

This model extends the ZetaMeanModel by incorporating a logarithmic transformation to better fit the data, particularly when the data is highly skewed.

## Installation

```julia

using ZetaReg

# Load data
Z = InsuranceData.num_insurances
W = InsuranceData.age

# Call models
(rho1, iters1, loglikelihoods1) = QDE(Z, W)
(rho2, iters2, loglikelihoods2) = ZetaMeanModel(Z, W)
(rho3, iters3, loglikelihoods3) = ZetaMeanLogModel(Z, W)

# Print results
println("Comparison of Results:")
println("===============================================")
println("Method           | β (first) | θ (min) | iters | log-likelihood (last)")
println("===============================================")
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "QDE", rho1[1], minimum(θ1), iters1, loglikelihoods1[end])
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanModel", rho2[1], minimum(θ2), iters2, loglikelihoods2[end])
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanLogModel", rho3[1], minimum(θ3), iters3, loglikelihoods3[end])
println("===============================================")
