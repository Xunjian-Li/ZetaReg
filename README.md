# ZetaReg

ZetaReg is a Julia package designed to implement and evaluate several statistical models for the Zeta (or Zipf) distribution, including QDE, ZetaMeanModel, and ZetaMeanLogModel.

## Models Overview

The package implements three main models:

### QDE (Quadratic distance estimation) [1]

The Quadratic Distance Estimator (QDE) is an alternative parameter estimation method based on an iteratively reweighted least-squares algorithm, which is consistent, asymptotically unbiased, and normally distributed.

### ZetaMeanModel (Zeta mean regression model via MLE)

This model connects the expectation of the Zeta distribution with the corresponding covariates, and parameter estimation is performed using the maximum likelihood estimation (MLE) method.


### ZetaMeanLogModel (Zeta mean log regression model via MLE)

This model links the expectation of the log-Zeta, where the random variable Zeta follows a Zeta distribution, with the corresponding covariates, and parameter estimation is carried out using the maximum likelihood estimation (MLE) method.

## Installation

To install the package, you can use Julia's package manager:

```julia
using Pkg
Pkg.add("https://github.com/Xunjian-Li/ZetaReg.git")
```

## Usage

After installing the package, you can start using it by including it in your Julia code. Below are the steps to load data and call the models.

### Loading the Package and Data

```julia
using ZetaReg  # Load the ZetaReg package

# Access data from the InsuranceData module
Z = InsuranceData.num_insurances  # Insurance data matrix
W = InsuranceData.age  # Age data vector
```

### Call models

```julia
(rho1, iters1, loglikelihoods1) = QDE(Z, W)
(rho2, iters2, loglikelihoods2) = ZetaMeanModel(Z, W)
(rho3, iters3, loglikelihoods3) = ZetaMeanLogModel(Z, W)
```

### Print results

```julia
println("Comparison of Results:")
println("===============================================")
println("Method           | β (first) | θ (min) | iters | log-likelihood (last)")
println("===============================================")
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "QDE", rho1[1], minimum(θ1), iters1, loglikelihoods1[end])
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanModel", rho2[1], minimum(θ2), iters2, loglikelihoods2[end])
@printf("%-17s | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanLogModel", rho3[1], minimum(θ3), iters3, loglikelihoods3[end])
println("===============================================")
```



## References

[1] Doray L, Arsenault M (2002). Estimators of the regression parameters of the zeta distribution. Insurance: Mathematics and Economics: 30(3), 439-450.
