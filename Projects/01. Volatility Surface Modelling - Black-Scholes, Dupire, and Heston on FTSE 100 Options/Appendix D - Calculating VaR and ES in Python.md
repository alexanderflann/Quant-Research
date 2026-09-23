# D. Calculating VaR and ES in Python

Define the Value at Risk and Expected Shortfall, or Conditional Value at Risk in Python.

```python
def var_es(ST, S0, alpha): # Function to calculate the Value at Risk and Expected Shortfall (CVaR)
    loss = -np.log(ST/S0)
    VaR  = np.quantile(loss, alpha)
    ES   = loss[loss >= VaR].mean()
    return VaR, ES
```

Let the time horizon be one month such that $T=\frac{22}{252}$.

```python
# Redefine variables for 1 month horizon
r_f = risk_free_rate.loc[risk_free_rate["Tenor"] == "1M", "Rate"].iloc[0] / 100
N_m = 22 # Typically used for 1 month trading
T_m = N_m/252

# Simulate Heston and GBM benchmark respectively
S_m, v_m = heston_model_sim(S0, v0, rho, kappa, theta, xi, T_m, N_m, M)
gbm_m = S0*np.exp((r_f - q - theta/2)*T_m + np.sqrt(theta)*np.sqrt(T_m)*np.random.normal(0,1,M))

def price_density_with_var(S, gbm, S0, alpha, xlim, title): # Function to display simulations with VaR thresholds

    # Compute VaR and ES
    VaR_h, ES_h = var_es(S[-1], S0, alpha)
    VaR_g, ES_g = var_es(gbm, S0, alpha)

    # Convert VaR back to price threshold
    S_VaR_h = S0 * np.exp(-VaR_h)
    S_VaR_g = S0 * np.exp(-VaR_g)

    # Plot density
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(S[-1], label="Heston", ax=ax)
    sns.kdeplot(gbm,   label="GBM ($\\theta$)", ax=ax)

    # Plot VaR threshold lines
    ax.axvline(S_VaR_h, color='tab:blue', linestyle='--',label=f'Heston VaR: £{S_VaR_h:.2f}')
    ax.axvline(S_VaR_g, color='tab:orange', linestyle='--',label=f'GBM    VaR: £{S_VaR_g:.2f}')

    # Set labels
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_xlabel("$S_T$")
    ax.set_ylabel("Density")
    ax.legend()

    plt.show()

    return (VaR_h, ES_h), (VaR_g, ES_g)
```

Call the function defined above such that plots for $\alpha$, the confidence level, equals 95\% and 99\%.

```python
# Call function for confidence level 95%
(VaR_h, ES_h), (VaR_g, ES_g) = price_density_with_var(S_m, gbm_m, S0, alpha=0.95, xlim=[8000, 12000], title="Asset Price Density with 95% VaR Thresholds")

# Print values of VaR and ES for confidence level 95%
print(f"Heston VaR (95%): {VaR_h:.4f}, ES: {ES_h:.4f}")
print(f"GBM    VaR (95%): {VaR_g:.4f}, ES: {ES_g:.4f}")

# Call function for confidence level 99%
(VaR_h, ES_h), (VaR_g, ES_g) = price_density_with_var(S_m, gbm_m, S0, alpha=0.99, xlim=[8000, 12000], title="Asset Price Density with 99% VaR Thresholds")

# Print values of VaR and ES for confidence level 99%
print(f"Heston VaR (99%): {VaR_h:.4f}, ES: {ES_h:.4f}")
print(f"GBM    VaR (99%): {VaR_g:.4f}, ES: {ES_g:.4f}")
```

Thus, a possible Monte-Carlo output for VaR and ES is:

<p  align="center">
	<img src="https://github.com/alexanderflann/Quant-Research/blob/main/Projects/01.%20Volatility%20Surface%20Modelling%20-%20Black-Scholes%2C%20Dupire%2C%20and%20Heston%20on%20FTSE%20100%20Options/Images/95%20VaR%20Graph.png?raw=true" width="500"/>
	<img src="https://github.com/alexanderflann/Quant-Research/blob/main/Projects/01.%20Volatility%20Surface%20Modelling%20-%20Black-Scholes%2C%20Dupire%2C%20and%20Heston%20on%20FTSE%20100%20Options/Images/99%20VaR%20Graph.png?raw=true" width="500"/>
    <img src="https://github.com/alexanderflann/Quant-Research/blob/main/Projects/01.%20Volatility%20Surface%20Modelling%20-%20Black-Scholes%2C%20Dupire%2C%20and%20Heston%20on%20FTSE%20100%20Options/Images/VaR%20ES%20Values%20Output.png?raw=true" width="300"/>

At the 95\% confidence level, the Heston model implies, with a one month horizon, a VaR of 4.96\% and an ES of 7.35\%, compared with 3.68\% and 4.54\%, respectively, under the GBM benchmark.
For an initial asset price, under Heston, this translates to a 95\% VaR loss of £480 and an ES of £700, while the GBM benchmark yields a VaR of £360 and an ES of around £440. At the 99\% level, Heston produces a VaR of 8.21\% and ES of 11.73\%, versus 5.16\% and 5.96\% for GBM, which in price terms corresponds to losses of approximately £780 and £1100 under Heston, compared with about £500 and £580 under the GBM benchmark.
These results reinforce that, for both 95\% and 99\% confidence levels, the Heston model places considerably more probability in the extreme loss region, for possible extreme market events, than the lognormal GBM model.
