# C. Monte Carlo Simulations for Asset Price and Variance in Python

We can simulate the asset price at maturity and the variance paths via Monte-Carlo simulations. Do note that, under the Heston model, for European options there is a closed-form solution once you have the characteristic function, (33).
Thus, the discretion of the SDE is not required when finding the fair value of a European-style option. For the Heston model, recall the asset dynamics, (21) and (22). Thus, the Euler discretion is:

$$ dS_{i+1} = S_i e^{\left( r - q - \frac{v_i}{2} \right) \Delta t + \sqrt{v_i} \Delta t X_{S, i+1}^{\mathbb{Q}}}, $$
$$ v_{i+1} = v_i + \kappa \left( \theta - v_i \right) \Delta t + \sigma \sqrt{v_i} \Delta t X_{v,i+1}^{\mathbb{Q}}. $$

We first need to import a new module.

```python
import seaborn as sns # Statistical visualisation based on matplotlib
```
	
Now, we can simulate the asset price and variance over a time interval, $T$. In this case, $T=1$ and so the number of time steps, $N$, is $252$, since this is the number of trading days in one year.

```python
r_f = risk_free_rate.loc[risk_free_rate["Tenor"] == "1Y", "Rate"].iloc[0] / 100 # Risk-free flat rate at 1 year maturity
T = 1 # Time in years
N = 252 # Number of time steps
M = 1000 # Number of scenarios/simulations

def heston_model_sim(S0, v0, rho, kappa, theta, xi, T, N, M): # Function to simulate asset prices and volatility using the Heston model

	# Initialise parameters
	dt = T/N
	mu = np.array([0,0])
	cov = np.array([[1,rho],
					[rho,1]])

      # Arrays for storing prices and variances
      S = np.full(shape=(N+1,M), fill_value=S0)
      v = np.full(shape=(N+1,M), fill_value=v0)

      # Sampling correlated brownian motions under the risk-neutral measure
      Z = np.random.multivariate_normal(mu, cov, (N,M))

	for i in range(1,N+1):
		S[i] = S[i-1] * np.exp((r_f - q - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1,:,0])
		v[i] = np.maximum(v[i-1] + kappa*(theta-v[i-1])*dt + xi*np.sqrt(v[i-1]*dt)*Z[i-1,:,1],0)

	return S, v
```

Once the function is defined we can plot the simulated asset prices and variances to visualise the possible paths using the model. Note, we have already found the unknown parameters, $\Theta$, using the ordinary least squared errors approach.

```python
# Define simulated prices and volatility
S_ukx, v_ukx = heston_model_sim(S0, v0, rho, kappa, theta, xi, T, N, M)

# Create plot for Monte-Carlo simulation for the Heston model on asset prices and variance
fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12,5))
time = np.linspace(0,T,N+1)
ax1.plot(time,S_ukx)
ax1.set_title('Heston Model Asset Prices')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')

ax2.plot(time,v_ukx)
ax2.set_title('Heston Model Variance Process')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')

plt.show()
```

To compare, a GBM where $\theta$, the long-run variance, is approximated to be $\sigma^2$ is also plotted, where:

$$ S_t = S_0 e^{\left( r - q - \frac{\theta}{2} \right)T + \sqrt{\theta T} X}, $$

where $X$ follows a standard normal distribution.

```python
# Simulate gbm process at time T, where theta ~ sigma**2
gbm = S0*np.exp((r_f - q - theta/2)*T + np.sqrt(theta)*np.sqrt(T)*np.random.normal(0,1,M))

def price_density(S, gbm, xlim, title): # Function to display asset price density at expiry
	fig, ax = plt.subplots()

	# Create plots
	ax = sns.kdeplot(S[-1], label=rf"$\rho = {rho:.2f}$", ax=ax)
	ax = sns.kdeplot(gbm, label="GBM ($\\theta$)", ax=ax)

	# Set title, labels, limit, and show legend
	ax.set_title(title)
    ax.set_xlim(xlim)
	ax.set_xlabel("$S_T$")
	ax.set_ylabel("Density")
	ax.legend()

	# Show plot
	plt.show()

price_density(S_ukx, gbm, xlim=[4000, 16000], title="Asset Price Density Under the Heston Model") # Call function to display asset price density at expiry
```

Since Monte-Carlo runs multiple paths and we are looking at the stochastic behaviour, the output will always be different, However, one possible output for the code is:

<p  align="center">
	<img src="https://github.com/alexanderflann/Quant-Research/blob/main/Projects/01.%20Volatility%20Surface%20Modelling%20-%20Black-Scholes%2C%20Dupire%2C%20and%20Heston%20on%20FTSE%20100%20Options/Images/Asset%20Price%20and%20Variance%20Paths%20Graph.png?raw=true" width="1000"/>
	<img src="https://github.com/alexanderflann/Quant-Research/blob/main/Projects/01.%20Volatility%20Surface%20Modelling%20-%20Black-Scholes%2C%20Dupire%2C%20and%20Heston%20on%20FTSE%20100%20Options/Images/Asset%20Price%20Density%20Graph.png?raw=true" width="500"/>

Note that under the Heston model, the distribution exhibits noticeably heavier skews as the asset price deviates further from the peak, reflecting the model’s ability to generate larger shocks in future extreme market conditions.
The estimated correlation parameter, $\rho = -0.29$, has negative correlation between asset returns and variance shocks, which is consistent with equity market data.
