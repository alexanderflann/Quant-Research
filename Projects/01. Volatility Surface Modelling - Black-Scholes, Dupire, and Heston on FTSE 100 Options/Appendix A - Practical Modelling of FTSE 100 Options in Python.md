# A. Practical Modelling of FTSE 100 Options in Python

Python is used to implement, compare and analyse the three models, with in-line comments included to clarify each step of the code. The code in this research project was specifically written for the purpose of this project.

	**A.1 Import Modules** \
	_We must first import some dependencies._
   
	import os # Import module to use operating system dependent functionality
	import pandas as pd # Import module to data analyse tabular datasets
	import numpy as np # Import module to handle mathematical analysis on arrays
	import jax.numpy as jnp # Import numpy through the JAX environment, used for core numerical computations

	from scipy.integrate import quad # Compute a definite integral (from a to b)
	from scipy.optimize import minimize # Minimisation of a scalar function of one or more variables
	from scipy.interpolate import interp1d # Interpolate OIS rates to get r(T) for any maturity
	from jax.scipy.stats import norm # Import normal distribution
	from jax import grad # Creates a function that evaluates the gradient

	import matplotlib.pyplot as plt # Collection of functions that make matplotlib work like MATLAB, each function makes a change to a figure

	from nelson_siegel_svensson import NelsonSiegelSvenssonCurve # To instantiate and evaluate a Nelson Siegel Svensson Curve
	from nelson_siegel_svensson.calibrate import calibrate_nss_ols # Calibrate a Nelson Siegel Svensson curve using an ordinary least squares approach

	import seaborn as sns # Statistical visualisation based on matplotlib

	**A.2 Set Up Data** \
	_Since the dataset is stored in an Excel file, I first load it into Python using pandas and then extract all relevant fields for the modelling process. Afterwards, I extract some variables individually, as these will serve as the core inputs for the models in the remainder of the code._
   
	os.chdir('C:\\Alex\\Documents\\2025-2026\\University Lectures\\Research Project') # Set the current working directory

	options_data = pd.read_excel("FTSE_100_options_data.xlsx", sheet_name=None) # Read all Excel sheets for the options data

	valuation_date = pd.Timestamp("2026-01-02") # Define the snapshot date

	asset_price = options_data["asset_price"] # Extract historical data for FTSE 100

	split = options_data["bid_ask"]["Ticker"].str.split(" ", expand=True) # Split the Ticker string into several components

	bid_ask = (
    		options_data["bid_ask"] # Extract options surface snapshot for 02-01-2026
    		.assign(
        		Mid=lambda df: (df["Bid"] + df["Ask"]) / 2, # Calculate the mid price
        		Expiry_Date=lambda df: pd.to_datetime(split[1], format="%m/%d/%y"), # Extract expiry date since split_Ticker[1] = expiry date (MM/DD/YYYY)
        		Type=lambda df: split[2].str[0], # Extract option type (C or P)
        		Strike=lambda df: split[2].str[1:].astype(float), # Extract strike, everything after the first character, since split_Ticker[2] has format C9775 (for example)
        		T=lambda df: (pd.to_datetime(split[1], format="%m/%d/%y") - valuation_date).dt.days / 365 # Calculate time to maturity
    		)
	)

	risk_free_rate = (
    		options_data["risk_free_rate"] # Extract GBP OIS data
    		.drop(columns=options_data["risk_free_rate"].columns[16:33]) # Since options data has T<=1, remove longer-dated ranges for more precise model calibration
    		.query("Date == @valuation_date") # Keep valuation date data only
    		.melt(id_vars="Date", var_name="Tenor", value_name="Rate") # Change data frame to put column headers as Tenor with multiple rows
	)

	units = risk_free_rate["Tenor"].str.extract(r"(\d+)([DWMY])")
	numbers = units[0].astype(int)
	suffix = units[1]

	mult = {"D": 1/365, "W": 7/365, "M": 1/12, "Y": 1.0}

	risk_free_rate["T"] = numbers * suffix.map(mult) # Calculate time to expiry based on Tenor

	q = (
    		options_data["dividend_yield"] # Extract dividend data
    		.query("Date == 'Current'")["Dividend Yield"] # Keep only current dividend yield
    		.iloc[0] / 100 # Calculate q
	)

	S = asset_price.loc[
    		asset_price["Date"] == valuation_date, "Close" # Extract underlying asset price on valuation date
	].iloc[0]

	# Build a continuous risk-free curve r(T)
	r_curve = interp1d(
    		risk_free_rate["T"],
    		risk_free_rate["Rate"] / 100,
    		kind="linear",
    		fill_value="extrapolate"
	)

	bid_ask = bid_ask.assign(
    		r=lambda df: r_curve(df["T"]), # Extract the risk-free rate for each option ticker
	)

	bid_ask = bid_ask[bid_ask["Type"] == "C"] # Keep only European call options

	**A.3 Implementation of the Black-Scholes Model** \
	_The Black–Scholes model can be implemented directly by expressing its closed-form formulas in Python and specifying the model parameters as function arguments._

	def black_scholes(S, E, T, r, q, option_type, sigma): # Black-Scholes model for European Vanilla options
    		d1 = (jnp.log(S / E) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T)) # Standard formula for d1
    		d2 = d1 - sigma * jnp.sqrt(T) # Standard formula for d2
    		if option_type == "C":
        		call = S * jnp.exp(-q * T) * norm.cdf(d1, 0, 1) - E * jnp.exp(-r * T) * norm.cdf(d2, 0, 1) # Standard formula for call option
        		return call
    	else:
        	put = E * jnp.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * jnp.exp(-q * T) * norm.cdf(-d1, 0, 1) # Standard formula for put option
        	return put

	_We define the difference ($P_{theory} - P_{actual}$) through a loss function, and calculate the gradient._

	def loss_func(S, E, T, r, sigma_guess, price, q, option_type): # Define the loss function

    		theoretical_price = black_scholes(S, E, T, r, q, option_type, sigma_guess) # Price with the guess for volatility

    		market_price = price # Actual price

    		return theoretical_price - market_price # Loss is the difference between the theoretical price and the actual price, we want to minimise this loss

	loss_grad = grad(loss_func, argnums=4) # Vega, the sensitivity of the Black-Scholes price to volatility

	_Implied volatility is found by solving the pricing error equation via Newton–Raphson iterations, effectively minimising the difference between model and market prices._

	def solve_for_iv(S, E, T, r, price, q, option_type, sigma_guess = 0.8,
        		N_iter = 20, epsilon = 0.001, verbose = True): # Function to solve for the implied volatility iteratively using the Newton-Raphson method

        	converged = False

        	sigma = sigma_guess # Make a guess for the volatility
        	for i in range(N_iter):

            		loss_val = loss_func(S, E, T, r, sigma, price, q, option_type) # Calculate the loss function

            		if verbose:
                		print("\nIteration: ", i)
                		print("Current Error in Theoretical vs Market Price:")
                		print(loss_val)

            		if abs(loss_val) < epsilon: # Check if the loss is less than the tolerance
                		converged = True
                		break # If yes, then stop

            		else: # If no, then continue another iteration

                		loss_grad_val = loss_grad(S, E, T, r, sigma, price, q, option_type) # Calculate the gradient of the loss function

                		sigma = sigma - loss_val / loss_grad_val # Update the volatility using the Newton-Raphson formula

		return sigma

	_For each argument, define the input values which are used to calculate the implied volatility for each strike and maturity._

	# Intialise a list for implied volatilities, moneyness (S/E), and time to expiration
	ivs = []
	moneyness = []
	dtes = []

	# For each row in bid_ask, calculate the implied volatility based on the variables
	for idx, row in bid_ask.iterrows():
    		E = row["Strike"]
    		T = row["T"]
    		r = row["r"]
    		price = row["Mid"] # Take the mid price as market price for valuation date
    		option_type = row["Type"]
    		iv = solve_for_iv(S, E, T, r, price, q, option_type, verbose=False)

    		# Append values into respective lists
    		ivs.append(float(iv))
    		moneyness.append(S/E)
    		dtes.append(T)

	_Define a function that generates the 3D surface plot, which will be used consistently across all models._

	def plot_surface_3d(x, y, z, z_label, title, cmap='viridis'): # Function to create 3D surface plot

    		# Create a 3D figure with a larger size and better resolution
    		fig = plt.figure(figsize=(12, 8), dpi=100)
    		ax = fig.add_subplot(111, projection='3d')

    		# Create a surface plot using plot_trisurf
    		surf = ax.plot_trisurf(
        		x, y, z,
        		cmap=cmap,
        		linewidth=0.1,
        		antialiased=True,
        		alpha=0.8
    		)

    		# Customise the colour bar
    		cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
    		cbar.set_label(z_label, rotation=270, labelpad=15)

    		# Set the axis labels
    		ax.set_xlabel('Moneyness (S/E)', labelpad=10)
    		ax.set_ylabel('Time to Expiration', labelpad=10)
    		ax.set_zlabel(z_label, labelpad=10)

    		plt.title(title, pad=20, size=14) # Set the title

    		ax.view_init(elev=20, azim=45) # Adjust the viewing angle for better visualisation

    		plt.tight_layout() # Adjust layout to prevent label clipping
    
    		plt.show() # Show the plot

	_With the function defined, we can now call it to visualise the implied volatility surface._


	# Call function to display implied volatility surface
	plot_surface_3d(
    		moneyness,
    		dtes,
    		ivs,
    		z_label='Implied Volatility',
    		title='Implied Volatility Surface'
	)

	**A.4 Implementation of Dupire’s Local Volatility Model** \
	_The local volatility formula, (17), is defined, and the Black–Scholes model is then used to evaluate the corresponding local volatility._

	# Attach implied volatilies from Black-Scholes model and filter calls
	dupire_data = (
    		bid_ask.assign(IV=ivs)
           		.query("Type == 'C'")
	)

	# Derivative functions
	dC_dT = grad(black_scholes, argnums=2) # ∂C/∂T
	dC_dE = grad(black_scholes, argnums=1) # ∂C/∂E
	d2C_dE2 = grad(dC_dE, argnums=1) # ∂²C/∂E²

	# Initialise lists
	C_price = []
	theta_list = []
	dC_dE_list = []
	d2C_dE2_list = []
	d_moneyness = []
	d_dtes = []

	# Re-price and compute derivatives
	for idx, row in dupire_data.iterrows():
    		E = row["Strike"]
    		T = row["T"]
    		r = row["r"]
    		sigma = row["IV"]

    		C = black_scholes(S, E, T, r, q, "C", sigma)
    		theta_val = dC_dT(S, E, T, r, q, "C", sigma)
    		dC_dE_val = dC_dE(S, E, T, r, q, "C", sigma)
    		d2C_dE2_val = d2C_dE2(S, E, T, r, q, "C", sigma)

    		# Append values into respective lists
    		C_price.append(float(C))
    		theta_list.append(float(theta_val))
    		dC_dE_list.append(float(dC_dE_val))
    		d2C_dE2_list.append(float(d2C_dE2_val))
    		d_moneyness.append(S/E)
    		d_dtes.append(T)

	# Dupire local volatility
	lvs = []
	for C, th, d1, d2 in zip(C_price, theta_list, dC_dE_list, d2C_dE2_list):
    		numerator = th + (r - q) * E * d1 + q * C
    		denominator = 0.5 * E**2 * d2
    		lv = jnp.sqrt(numerator / denominator) # Formula for the local variance
    		lvs.append(float(lv))

	_Call the function to visualise the local volatility surface._

	# Call function to display local volatility surface
	plot_surface_3d(
    		d_moneyness,
    		d_dtes,
    		lvs,
    		z_label='Local Volatility',
    		title='Local Volatility Surface'
	)

	**A.5 Implementation of Heston’s Stochastic Volatility Model** \
	_First, we implement the Heston characteristic function by defining the quantities $d$, $g$, $a$, and $b$, which allow us to compute the required components of the function._

	def heston_charfunc(phi, S0, v0, kappa, theta, xi, rho, lambd, tau, r): # Function to work out the characteristic function for the Heston model

    		# Calculate constants
    		a = kappa*theta
    		b = kappa+lambd

    		# Rspi since it is a common term w.r.t. phi for easier computation
    		rspi = rho*xi*phi*1j

    		# Define d parameter given phi and b
    		d = np.sqrt( (rho*xi*phi*1j - b)**2 + (phi*1j+phi**2)*xi**2 )

    		# Define g parameter given phi, b and d
    		g = (b-rspi+d)/(b-rspi-d)

    		# Calculate characteristic function by components
    		exp1 = np.exp(r*phi*1j*tau)
    		term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/xi**2)
    		exp2 = np.exp(a*tau*(b-rspi+d)/xi**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/xi**2)

		return exp1*term2*exp2

	_We begin by defining the integrand as a function, allowing numerical integration methods, such as those provided by scipy, to be applied. The option price is then obtained by integrating this function, using either rectangular integration or the scipy integrate quad function._

	def integrand(phi, S0, v0, kappa, theta, xi, rho, lambd, tau, r): # Define integrand as a function
    		args = (S0, v0, kappa, theta, xi, rho, lambd, tau, r) # Label arguments
    		numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    		denominator = 1j*phi*K**(1j*phi)
    		return numerator/denominator

	def heston_price_rec(S0, K, v0, kappa, theta, xi, rho, lambd, tau, r): # Function to determine Heston price via rectangular integration
    		args = (S0, v0, kappa, theta, xi, rho, lambd, tau, r)

    		P, umax, N = 0, 100, 10000 # 100 is taken compared to infinity as it will compute it approximately
    		dphi=umax/N # dphi is width

    		for i in range(1,N):
        		# Rectangular integration
        		phi = dphi * (2*i + 1)/2 # Midpoint to calculate height
        		numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        		denominator = 1j*phi*K**(1j*phi)

        		P += dphi * numerator/denominator # Calculate f(midpoint) and multiply by width over this integration space

    		return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

	def heston_price(S0, K, v0, kappa, theta, xi, rho, lambd, tau, r): # Function to determine Heston price via scipy integrate quad function
    		args = (S0, v0, kappa, theta, xi, rho, lambd, tau, r)

    		real_integral, err = np.real( quad(integrand, 0, 100, args=args) )

    		return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

	_We estimate the risk-free rate curve using a parametric approach based on the Nelson Siegel Svensson model, fitted via ordinary least squares._
	
	yield_maturities = np.array(risk_free_rate["T"]).astype(float) # Return maturities as a numpy array
	yields = np.array(risk_free_rate["Rate"]).astype(float)/100 # Return risk free rate as a numpy array

	curve_fit, status = calibrate_nss_ols(yield_maturities,yields) # cCalculate the NSS OLS curve

	heston_data = bid_ask[bid_ask["Type"] == "C"][["T", "Strike", "Mid"]] # Create Heston dataset using the bid-ask data
	heston_data["Rate"] = heston_data["T"].apply(curve_fit) # Apply NSS curve to calculate the risk free rate and add it to the Heston data

	_The Heston model features five parameters that are unknown and can be inferred from market data. This is achieved by minimising the squared difference between model and market prices, thereby optimising the calibration objective function._

	# Define variables to be used in the optimisation
	S0 = S
	r = heston_data['Rate'].to_numpy(float)
	K = heston_data['Strike'].to_numpy('float')
	tau = heston_data['T'].to_numpy('float')
	P = heston_data['Mid'].to_numpy('float')

	# Parameters are v0, kappa, theta, xi, rho, lambd
	params = {"v0": {"x0": 0.1, "lbub": [1e-3,0.1]},
          	  "kappa": {"x0": 3, "lbub": [1e-3,5]},
          	  "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
          	  "xi": {"x0": 0.3, "lbub": [1e-2,1]},
          	  "rho": {"x0": -0.8, "lbub": [-1,0]}, # Correlation coefficient
          	  "lambd": {"x0": 0.03, "lbub": [-1,1]},
          	  } # Initial guess with upper and lower bound

	x0 = [param["x0"] for key, param in params.items()]
	bnds = [param["lbub"] for key, param in params.items()]

	def SqErr(x): # Calibration where we want to optimise the objective function
    		v0, kappa, theta, xi, rho, lambd = [param for param in x]

    		# Use rectangular integration function to find sum of the error
    		err = np.sum( (P-heston_price_rec(S0, K, v0, kappa, theta, xi, rho, lambd, tau, r))**2 /len(P) )
	
    		# Zero penalty term
    		pen = 0 # np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] ) - otherwise can add a penalty fucntion to be the distance to initial parameter vector

    		return err + pen

	result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=bnds) # Minmise the SqErr function

	v0, kappa, theta, xi, rho, lambd = [param for param in result.x] # Unpacking parameters with list comprehension

	heston_prices = heston_price_rec(S0, K, v0, kappa, theta, xi, rho, lambd, tau, r) # Calculate estimated option prices

	heston_data['Heston'] = heston_prices # Add Heston prices to Heston data

	_In essence, once the Heston model prices have been computed, the implied volatility surface is obtained by finding the volatility, $\sigma_I$, that, when inserted into the Black–Scholes formula, minimises the difference between the Heston price and the corresponding Black–Scholes price. Thus:
\setcounter{equation}{40}
\begin{equation}
    \sigma_I (E, T) = \text{Call}_{BS}^{-1} (S_0, r, E, T, \text{Call}_{Heston}(S_0, r, E, T, v_0,  \kappa^{\mathbb{Q}}, \theta^{\mathbb{Q}}, \xi, \rho))
\end{equation}. We then call the function to visualise the Heston Implied Volatility surface._
	
	# Initialise lists
	heston_ivs = []
	heston_moneyness = []
	heston_dtes = []

	# Solve for Heston implied volatility by minmising the difference of Heston and Black-Scholes prices by iterating for xi through the Newton-Raphson method
	for idx, row in heston_data.iterrows():
    		E = row["Strike"]
    		T = row["T"]
    		r = row["Rate"]
    		price = row["Heston"]
    		iv = solve_for_iv(S, E, T, r, price, q, option_type ="C", verbose=False)

    		# Append values into respective lists
    		heston_ivs.append(float(iv))
    		heston_moneyness.append(S/E)
    		heston_dtes.append(T)

	# Call function to display Heston implied volatility surface
	plot_surface_3d(
    		heston_moneyness,
    		heston_dtes,
    		heston_ivs,
    		z_label='Heston Implied Volatility',
    		title='Heston Implied Volatility Surface'
	)
