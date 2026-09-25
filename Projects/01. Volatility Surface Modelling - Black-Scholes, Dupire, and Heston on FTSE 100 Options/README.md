# 01. Volatility Surface Modelling - Black-Scholes, Dupire, and Heston on FTSE 100 Options

This research project is on volatility surface modelling and option pricing, progressing from the Black-Scholes model through Dupire's Local Volatility model, and Heston's Stochastic Volatility model.
The project combines theoretical derivations with practical Python implementations using FTSE 100 option market data.

  ## Contents
  
  The project is split between theory and implementation. The PDF reports provide a mathematical lens of the models and concepts, written in LaTeX. Whilst the accompanying Markdown files contain Python implementations.
  Along the way, the project explores a number of interesting concepts, including stochastic calculus in Heston's Stochastic Volatility model, Girsanov's theorem, volatility surface construction, Monte-Carlo simulation, _etc._
   
    01. Introduction and General Concepts.pdf
    02. The Black-Scholes Model.pdf
    03. Dupire's Local Volatility Model.pdf
    04. Heston's Stochastic Volatility Model.pdf
    05. Conclusion and References.pdf
    Appendix A - Practical Modelling of FTSE 100 Options in Python.md
    Appendix B - Application of Girsanov’s Theorem to Heston’s Model.pdf
    Appendix C - Monte Carlo Simulations for Asset Price and Variance.md
    Appendix D - Calculating VaR and ES in Python.md

  ## About

In order to understand the purpose of this project, we need to understand volatility in the market. Volatility reflects the uncertainty in an asset's price, and mathematically is the standard deviation of an underlying asset's return.
Historical volatility looks backwards up to the current point in time, whilst implied volatility represents variations looking forward in time. It is important to note that implied volatility is mathematically derived from market option prices.

The aim of the project is to compare three different volatility models, which use different assumptions in calculating the implied volatility surface, a well-known tool for evaluating option pricing.
The Black-Scholes model uses constant volatility, the Dupire model uses local volatility, measuring volatility as a deterministic function, and the Heston model uses stochastic volatility.
These models were chosen as they provide closed-form analytical solutions to European-style options.

In order to support the analysis, real option market data was used, and consists of option prices on the FTSE 100 index for the second of January 2026.
Subsequent to evaluating the analytical solutions, it is necessary to use numerical methods to determine the volatilities for each model. The FTSE 100 data was operated on using Newton-Raphson, Monte-Carlo simulations, and other numerical techniques.
The result of this analysis are the volatility surfaces for each model.

 _<div align="right"> by Alexander Flann </div>_
 
