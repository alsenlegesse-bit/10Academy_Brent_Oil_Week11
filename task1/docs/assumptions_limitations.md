# Assumptions and Limitations

## Key Assumptions
1. **Data Quality**: Historical Brent oil price data is accurate and complete
2. **Market Efficiency**: Oil prices reflect all available information
3. **Event Timing**: Event dates are accurately recorded and significant
4. **Ceteris Paribus**: Other factors remain constant when analyzing specific events
5. **Model Specification**: Single change point per segment is sufficient

## Statistical Limitations
1. **Correlation ≠ Causation**: Detected change points coinciding with events suggest but don't prove causality
2. **Confounding Factors**: Multiple simultaneous events may obscure individual impacts
3. **Temporal Resolution**: Daily data may miss intra-day shocks
4. **Model Simplicity**: Single change point model may oversimplify complex market dynamics
5. **Parameter Stability**: Assumes constant variance within segments

## Data Limitations
1. **Time Range**: Data ends in September 2022, missing recent events
2. **External Factors**: Excludes weather, technological changes, renewable energy growth
3. **Currency Effects**: Prices in USD may be affected by exchange rate fluctuations
4. **Inflation**: Not adjusted for inflation in this analysis

## Methodological Limitations
1. **Bayesian Priors**: Results depend on prior distributions chosen
2. **MCMC Convergence**: Requires careful checking of sampler performance
3. **Stationarity**: Non-stationary time series may violate model assumptions
4. **Multiple Testing**: Multiple change points increase false discovery risk

## Communication of Uncertainty
All findings should be presented with:
- Probabilistic statements (e.g., "90% credible interval")
- Clear distinction between statistical association and causal impact
- Transparency about model assumptions
- Recommendations for further validation
