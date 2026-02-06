# Brent Oil Price Analysis Plan
## Data Science Workflow for Change Point Analysis

### 1. Data Acquisition & Loading
- Load historical Brent oil prices dataset (1987-2022)
- Parse date column to datetime format
- Handle missing values if any exist

### 2. Exploratory Data Analysis (EDA)
- Visualize price trends over time
- Calculate basic statistics (mean, median, volatility)
- Check for stationarity using Augmented Dickey-Fuller test
- Analyze volatility patterns and clustering

### 3. Event Data Collection
- Research major geopolitical and economic events (2012-2022)
- Compile event dataset with dates and descriptions
- Categorize events by type (Political, Economic, OPEC, Conflict)

### 4. Change Point Detection Modeling
- Implement Bayesian change point model using PyMC
- Define prior distributions for change points
- Configure MCMC sampling (NUTS sampler)
- Run multiple chains for convergence checking

### 5. Model Interpretation & Validation
- Check convergence using R-hat statistics
- Analyze posterior distributions of change points
- Compare detected change points with event timeline
- Quantify impact through before/after parameter analysis

### 6. Insight Generation & Reporting
- Associate statistical change points with real-world events
- Quantify price impact in percentage terms
- Create visualizations for stakeholders
- Document assumptions and limitations

### 7. Dashboard Development
- Design Flask API endpoints for data access
- Build React frontend with interactive visualizations
- Implement event highlight functionality
- Ensure responsive design for all devices

## Key Modeling Decisions
- Use Bayesian approach for uncertainty quantification
- Focus on mean shifts as primary structural breaks
- Consider log returns for stationarity in advanced models
- Use PyMC for probabilistic programming

## Success Metrics
- Identify at least 5-7 significant change points
- Associate change points with documented events
- Provide quantitative impact estimates
- Deliver functional dashboard for exploration
