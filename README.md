# McGill - FIAM Quantitative Asset Management Hackathon

📅 September–October 2024  
🏫 McGill University – Master of Management in Analytics  
🏆 Finalist Project

## 🔍 Project Overview

This project was developed for the McGill–FIAM Quantitative Asset Management Hackathon. We built a machine learning-driven long-short equity strategy using XGBoost and financial NLP. The model was trained on historical fundamental and alternative data to forecast stock returns and construct an optimized portfolio that significantly outperformed the S&P 500.

## 🎯 Objectives

- Predict monthly stock returns using a machine learning model.
- Incorporate NLP-based sentiment signals from financial news.
- Construct and optimize a long-short portfolio using predicted returns.
- Benchmark against market indices over a 14-year period (2010–2024).

## 🧠 Methodology

- **Data Sources**: Historical financial statements, alternative data, and financial news.
- **Preprocessing**: Feature engineering on temporal data and NLP extraction from financial text.
- **Model**: XGBoost regression for return prediction.
- **Strategy**: Monthly rebalancing with top/bottom decile stock selection for long-short positioning.
- **Backtesting**: Evaluated on historical data with key risk and performance metrics.

## ⚙️ Technologies Used

- Python (Pandas, NumPy, Matplotlib, Scikit-learn)
- XGBoost
- NLP: spaCy, NLTK
- Portfolio Optimization: CVaR/Markowitz Framework
- Visualization: Matplotlib, Seaborn

## 📈 Results

- **Annualized Return**: > 20%  
- **Total Return (2010–2024)**: **650%+**  
- **Sharpe Ratio**: **2.47**  
- **Max Drawdown**: Controlled within strategy thresholds  
- Outperformed the S&P 500 and baseline models

## 📊 Visuals

<img src="images/returns_chart.png" width="600">

*You can include charts or screenshots of your portfolio return comparisons, feature importance from XGBoost, or heatmaps here.*

## 👤 Contributors

- **Yves E. Assali** – Model development, portfolio strategy, analytics  
- [Other team members if applicable]

## 📎 Project Highlights

- Blended quantitative modeling with practical investment strategy
- Applied NLP to extract alpha from unstructured financial data
- Designed end-to-end pipeline from data ingestion to performance evaluation

## 📬 Contact

Feel free to reach out via [LinkedIn](https://linkedin.com/in/yves-assali) or email at yvesassali07@gmail.com for more information.

---

