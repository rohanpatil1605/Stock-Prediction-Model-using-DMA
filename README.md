# Stock-Prediction-Model-using-DMA
Stock Market Prediction Using Multi-Agent Federated Learning
This project leverages a Multi-Agent System (MAS) combined with Federated Learning and Nonlinear Models (Hammerstein-Wiener and LMS) to predict the next day's stock market opening price. By incorporating diverse financial indicators, including OHLCV data, RSI, SMA, and Sentiment Scores, this model ensures robust and accurate predictions, even in the volatile financial market.

Key Features:
Multi-Agent System (MAS) with federated learning for decentralized model training.

Nonlinear models, such as Hammerstein-Wiener and LMS, used to capture complex market behavior.

Utilization of Metropolis communication for node-to-node interaction across various financial data features.

Autocorrelation-based feature selection for improved prediction accuracy.

Achieved R² accuracy of 86% and an overall model accuracy of 96%.

Key Results:
The nonlinear model outperforms the linear model, with a Mean Absolute Error (MAE) of 2.95, predicting the next day's opening price more accurately.

The system adapts and stabilizes over time, with steady-state MSE reduction, ensuring model robustness and reliable performance.
