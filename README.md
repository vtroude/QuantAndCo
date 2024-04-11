# Quant & CO

The goal of this project is to build data pipelines and trading strategies.

## Structure

### DataPipeline

This repository contains all data related code such as:

1. Get historical data and **live data (TO DO!)** e.g. _binance.py_ 
2. Gather data and save them in files e.g. _make_data.py_
3. Get data from files and format them e.g. _get_data.py_
4. Build technical indicators e.g. _technicals_indicators.py_

## Model

Make ML models to predict market behaviors and/or build trading strategies.

### Classifier

Contains training methodology for binary classifier.

E.g. In _randomforest_barrier.py_, we train a RF to estimate the probability for the price to hit a take profit or stop loss barrier.

And contain the methodology to study the feature importance of the inputs e.g. _importance_study.py_

## Strategy

Contains file to build strategies over data, technical indicators and/or ML.