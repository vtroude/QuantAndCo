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

## Goals

### Classifiers

The first tasks I tried to accomplish is to:

1- Gather Data
2- Build technical Indicators
3- Train ML model to predict market features
4- Use this prediction in Trading Strategy

Currently, there is a code which gather candlestick data and make compute indicators and statistics
```bash
python3 -m DataPipeline.make_data
```

Once you have saved the data and indicators in the directory _Data_, you can train random forest by doing
```bash
python3 -m Model.Classifier.randomforest_barrier
```

This function train two classier. The idea is the following:
1- We define at time _t_ two bars: _P_+_ and _P_-_; the goal of the first RF is to predict if the price _P_t_ will hit one of these bars during the next _n_ points i.e. for _s=t,t+1,..,t+n_
2- The second RF predicts in the context that we hit the bar, which one we hit first i.e. _P_+_ or _P_-_

Once you trained the RFs you can build a straightforward trading strategy such that for some threshold _epsilon_, if the probability of hitting a bar is _h_ and the sign of the bar _s_, then if _h>epsilon_ long the asset if _s>0_ or short it if _s<0_

```bash
python3 -m Strategy.backtest
```

After a strategy has been tested, the results are saved in _Score/backtest.csv_, and ready to be studied to decide if the strategy suit your objective.