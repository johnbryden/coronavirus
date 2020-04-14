# Coronavirus model

This is a framework for fitting models to coronavirus data.  The
framework calculates how good different mathematical models are at
predicting the death data we have from several countries. If a model
fits the data better than another model, then that can tell us
something about how the disease is spreading. The best fitting models
can be run forward in time to see how likely different outcomes are.


It uses differential evolution to optimse a model's fit to the death
data from several countries. You can play around with the settings at
the beginning of fitModel.py to specify different countries, etc.

To run it, use...

python fitModel.py

## Prerequisites

This uses python v3.

Install numpy,matplotlib,scypi,pandas using

pip install

pymc3 likely to come soon.

## Getting the latest data

We are currently using data from European Centre for Disease Prevention and Control. To get the latest version use the following command...

wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv -O cases.csv

## Notes

All dates in the model are calculated as days since the origin (1 Dec 2019).

It expects a computer with at least 2 cores, but you can adjust that
by changing the 'workers' setting.

The models I have added are some simple extensions to the Oxford
model. (See
https://www.medrxiv.org/content/10.1101/2020.03.24.20042291v1). The
two model either have a lockdown phase with a lower beta value, or a
proportion of the population which are excluded from getting infected
after the lockdown. An interesting note about the Oxford model is that
they set a proportion of infected+recovered individuals to die. In
this case, the death rate will never go down.

I have also added the ability to fit the same model to multiple
countries, adapting some of the parameters (like transmission rate) to
each country.

I was thinking about adding MCMC to this, so it will do
forecasting. However, I think finding a good model that fits the data
first is probably a better plan.