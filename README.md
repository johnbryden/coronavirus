# Coronavirus model

This is a framework for fitting a model to coronavirus data. It uses
differential evolution to optimse a model fit.

It expects a computer with at least 12 cores, but you can adjust that
by changing the 'workers' setting.

## Prerequisites

This uses python v3.

Install numpy,matplotlib,scypi,pandas using

pip install

pymc3 likely to come soon.

## Getting the latest data

We are currently using data from European Centre for Disease Prevention and Control. To get the latest version use the following command...

wget https://opendata.ecdc.europa.eu/covid19/casedistribution/csv -O cases.csv

## Notes

All dates in the model are calculated as days since the origin (31-12-2019).

Current issue is the differential evolution doesn't always find the
best solution. We'd need to be surer about that.