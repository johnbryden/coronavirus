from pylab import *
import pandas as pd
from scipy.stats import binom
from scipy.optimize import minimize,differential_evolution
import coronaModel as cm

import coronaModel as cm
import dataProcessing as dp
from types import SimpleNamespace

# In development you need to reload the libraries
import importlib
importlib.reload(cm)
importlib.reload(dp)


    
def fitModel(x0,args):
    global min_val
    counter = 0
    neg_log_likelihood = 0
    for country in countries:
        params = dp.extractParameters(x0,counter,args.single_beta1)
        neg_log_likelihood += dp.fitModelToCountry(args.model,country,params,args)
        counter += 1
    if neg_log_likelihood<min_val:
        min_val = neg_log_likelihood
        print (min_val,x0)
    return neg_log_likelihood


data = pd.read_csv('cases.csv', encoding = "ISO-8859-1")

#### Settings

# All time in this model is in days since the start date which is
# currently set as the end of 2019

origin_date = '2019-12-1'

# The earliest possible start day for an infection of a country is
# recorded as the earliest recorded case, minus the prepend value.
prepend = 21

# Is there a single beta1 for all countries, or a beta1 for each
single_beta1 = False

# Number of parallel cores to work on optimisation
workers = 12

#Which model
model = cm.model3

# Use infected or infected+recovered as the compartment to select
# mortalities from
use_infected = True

#Which countries
#test_countries = ["GBR","DEU","USA","ESP","ITA","CAN","FRA","BEL"]
#test_countries = ["GBR","ITA","CHN"]
#test_countries = ["ISL",]
test_countries = ["GBR"]
countries = []

for code_i in test_countries:
    country = dp.Country(code_i,data,origin_date,prepend)
    countries += [country,]

min_val = 1e9

# Old code which used the minimize function. I've left this here just
# in case I want to try it again.
#x0 = array([2.5,.01,14,.3,0.3,10])
#res = minimize (fitModel,x0, method='L-BFGS-B')
#res = minimize (fitModel,x0, method='nelder-mead')


# bounds for parameters
# max rates are to double every day
beta_bds = (1e-9,1)
# or to halve every day
sigma_bds = (1e-9,1)
# Time to mortality (psi)
psi_bds = (7,20)
# Fraction of mortality (theta)
theta_bds = (1e-9,0.5)

# The boundaries for the infection start from the beginning of the
# prepend period to a week after the first detected case
inf_start_bds = (0,prepend+7)

# The earliest start of the lockdown is 3 weeks (21+prepend days)
# after the first detected case
lock_bds = (21+prepend,60+prepend)


# This is an arguments object to be passed around all the functions.
# It makes it easy to add new arguments that will be propagated through the algorithm.
args = SimpleNamespace()
args.model = model
args.single_beta1 = single_beta1
args.use_infected = use_infected

if single_beta1:
    bounds = [sigma_bds,theta_bds,psi_bds,beta_bds]
    for country in countries:
        bounds += [beta_bds,lock_bds,inf_start_bds]
else:
    bounds =  [sigma_bds,theta_bds,psi_bds]
    for country in countries:
        bounds += [beta_bds,beta_bds,lock_bds,inf_start_bds]


res = differential_evolution(fitModel,bounds, updating='deferred',workers=workers,args=(args,))
dp.prettyOutputParams (countries,res.x,args)

height = min(3,len(countries))
width = int(.99+len(countries)/height)

print ("params=[",','.join([str(x) for x in res.x]),"]")


figure(1)
clf()
counter = 0
for country in countries:
    ax = subplot(width,height,counter+1)
    country.country_data.plot(kind='scatter',x='daysSinceOrigin',y='deaths',logy=True,ax=ax)
   
    
    params = dp.extractParameters(res.x,counter,args.single_beta1)
    counter += 1
    death_vals = dp.generateDeathValues(model,country,params,args)
    plot (death_vals,'k')
    plot ([country.start_day,country.start_day],[.5,1e5])
    start_t,end_t = country.getFullModelTimespan()
    plot ([start_t+params[6],start_t+params[6]],[.5,1e5])
    plot ([start_t+params[5],start_t+params[5]],[.5,1e5])
    ax.set_ylim(.5,max(country.country_data.deaths)*1.5)
    ax.set_xlim(start_t-1,end_t+1)
    title(country.country_code)
    if counter == 1:
        legend (['model','First case','Model start','Lockdown start','Death data',])
tight_layout()
figure(2)
clf()
counter = 0
for country in countries:
    ax = subplot(width,height,counter+1)
    counter += 1
    sus_vals = dp.generateSusceptibleValues(model,country,params)
    plot (sus_vals,'k')
    ax.set_xlim(start_t-1,end_t+1)
    title(country.country_code)
    xlabel('daysSinceOrigin')
    ylabel('Susceptible density')
tight_layout()

figure(3)
clf()
counter = 0
for country in countries:
    ax = subplot(width,height,counter+1)
    counter += 1
    sol = dp.runModelForCountry(model,country,params,extra_time=90)
    plot (sol)
    ax.set_xlim(start_t-1,end_t+1+90)
    title(country.country_code)
    xlabel('daysSinceOrigin')
    ylabel('solution')
    if counter == 1:
        if shape(sol)[1] == 2:
            legend (['y','z'])
        else:
            legend (['y','z','x'])
tight_layout()
show()

