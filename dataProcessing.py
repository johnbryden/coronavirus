import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.integrate import odeint
import coronaModel as cm

# this class basically strips out the relevant country data from the
# country code

class Country:
    def __init__(self,code,data,origin_date,prepend=7):
        self.country_code = code
        self.prepend = prepend

        self.country_data = data[data['countryterritoryCode'] == code].copy()
        self.pop_size=data[data.countryterritoryCode==code].popData2018.values[0]
        temp_days = pd.to_datetime(self.country_data['dateRep'], format='%d/%m/%Y')-pd.to_datetime(origin_date)
        self.country_data['daysSinceOrigin'] = temp_days.copy().dt.days
        self.country_data = self.country_data.sort_values('daysSinceOrigin')
        start_idx = self.country_data[self.country_data['cases'].gt(0)].index[0]
        self.start_day = self.country_data.loc[start_idx]['daysSinceOrigin']
        
        
    # This is the timespan of the data for fitting models
    # against. There's an extra period prepended at the beginning
    # before the first detected case
    def getTimespan(self):
        return self.start_day-self.prepend,len(self.country_data)


## Some functions for dealing with the array of parameters which is used by optimisers

def extractParameters(x0,i_country,single_beta1):
    if single_beta1:
        # Number of global parameters for all countries
        n_gl=4
        # Number per country
        n_per = 3
    else:
        n_gl=3
        n_per = 4
    
    params = np.zeros(n_gl+n_per)

    params[0:n_gl] = x0[0:n_gl]

    params [n_gl:n_gl+n_per] = x0[n_gl+i_country*n_per:n_gl+n_per*(1+i_country)]
    return params

    
def prettyOutputParamsSingleBeta1(countries,params):
    # new infections/recoveries
    sigma = params[0]

    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    psi = int(params[2])
    
    # growth before lockdown
    beta1 = params[3]

    # recovery rate
    R0 = sigma / beta1

    print ("R0 =",R0,"theta =",theta,"psi =",psi,"beta1=",beta1,"sigma=",sigma)
    counter = 0
    for country in countries:
        country_ps = extractParameters(params,counter,True)
        counter += 1
        R0_2 = country_ps[4]/sigma
        print (country.country_code,": beta2 =",country_ps[4]," delta_tau2 =",country_ps[5], " R0_2=",R0_2," delta_tau1 =",country_ps[6])

def prettyOutputParamsManyBetas(countries,params):
    # new infections/recoveries
    sigma = params[0]

    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    psi = int(params[2])
    
    print ("theta =",theta,"psi =",psi,"sigma=",sigma)
    counter = 0
    for country in countries:
        country_ps = extractParameters(params,counter,False)
        counter += 1
        R0_2 = country_ps[4]/sigma
        R0_1 = country_ps[3]/sigma
        print (country.country_code,":beta1 ",country_ps[3],": beta2 =",country_ps[4]," delta_tau2 =",country_ps[5],"R0_1 =",R0_1, " R0_2 =",R0_2," delta_tau1 =",country_ps[6])

def prettyOutputParams(countries,params,single_beta1=True):
    if single_beta1:
        prettyOutputParamsSingleBeta1(countries,params)
    else:
        prettyOutputParamsManyBetas(countries,params)

## Functions for fitting the model to the data

# This fits the recorded death values against the model
def fitDeathValuesToData (country,theta,psi,model_output):
    start_t,end_t = country.getTimespan()
    N = country.pop_size
    # add psi zeros to the front of the zvals array
    historic_zvals = np.concatenate((np.zeros(psi),model_output[:-psi,1]))
    recorded_deaths = country.country_data[country.country_data.daysSinceOrigin.between(start_t,end_t)].deaths.values
#    print (historic_zvals)
    log_likelihood = sum(binom.logpmf(recorded_deaths,N*historic_zvals,theta))
    return -log_likelihood

# This function is for plotting purposes
def generateDeathValues(country,params):
    start_t,end_t = country.getTimespan()
    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    psi = int(params[2])
    model_output = runModelForCountry(country,params)
    historic_zvals = np.concatenate((np.zeros(start_t),np.zeros(psi),model_output[:-psi,1]))
    death_vals = historic_zvals*country.pop_size*theta
    return death_vals

def runModelForCountry (country,params):
    # We start up to a week before the first obs
    start_t,end_t = country.getTimespan()

    tau = start_t + params[6]

    # new infections/recoveries
    sigma = params[0]
    
    # growth before lockdown
    beta1 = params[3]
    # growth after lockdown
    beta2 = params[4]
    # delay to lockdown
    tau2 =  start_t + params[5]

    N = country.pop_size

    y1 = 1.0/N
    z1 = 1.0/N

    tspan = np.arange (start_t,end_t)
    sol = odeint (cm.model,[1.0/N,1.0/N],tspan,args=(beta1,beta2,tau2,sigma))
    return sol

        

def fitModelToCountry (country,params):

#    print (params)
    # This probably isn't needed but I've left it just in case
    for p in params:
        if p<0 or np.isnan(p):
            return np.inf

    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    psi = int(params[2])
    if psi < 7:
        return np.inf

    sol = runModelForCountry(country,params)
#    print (sol)
    
    neg_log_likelihood = fitDeathValuesToData (country,theta,psi,sol)

    return neg_log_likelihood




