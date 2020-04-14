import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.integrate import odeint
#import coronaModel as cm

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

        temp_vals = []
        start_t,end_t = self.getFullModelTimespan()
        for i in range (start_t,end_t+1):
            val = self.country_data[self.country_data.daysSinceOrigin == i].deaths.values
            if val:
                temp_vals = temp_vals + [val[0],]
            else:
                temp_vals = temp_vals + [0.0,]
        self.mortalityValues = np.array(temp_vals)
        
    # This is the timespan of the data for fitting models
    # against. There's an extra period prepended at the beginning
    # before the first detected case. .
    def getFullModelTimespan(self):
        return self.start_day-self.prepend,max(self.country_data.daysSinceOrigin.values)

    def getRecordedMortalityValues(self):
        return self.mortalityValues
    

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

def prettyOutputParams(countries,params,args):
    if args.single_beta1:
        prettyOutputParamsSingleBeta1(countries,params)
    else:
        prettyOutputParamsManyBetas(countries,params)

## Functions for fitting the model to the data

# This fits the recorded death values against the model
def fitDeathValuesToData (country,theta,psi,model_output,args):
    mortal_idx = 1
    if args.use_infected:
        mortal_idx = 0
    start_t,end_t = country.getFullModelTimespan()
    N = country.pop_size
    # add psi zeros to the front of the zvals array
    historic_zvals = np.concatenate((np.zeros(psi),model_output[:-psi,mortal_idx]))
    recorded_deaths = country.getRecordedMortalityValues()
    #country.country_data[country.country_data.daysSinceOrigin.between(start_t,end_t)].deaths.values
#    print (len(historic_zvals),len(recorded_deaths))
#    print (historic_zvals)
    log_likelihood = sum(binom.logpmf(recorded_deaths,N*historic_zvals,theta))
    return -log_likelihood

# This function is for plotting purposes
def generateDeathValues(model,country,params,args):
    mortal_idx = 1
    if args.use_infected:
        mortal_idx = 0
    start_t,end_t = country.getFullModelTimespan()
    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    psi = int(params[2])
    model_output = runModelForCountry(model,country,params)
    historic_zvals = np.concatenate((np.zeros(start_t),np.zeros(psi),model_output[:-psi,mortal_idx]))
    death_vals = historic_zvals*country.pop_size*theta
    return death_vals

# This function is for plotting purposes
def generateSusceptibleValues(model,country,params):
    start_t,end_t = country.getFullModelTimespan()
    # Proportion that die
    theta=params[1]
    # Number of days from infection to death
    model_output = runModelForCountry(model,country,params)
    susceptible_vals = np.concatenate((np.zeros(start_t),model_output[:,1]))
    return 1.0-susceptible_vals

def runModelForCountry (model,country,params):
    # We start up to a week before the first obs
    start_t,end_t = country.getFullModelTimespan()

    tau = int(start_t + params[6])

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

    # we run the model from tau to end_t (inclusive)
    tspan = np.arange (tau,end_t+1)
    sol = odeint (model,[y1,z1],tspan,args=(beta1,beta2,tau2,sigma))
    # and then append some zeros at the start to model 0 infection levels
    #print (tau-start_t)
    result = np.concatenate ((np.zeros((tau-start_t,2)),sol))
    return result

        

def fitModelToCountry (model,country,params,args):

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

    sol = runModelForCountry(model,country,params)
#    print (sol)
    if (sol.min()<0):
        return np.inf
    neg_log_likelihood = fitDeathValuesToData (country,theta,psi,sol,args)

    return neg_log_likelihood




