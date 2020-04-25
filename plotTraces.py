from pylab import *
import coronaModel as cm
import dataProcessing as dp

def plotTrace (trace,countries,args):

    height = min(len(countries),2)
    width = int(.99+len(countries)/height)
    extra_time = 60
    counter = 0
    for country in countries:
        ax = subplot(width,height,counter+1)
        country.country_data.plot(kind='scatter',x='daysSinceOrigin',y='deaths',logy=True,ax=ax)

        all_params = args.params
        params = dp.extractParameters(all_params,counter,args.single_beta1)
        plot ([country.start_day,country.start_day],[.5,1e5])
        start_t,end_t = country.getFullModelTimespan()
        plot ([start_t+params[6],start_t+params[6]],[.5,1e5])
        plot ([start_t+params[5],start_t+params[5]],[.5,1e5])
        # This only works for one mcmc parameter
        param_trace = trace.get_values (str(args.bs[0]),burn=1000)
        param_chosen = choice(param_trace,100,replace=True)

        for i in range (0,100):
            all_params = args.params
            all_params[args.pfit[0]] = param_chosen[i]
            params = dp.extractParameters(all_params,counter,args.single_beta1)
            death_vals = dp.generateDeathValues(args.model,country,params,args,extra_time)
            plot (death_vals,'k',alpha=0.1)

        ax.set_ylim(.5,max(country.country_data.deaths)*1.5)
        ax.set_xlim(start_t-1,end_t+1+extra_time)
        title(country.country_code)
        counter += 1
        if counter == 1:
            legend (['First case','Model start','Lockdown start'])
    tight_layout()
    
