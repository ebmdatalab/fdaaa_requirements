import statsmodels.api as sm
import pandas as pd
import numpy as np

#Functions needed for analyses

#For quick crosstabs in pandas
def crosstab(df, outcome, exposure):
    """
    For quick crosstabs in pandas

    Keyword arguments:
    df -- The dataframe that contains the data
    outcome -- A string of the column name that contains the outcome variable
    exposure -- A string of the column name that contains the exposure variable
    """
    return pd.crosstab(df[exposure], df[outcome], margins=True)

#One-line logistic regression
def simple_logistic_regression(outcome_series, exposures_df, cis=.05):
    """
    Simple function for tidy logistic regression output.]

    Keyword arguments:
    outcome_series -- The outcome variable as a series
    exposure_df -- A DataFrame containing all your exposures
    cis -- Define what size you want your CIs to be. Default is .05 which provides 95% CIs

    """

    exposures_df['cons'] = 1.0
    mod = sm.Logit(outcome_series, exposures_df)
    res = mod.fit()
    print(res.summary())
    params = res.params
    conf = res.conf_int(cis)
    p = res.pvalues
    conf['OR'] = params
    ci_name = round((cis/2)*100,2)
    lower = str(ci_name) + '%'
    upper = str(100 - ci_name) + '%'
    conf.columns = [lower, upper, 'OR']
    conf = np.exp(conf)
    conf['p_value'] = p
    conf = conf[['OR', lower, upper, 'p_value']]
    conf = conf.round({'OR':2, 'p_value':5, lower:2, upper:2})
    return conf

#To quickly create rankings
#marker should be set to whatever represents the undesirable result in your binary variable
def create_ranking(df, field, marker=1):
    """
    Quick way to make sponsor rankings from the data.

    Keyword arguments:
    df -- The DataFrame containing your data
    field -- A string of the column name in the df that contains the data of interest
    marker -- The function will count whatever this is set to. Default is 1 assuming a binary variable
    """
    return (df[df[field] == marker][['sponsor', field]].groupby(by='sponsor', as_index=False).count().sort_values(
        field, ascending=False))



def get_count(df, field):
    """
    Quick way to get count of values in a specific column

    Keyword arguments:
    df -- The DataFrame containing your data
    field -- A string of the column name in the df that contains the data of interest
    """
    return df[['nct_id', field]].groupby(field).count()