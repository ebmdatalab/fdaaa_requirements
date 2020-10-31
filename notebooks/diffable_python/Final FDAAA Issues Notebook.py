# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import csv
import json
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta 
from tqdm.auto import tqdm
import statsmodels.api as sm
import seaborn as sns
from zipfile import ZipFile
 
from time import time
from time import sleep
from io import StringIO
import os
import re
import matplotlib.pyplot as plt

import sys
from pathlib import Path
cwd = os.getcwd()
parent = str(Path(cwd).parents[0])
sys.path.append(parent)
# -

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# # Loading and Managing Data

# +
#If the main data file exists, it will read it, if not it will create it from the raw data.
#See files in lib folder and function docstrings for details of functions.
#For the full analysis, We've run this process twice, once for all trials and once just for 
#applicable trials

try:
    df = pd.read_csv(parent + '/data/applicable_trials_2020-01-17.csv')    
except FileNotFoundError:
    old_fda = parent + '/data/fdaaa_regulatory_snapshot.csv'
    
    #This data is the full ClinicalTrials.gov dataset for 17 Jan 2020.
    #Due to size, this is not in our GitHub repo, but stored separately on Figshare 
    #If you want to run this from scratch, unzip to the CSV and make sure the existing processed
    #data file is deleted or renamed.
    path = parent + '/data/raw_data/clinicaltrials_raw_clincialtrials_json_2020-01-17.csv'

    from lib.data_functions import fda_reg, get_data

    fda_reg_dict = fda_reg(old_fda)
    lines = get_data(path)

    #header names needed to create the dataset
    headers = ['nct_id', 'act_flag', 'included_pact_flag', 'results_due', 'has_results','pending_results', 'pending_data',
               'has_certificate', 'late_cert', 'certificate_date', "certificate_date_qc", "certificate_posted_date",
               'primary_completion_date', 'completion_date', 'available_completion_date', 'due_date', 'last_updated_date', 
               'last_verified_date', 'results_first_submitted_date', 'results_submitted_date_qc', 'results_first_posted_date', 
               'first_results_submission_any', 'study_first_submitted_date', 'study_submitted_date_qc', 
               'study_first_posted_date', 'documents', 'sponsor', 'sponsor_type', 'phase', 'location', 'study_status', 
               'study_type', 'primary_purpose', 'fda_reg_drug', 'fda_reg_device', 'is_fda_regulated', 'discrep_date_status', 
               'defaulted_date', 'collaborators','start_date', 'used_primary_completion_date', 'defaulted_pcd_flag', 
               'defaulted_cd_flag', 'intervention_types']

    from lib.final_df import make_row, make_dataframe

    df = make_dataframe(tqdm(lines), fda_reg_dict, headers, act_filter=True, scrape_date = date(2020,1,17))

    #Rename and uncomment this to save as a csv as appropriate
    #df.to_csv('applicable_trials_2020-01-17.csv', index=False)

# +
#creating the sponsor size variable for regressions

#This file is zipped for easier storage on GitHub
zip_file = ZipFile(parent + '/data/all_trials_2020-01-17.csv.zip')

#Load from zipped file
all_trials = pd.read_csv(zip_file.open('all_trials_2020-01-17.csv'))
del zip_file

#Getting counts of each sponsor across all of ClinicalTrials.gov
#Making a single column and dummies
group = all_trials[['nct_id', 'sponsor']].groupby('sponsor', as_index = False).count()
group.columns = ['sponsor', 'sponsored_trials']
df = df.merge(group, how='left', on='sponsor')
df['sponsor_quartile'] = pd.Categorical(pd.qcut(df.sponsored_trials, 4, labels=False), ordered=True)
s_q_df = pd.get_dummies(df.sponsor_quartile, prefix='s_q')
df = df.join(s_q_df)

#renaming columns
quart_rename = {'s_q_0': 'quartile_1', 's_q_1': 'quartile_2',  
                's_q_2': 'quartile_3', 's_q_3': 'quartile_4'}
df.rename(columns=quart_rename, inplace=True)

#Checking the ranges
quartile_ranges = pd.qcut(df.sponsored_trials, 4)
print(quartile_ranges.unique())

#creating a count of sponsors for applicable trials for rankings
app_group = df[['nct_id', 'sponsor']].groupby('sponsor', as_index = False).count()
app_group.columns = ['sponsor', 'covered_trials']
df = df.merge(app_group, how='left', on='sponsor')
#This is grouped by mean because each "group" of a single sponsor contains the same number of trials
#Could easily just be .max() or .min() as well
covered_trials = df[['sponsor', 'covered_trials']].groupby(by='sponsor', as_index=False).mean()
# -

#Check that the quartiles assigned correctly
df['sponsor_quartile'].unique()

# +
#Creating regression variables for use throughout
df['ind_spon'] = np.where(df.sponsor_type == 'Industry', 1, 0)
df['drug_trial'] = np.where(df.intervention_types.str.contains('Drug'), 1, 0)
phase_cats = ['Phase 1/Phase 2', 'Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4', 'N/A']
df.phase.fillna('N/A', inplace=True)
df['phase_collapsed'] = np.where(df.phase.isin(phase_cats[0:2]), 'Early Phase', 
                                np.where(df.phase.isin(phase_cats[2:4]), 'Late Phase', "N/A"))
df['phase_var'] = pd.Categorical(df.phase_collapsed, ordered=True, 
                                 categories = ['Early Phase', 'Late Phase', 'N/A'])
df['phase_var'] = df['phase_var'].cat.codes.astype('category')
phase_df = pd.get_dummies(df.phase_var, prefix = 'phase_cat')

df = df.join(phase_df)

phase_rename = {'phase_cat_0': 'early_phase', 'phase_cat_1': 'late_phase', 'phase_cat_2': 'N/A'}

df.rename(columns=phase_rename, inplace=True)

#Making sure date columns are dates
date_cols = ['certificate_date', "certificate_date_qc", "certificate_posted_date", 'primary_completion_date', 'completion_date', 
             'available_completion_date', 'due_date', 'last_updated_date', 'last_verified_date', 'results_first_submitted_date', 
             'results_submitted_date_qc', 'results_first_posted_date',  'first_results_submission_any', 
             'study_first_submitted_date', 'study_submitted_date_qc', 'study_first_posted_date', 'start_date']

for col in date_cols:
    df[col] = pd.to_datetime(df[col])
# -

#Importing functions created for analysis
from lib.analysis_functions import crosstab, simple_logistic_regression, create_ranking, get_count

# # Overall Cohort Description

# +
#Describing full data
total = len(all_trials)
all_applicable = len(df)
acts = df.act_flag.sum()
pacts = df.included_pact_flag.sum()
results_due = df.results_due.sum()
due_reported = len(df[(((df.has_results == 1) | (df.pending_results == 1)) & (df.results_due == 1))])
results_all = df.has_results.sum() + df.pending_results.sum()

print(
    f'''As of 17 January 2020, there are {total} trials and
{all_applicable} publicly identifiable applicable trials covered by the law.
{acts} ({round(acts/all_applicable * 100,1)}%) of these are identifiable as ACTs 
and {pacts} ({round(pacts/all_applicable * 100,1)}%) as pACTs.
{results_due} ({round(results_due/all_applicable * 100 ,1)}%) are due to report results.
{results_all} ({round(results_all/all_applicable * 100 ,1)}%) of the entire cohort 
and {due_reported} ({round(due_reported/results_due * 100 ,1)}%) of the due cohort have any results submitted.
    '''
)


# +
#use this function to get the counts of values for any variable in the dataset
#Can use this on any study population throughout the analysis

#Example
get_count(df, 'sponsor_quartile')
# -

# # Registration - Prospective and >21 Days Late

# +
#Getting the fields we need
pr_cats = ['nct_id', 'act_flag', 'included_pact_flag', 'start_date', 'study_first_submitted_date', 
           'study_submitted_date_qc', 'study_first_posted_date', 'available_completion_date', 'sponsor', 
           'ind_spon', 'drug_trial', 'phase_var', 'early_phase', 'late_phase', 'N/A', 'quartile_1', 'quartile_2', 
           'quartile_3', 'quartile_4']

pr_df = df[pr_cats].reset_index(drop=True)

# +
#this accounts for when the reg requirement came fully into effect as of Sept 27, 2008
pr_df['start_date_mod'] = np.where(pr_df.start_date < pd.Timestamp(2008,9,27), pd.Timestamp(2008,9,27) - pd.DateOffset(days=21),
                                   pr_df.start_date.dt.date)

#filter to check for trials registered within 21 days of the start date
legal_check = pr_df.study_first_submitted_date > (pr_df.start_date_mod + pd.DateOffset(days=21))
#1 means it was registered within the legal limit, 0 means in violation
pr_df['legal_reg'] = np.where(legal_check, 0, 1)

#filter to check for registration before the start date
pros_check = pr_df.study_first_submitted_date > pr_df.start_date
#1 means prospectively registered, 0 means retrospectively
pr_df['pros_reg'] = np.where(pros_check, 0, 1)

# +
print('{} trials out of {} ({}%) covered trials were registered late by the legal definition'.format(
    len(pr_df[pr_df.legal_reg == 0]), len(pr_df), round(len(pr_df[pr_df.legal_reg == 0])/len(pr_df) * 100,2)))

print('{} trials out of {} ({}%) covered trials were registered retrospectively'.format(
    len(pr_df[pr_df.pros_reg == 0]), len(pr_df), round(len(pr_df[pr_df.pros_reg == 0])/len(pr_df) * 100,2)))

# +
#calculating days late to register by the legal standard

legally_late = pr_df[pr_df.legal_reg == 0].reset_index(drop=True)
legally_late['days_late'] = (legally_late.study_first_submitted_date - (legally_late.start_date_mod + pd.DateOffset(days=21))) / pd.Timedelta('1 day')
# -

#Getting legal registration status by ACT/pACT status
act_late = crosstab(pr_df, 'legal_reg','act_flag')
act_late

# +
#Percent of ACTs and pACTs registered late
late_pacts = round((act_late.iloc[0, :][0] / (act_late.iloc[0, :][0] + act_late.iloc[0, :][1])) * 100,1)
print(f"{late_pacts}% of pACTs were registered late")

late_acts = round((act_late.iloc[1, :][0] / (act_late.iloc[1, :][0] + act_late.iloc[1, :][1])) * 100,1)
print(f"{late_acts}% of ACTs were registered late")
# -

#Descriptive statistics for days late
legally_late['days_late'].describe()

# +
#reg_bins = np.arange(0,int(legally_late['days_late'].max()) + 1, 100)
reg_bins = np.arange(0,1100 + 1, 100)
xlabels = ['0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(legally_late['days_late'],0,1000), hist=True, kde=False, bins=reg_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,1100))
plt.xticks(reg_bins)
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days Late', fontsize=25, labelpad=10)
plt.title("a. Days Registered Beyond the 21 Day Legal Limit", pad = 20, fontsize = 30)
#plt.savefig('figures/late_registration_1a.svg')

# +
#Outcome here is legal registration

x_reg = pr_df[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 
               'quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)
y_reg = pr_df['legal_reg'].reset_index(drop=True)

conf = simple_logistic_regression(y_reg,x_reg,cis=.001)
conf

# +
#Use this cell to check crude regression analysis of interest:

crude_x = pr_df[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_reg,crude_x,cis=.001)

# +
reg_rank = create_ranking(pr_df, 'legal_reg', marker=0)
#r_top_10_prct = reg_rank.legal_reg.quantile(.95)
reg_rank_merge = reg_rank.merge(covered_trials, on='sponsor')
reg_rank_merge['prct'] = round((reg_rank_merge['legal_reg'] / reg_rank_merge['covered_trials']) * 100,2)

#Check beyond top 10 to make sure no ties
reg_rank_merge[reg_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)

# +
comp_by_year = pr_df[['study_first_submitted_date', 'legal_reg']].groupby(pr_df.study_first_submitted_date.dt.year).agg(['sum', 'count'])

comp_by_year['prct_comp'] = round((comp_by_year['legal_reg']['sum'] / comp_by_year['legal_reg']['count']) * 100,2)

reg_trends = comp_by_year[(comp_by_year.index >= 2009) & (comp_by_year.index <= 2019)]['prct_comp']

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
plt.plot(reg_trends, marker='o')
plt.xticks(reg_trends.index)
plt.yticks(range(0,101,10))
plt.grid()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('% FDAAA Compliant', fontsize=12, labelpad=5)
plt.xlabel('Registration Year', fontsize=12, labelpad=7)
plt.title("Compliant Registrations by Year Registered", pad = 20, fontsize = 15)
#plt.savefig('figures/reg_trends.svg')
# -

comp_by_year

# # Last Verified Date

# +
cols = ['nct_id', 'has_results', 'pending_results', 'primary_completion_date', 'completion_date', 'study_first_posted_date', 
        'last_verified_date', 'last_updated_date', 'sponsor', 'act_flag', 'ind_spon', 'drug_trial', 'phase_var', 
        'early_phase', 'late_phase', 'N/A', 'quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']
update_dataset = df[cols].reset_index(drop=True)

update_dataset['scrape_date'] = date(2020,1,17)

# +
#Logic for exclusion
print('We start with all applicable trials: {}'.format(len(df)))

#We exclude trials that were first posted to ClinicalTrials.gov within the last year 
#as they don't have a full year of follow-up
exclude_under_a_year = update_dataset.study_first_posted_date >= pd.Timestamp(2019,1,17)
print("Exclude {} for starting within the last year (since 17 Jan 2019)".format(len(update_dataset[exclude_under_a_year])))
new_excluded = update_dataset[~exclude_under_a_year].reset_index(drop=True)
print("{} remaining".format(len(new_excluded)))

#We then exclude trials that have reached full completion as of the scrape date and have posted results.
#The law frees you from your responsibility to verify once a year when you have posted all results following
#completion of the trial
complete_results = (new_excluded.completion_date < new_excluded.scrape_date) & (new_excluded.has_results == 1)
print("Exclude {} for being completed with public results".format(len(new_excluded[complete_results])))
complete_excluded = new_excluded[~complete_results].reset_index(drop=True)
print("{} remaining".format(len(complete_excluded)))

#Lastly we exclude trials with pending results as these likely have a newer verification that will appear once the
#results complete QC review.
print("Eclude {} for being currently pending".format(len(complete_excluded[complete_excluded.pending_results==1])))
cohort = complete_excluded[complete_excluded.pending_results == 0].reset_index(drop=True)
print("{} remaining".format(len(cohort)))

# +
#Dummy for late verification
#Our data is from 17 Jan 2020 meaning verifications older than 17 January 2019 are officially out of date. 
#However, verifications are usually only provided in "Month Year" format with no date. As such, they are defaulted
#to the beginning of the month. Conservatively, we will treat 1 January 2019 as our cutoff.

# Late Verification = 1, Currently Verified = 0
cohort['late_veri'] = np.where(cohort.last_verified_date < pd.Timestamp(2019,1,1), 1,0)
late_veri = cohort.late_veri.sum()
prct_late = round(cohort.late_veri.sum()/len(cohort)*100,1)
print('{} of {} ({}%) of eligible trials are overdue to verify their records'.format(late_veri, len(cohort), prct_late))   

# +
#describing the days late for unverified trials

cohort['verification_due'] = cohort.last_verified_date + pd.DateOffset(years=1)
cohort['days_late'] = np.where(cohort.late_veri == 1, (pd.Timestamp(2020,1,1) - cohort.verification_due) / pd.Timedelta('1 day'), 0)
cohort[cohort['late_veri'] == 1].days_late.describe()

# +
late_with_update = len(cohort[(cohort.late_veri == 1) & (cohort.last_updated_date > pd.Timestamp(2019,1,17))])

print('{} trials with a late verification updated since 1 January 2019'.format(late_with_update))
print('This is {}% of the currently late trials'.format(round(late_with_update/late_veri * 100,2)))

# +
#reg_bins = np.arange(0,int(legally_late['days_late'].max()) + 1, 100)
ver_bins = np.arange(0,1100 + 1, 100)
xlabels = ['0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(cohort[cohort['late_veri'] == 1].days_late,0,1000), hist=True, kde=False, bins=ver_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,1100))
plt.xticks(ver_bins)
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days Late', fontsize=25, labelpad=10)
plt.title("b. Days Late to Verify Trial Data", pad = 20, fontsize = 30)
#plt.savefig('figures/last_verified_1b.svg')
# -

y_veri = cohort.late_veri.replace({0:1, 1:0})
x_veri = cohort[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 
                 'quartile_3', 'quartile_4']].reset_index(drop=True)

# +
#Outcome here is having a current verification date

simple_logistic_regression(y_veri,x_veri, cis=.001)

# +
#Use this cell to check crude regression analysis of interest:

crude_x = cohort[['act_flag']].reset_index(drop=True)

simple_logistic_regression(y_veri,crude_x,cis=.001)

# +
veri_rank = create_ranking(cohort, 'late_veri')
#v_top_10_prct = veri_rank.late_veri.quantile(.95)
veri_rank_merge = veri_rank.merge(covered_trials, on='sponsor')
veri_rank_merge['prct'] = round((veri_rank_merge['late_veri'] / veri_rank_merge['covered_trials']) * 100,2)

veri_rank_merge[veri_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)
# -

# # Certificate Analysis

# +
cert_analysis = df[['nct_id','due_date', 'has_results', 'has_certificate', "certificate_date_qc", "certificate_posted_date",
                    'certificate_date', 'late_cert', 'results_submitted_date_qc', 'sponsor', 'act_flag', 'ind_spon', 'drug_trial', 
                    'phase_var', 'early_phase', 'late_phase', 'N/A', 'quartile_1', 'quartile_2', 'quartile_3', 
                    'quartile_4']].reset_index(drop=True)

#all_trials = df[['nct_id', 'due_date', 'results_due', 'has_certificate']].reset_index(drop=True)
#all_trials['due_date'] = pd.to_datetime(all_trials.due_date)
# -

certificate = cert_analysis[cert_analysis.has_certificate == 1].reset_index(drop=True)
all_certificates = certificate.nct_id.count()
late_certificates = certificate.late_cert.sum()
certificate['on_time_cert'] = np.where(certificate.late_cert==1, 0, 1)
certs_with_results = certificate.has_results[certificate.has_results == 1].sum()
late_certs_with_results = certificate.has_results[(certificate.has_results == 1) & (certificate.late_cert == 1)].sum()

print('As of 17 Jan 2020, {} ({}%) applicable trials had recieved Certificates of Delay out of {} applicable trials'
      .format(all_certificates, round(all_certificates/len(df) * 100, 2), len(df)))
print('{} of those {} ({}%) have results'
      .format(certs_with_results, all_certificates, round(certs_with_results/all_certificates * 100,2)))
print('{} of certificates were submitted late. That is {}% of all certificates'
      .format(late_certificates, round((late_certificates/all_certificates)*100)))
print('Of those submitted late, only {} have since posted any results, {}% of all late certificates'
      .format(late_certs_with_results, round(late_certs_with_results/late_certificates * 100,2)))

#certificate['due_date'] = pd.to_datetime(certificate.due_date)
#certificate['certificate_date'] = pd.to_datetime(certificate.certificate_date)
#certificate['results_submitted_date_qc'] = pd.to_datetime(certificate.results_submitted_date_qc)
certificate['scrape_date'] = pd.Timestamp(2019,11,1)
certificate['days_late'] = certificate.certificate_date - certificate.due_date
days_late_count = certificate.days_late[certificate.late_cert == 1] / pd.Timedelta(days=1)

days_late_count.describe()

print(f"{len(days_late_count[days_late_count > 100])} of {len(days_late_count)} trials with a \
late certificate were more than 100 days late to apply. That is \
{round((len(days_late_count[days_late_count > 100])/len(days_late_count))*100,1)}%")

# +
lc_bins = np.arange(0,450 + 1, 50)
xlabels = ['0', '50', '100', '150', '200', '250', '300', '350', '400+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(days_late_count,0,400), hist=True, kde=False, bins=lc_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,450))
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days Late', fontsize=25, labelpad=10)
plt.title("c. Days Late to Apply for Certificate of Delay", pad = 20, fontsize = 30)
#plt.savefig('figures/late_certificate_1c.svg')
# -

x_cert = certificate[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 'quartile_3', 
                      'quartile_4']].reset_index(drop=True)
y_cert = certificate['on_time_cert'].reset_index(drop=True)

# +
#Outcome here is having an on-time certificate

simple_logistic_regression(y_cert,x_cert, cis=.001)

# +
#Use this cell to check crude regression analysis of interest:

crude_x = certificate[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_cert,crude_x,cis=.001)
# -

cert_rank = create_ranking(certificate, 'late_cert')
#c_top_10_prct = cert_rank.late_cert.quantile(.95)
cert_rank_merge = cert_rank.merge(covered_trials, on='sponsor')
cert_rank_merge['prct'] = round((cert_rank_merge['late_cert'] / cert_rank_merge['covered_trials']) * 100,2)
cert_rank_merge[cert_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).reset_index(drop=True).head(11)

# # QC Data
#
# From the FDAAA TrialsTracker, we had some detailed data available we scraped ourselves but not across all appliable trials, only trials that became due. Data on, at the very least, the time between first submission, final submission and posting date can be garnered for every applicable trial directly from the raw data.
#
# Given that this field is not preserved in the raw XML once results complete the QC process, this database was created through manual assessment of current and historic data both held by the DataLab and on the ClincialTrials.gov archive site. The Notebook `QC Expansion` can aid in recreating the pending data timeline for all trials that were pending on a given date from a processed data file such as the one created earlier.

# +
#Cutting down our full dataset with only what we need for the QC analysis

qc_cols = ['nct_id', 'results_due', 'due_date', 'available_completion_date', 'primary_completion_date', 'completion_date', 
           'certificate_date', 'last_updated_date', 'results_first_submitted_date', 'results_submitted_date_qc', 
           'results_first_posted_date', 'sponsor', 'act_flag', 'ind_spon', 'drug_trial', 'phase_var', 'early_phase', 
           'late_phase', 'N/A', 'quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']
xml_data = df[qc_cols].reset_index(drop=True)
    
#Calling in our QC data
data = pd.read_excel(parent + '/data/new_qc_dataset.xlsx', sheet_name='qc_data')
data['scrape_date'] = pd.Timestamp(2020,1,17)

# +
#This creates new columns that details the days between each step in the QC process. This details the time between 
#each submission, return and then subsequent resubmission

sub = 1
ret = 1
cols = list(range(4,34))
for col in cols:
    if col % 2 == 0:
        col_name = 'db_sub{}_ret{}'.format(sub, ret)
        sub += 1
    else:
        col_name = 'db_ret{}_sub{}'.format(ret, sub)
        ret += 1
    data[col_name] = (data.iloc[:,col+1] - data.iloc[:,col]) / pd.Timedelta('1 day')

# +
#Merging the fields taken from the XML with the detailed QC data.

qc_data = data.merge(xml_data, how='left', on='nct_id')
    
#Creating some fields we will need
qc_data['time_to_results'] = (qc_data.results_first_posted_date - qc_data.results_first_submitted_date) / pd.Timedelta('1 day')
qc_data['time_to_results_nc'] = (qc_data.results_first_posted_date - qc_data.first_sub_nc) / pd.Timedelta('1 day')
qc_data['ever_cancelled'] = np.where(qc_data.cancelled_details.notnull(),1,0)
qc_data['inferred_data'] = qc_data.inferred_data.replace(np.nan, '')

#Putting the columns in a sensible order to work with
col_order = ['nct_id', 'qc_data_status', 'cancelled_details', 'first_sub_any', 'first_sub_nc', 'db_sub1_ret1',
             'qc1_return', 'db_ret1_sub2', 'second_sub', 'db_sub2_ret2', 'qc2_return', 'db_ret2_sub3', 'third_sub',
             'db_sub3_ret3', 'qc3_return', 'db_ret3_sub4', 'fourth_sub', 'db_sub4_ret4', 'qc4_return', 
             'db_ret4_sub5', 'fifth_sub', 'db_sub5_ret5', 'qc5_return', 'db_ret5_sub6', 'sixth_sub', 'db_sub6_ret6', 
             'qc6_return', 'db_ret6_sub7', 'seventh_sub', 'db_sub7_ret7', 'qc7_return', 'db_ret7_sub8', 
             'eighth_sub', 'db_sub8_ret8', 'qc8_return', 'db_ret8_sub9', 'ninth_sub', 'db_sub9_ret9', 'qc9_return', 
             'db_ret9_sub10', 'tenth_sub', 'db_sub10_ret10', 'qc10_return', 'db_ret10_sub11', 'eleventh_sub', 
             'db_sub11_ret11', 'qc11_return', 'db_ret11_sub12', 'twelfth_sub', 'db_sub12_ret12', 'qc12_return', 
             'db_ret12_sub13', 'thirteenth_sub', 'db_sub13_ret13', 'qc13_return', 'db_ret13_sub14', 'fourteenth_sub', 
             'db_sub14_ret14', 'qc14_return', 'db_ret14_sub15', 'fifteenth_sub', 'db_sub15_ret15', 'qc15_return', 
             'db_ret15_sub16', 'sixteenth_sub', 'inferred_data', 'scrape_date', 'results_due', 'due_date', 
             'available_completion_date', 'primary_completion_date', 'completion_date', 'certificate_date', 
             'last_updated_date', 'results_first_submitted_date', 'results_submitted_date_qc', 
             'results_first_posted_date', 'time_to_results', 'time_to_results_nc', 'ever_cancelled', 'sponsor', 'act_flag', 
             'ind_spon', 'drug_trial', 'phase_var', 'early_phase', 'late_phase', 'N/A', 'quartile_1', 
             'quartile_2', 'quartile_3', 'quartile_4']

qc_data = qc_data[col_order]

# +
#get rid of trials that cancelled their first round submission and never resubmitted and get new look at counts
#NCT02684136
#NCT03134703
#NCT02176928

qc_data = qc_data[qc_data.first_sub_nc.notnull()].reset_index(drop=True)

qc_data[['nct_id','qc_data_status']].groupby(by='qc_data_status').count()

# +
qc_survivorship = qc_data[['nct_id', 'scrape_date', 'results_first_posted_date', 'first_sub_nc']].reset_index(drop=True)
qc_survivorship['posted_results'] = np.where(qc_survivorship.results_first_posted_date.notnull(), 1, 0)
qc_survivorship['days_to_posted'] = qc_survivorship.results_first_posted_date - qc_survivorship.first_sub_nc
qc_survivorship['censored'] = qc_survivorship.scrape_date - qc_survivorship.first_sub_nc
qc_survivorship.loc[qc_survivorship['posted_results'] == 1, 'censored'] = None
qc_survivorship['duration'] = np.where(qc_survivorship.censored.notnull(), 
                                       qc_survivorship.censored, qc_survivorship.days_to_posted)

#import lifelines
from lifelines import KaplanMeierFitter
from lib.lifelines_fix import add_at_risk_counts

fig = plt.figure(dpi=300)
ax = plt.subplot()


yticks = list(np.arange(0,1.1,.1))
kmf = KaplanMeierFitter()
kmf.fit((qc_survivorship['duration'] / pd.Timedelta('1 day')), qc_survivorship['posted_results'].astype(float))
ax = kmf.plot(ci_show=False, figsize=(15,10), grid=True, show_censors=True, censor_styles={'ms':10, 'marker':'|'},
              yticks = yticks, legend=False, ax=ax, lw=2.5)

ax.tick_params(labelsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.title("Time From Submission to Posting of Results on ClinicalTrials.gov for Applicable Trials", pad=20, fontsize=20)
plt.ylabel('Proportion Not Posted', labelpad=10, fontsize=15)
plt.xlabel('Days From Submission', labelpad=10, fontsize=15)
add_at_risk_counts(14, kmf, labels=None)
#plt.savefig('figures/qc_km_curve.svg')

print('Median time to report: {} days'.format(kmf.median_survival_time_))
# -

#This provides the values for the 95% CI at the median
from lifelines.utils import median_survival_times
median_survival_times(kmf.confidence_interval_)

# +
#Identify columns for returns and resubmissions for later filtering and analysis

ctgov_returns = ['db_sub1_ret1', 'db_sub2_ret2', 'db_sub3_ret3', 'db_sub4_ret4', 'db_sub5_ret5', 'db_sub6_ret6',
               'db_sub7_ret7', 'db_sub8_ret8', 'db_sub9_ret9', 'db_sub10_ret10', 'db_sub11_ret11', 'db_sub12_ret12',
               'db_sub13_ret13', 'db_sub14_ret14', 'db_sub15_ret15']

sponsor_subs = ['db_ret1_sub2', 'db_ret2_sub3', 'db_ret3_sub4', 'db_ret4_sub5', 'db_ret5_sub6', 'db_ret6_sub7', 
                 'db_ret7_sub8', 'db_ret8_sub9', 'db_ret9_sub10', 'db_ret10_sub11', 'db_ret11_sub12', 
                 'db_ret12_sub13', 'db_ret13_sub14', 'db_ret14_sub15', 'db_ret15_sub16']

qc_returns = ['qc1_return', 'qc2_return', 'qc3_return', 'qc4_return', 'qc5_return', 'qc6_return', 'qc7_return', 'qc8_return',
             'qc9_return', 'qc10_return', 'qc11_return', 'qc12_return', 'qc13_return', 'qc14_return', 'qc15_return']

qc_submissions = ['first_sub_nc', 'second_sub', 'third_sub', 'fourth_sub', 'fifth_sub', 'sixth_sub', 'seventh_sub', 'eighth_sub',
                 'ninth_sub', 'tenth_sub', 'eleventh_sub', 'twelfth_sub', 'thirteenth_sub', 'fourteenth_sub', 'fifteenth_sub',
                 'sixteenth_sub']


# +
#Function that does a quick check for re-submission delays longer than the 25 day legal deadline
def late_check(x):
    return(x > 25)

#Applying this function across all the sponsor re-submissions to flag when they contain a late resubmission at all
qc_data['ever_late_resub'] = np.where(qc_data[sponsor_subs].apply(late_check).any(axis=1),1,0)

#Counting how many re-submission were late for each trial
qc_data['total_late_subs'] = (qc_data[sponsor_subs] > 25).sum(axis=1)

#Counting the numbers of submissions and returns
qc_data['submissions'] = qc_data[qc_submissions].notnull().sum(axis=1)
qc_data['returns'] = qc_data[qc_returns].notnull().sum(axis=1)

#If a trial has results, this gets the time between the last submission and the posting of results which is essentially
#Another round of ClinicalTrials.gov review but without a return
qc_data['final_review'] = np.where(qc_data.results_first_posted_date.notnull(), 
                                   (qc_data.results_first_posted_date - qc_data.results_submitted_date_qc) / 
                                   pd.Timedelta('1 day'), None)

#The latest return date
qc_data['max_return'] = qc_data[qc_returns].max(axis=1)


#For trials that are currently pending, this gets the current days between the most recent QC return and 
#the date of this dataset
qc_data['cur_sub_day_outstanding'] = np.where((qc_data.returns == qc_data.submissions) & 
                                              (qc_data.qc_data_status == 'Pending'), 
                                             (qc_data.scrape_date - qc_data.max_return) / pd.Timedelta(days=1), np.nan)

#This looks for trials that have always been compliant, even if you consider currently outstanding re-submissions
qc_data['always_compliant'] = np.where((qc_data.cur_sub_day_outstanding < 25) & (qc_data.ever_late_resub == 0), 1, 0)

# +
#This summarizes which trials have detailed QC data available for further analysis, and which do not.

pending_all = len(qc_data[(qc_data.qc_data_status == "Pending") | 
                      (qc_data.qc_data_status == "Currently Canceled")])
available_all = len(qc_data[(qc_data.qc_data_status == "Results Available") | 
                            (qc_data.qc_data_status == 'No Detailed QC Available')])


print(f"{len(qc_data)} ({round((len(qc_data))/len(df) * 100,1)}%) of {len(df)} covered trials \
have results submitted or available")

print(f"{pending_all} are pending ({round((pending_all)/len(qc_data) * 100,1)}%)")

print(f"{available_all} are available ({round(available_all/len(qc_data) * 100,1)}%)")

never_pub = len(qc_data[qc_data.qc_data_status == 'No Detailed QC Available'])

print((f'''{never_pub} trials ({round(never_pub/len(qc_data) * 100,1)}%) do not have detailed QC data available'''))

trials_left = (len(qc_data) - len(qc_data[qc_data.qc_data_status == 'No Detailed QC Available']))

print(f"This leaves {trials_left} ({round(trials_left/len(qc_data) * 100,1)}%) with detailed quality control \
information available")

# +
#This is the dataset of only trials with detailed QC data we will use for the remaining analysis
qc_stats_detailed = qc_data[(qc_data.qc_data_status != 'No Detailed QC Available')].reset_index(drop=True)

#Describing trials with full Results Available
qc_d = len(qc_stats_detailed)

qc_full_results = len(qc_stats_detailed[qc_stats_detailed.results_first_posted_date.notnull()])

print(f"Among the {qc_d} trials with detailed QC data available, {qc_full_results} \
({round(qc_full_results/qc_d *100,1)}%) have full results available.")

#Stats about submissions for trials with full results
qc_stats_detailed[qc_stats_detailed.results_first_posted_date.notnull()].submissions.describe()

# +
#Stats about the submission of pending data

print(f"{pending_all} trials are currently plending")

#Stats about submissions for pending trials
qc_stats_detailed[qc_stats_detailed.results_first_posted_date.isnull()].submissions.describe()

# +
#Building some variable to describe the QC process

curr_pend_qc = qc_stats_detailed[(qc_stats_detailed.qc_data_status == "Pending") | 
                                 (qc_stats_detailed.qc_data_status == "Currently Canceled")]

print(f"{len(curr_pend_qc)} trials are currently pending")

cur_out_100 = len(curr_pend_qc[curr_pend_qc.cur_sub_day_outstanding > 100])

cur_late = len(curr_pend_qc[curr_pend_qc.cur_sub_day_outstanding > 25])

print(f"{cur_late} ({round(cur_late/len(curr_pend_qc) * 100,1)}%) currently outstanding trials are over 25 days outstanding")

print(f"{cur_out_100} ({round(cur_out_100/len(curr_pend_qc) * 100,1)}%) currently outstanding trials are over 100 days outstanding")

print("Stats on trials that are currently outstanding:")
curr_pend_qc.cur_sub_day_outstanding.describe()

# +
qc_bins = np.arange(0,450 + 1, 25)
xlabels = ['0', '25', '50', '75', '100', '125', '150', '175', '200', '225', '250', '275', '300', '325', '350', '375', '400+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(curr_pend_qc.cur_sub_day_outstanding,0,400), hist=True, kde=False, bins=qc_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,425))
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(qc_bins[:-2])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days Since QC Return', fontsize=25, labelpad=10)
plt.title("d. Days Currently Outsanding for Pending Trials (n=601)", pad = 20, fontsize = 30)
#plt.savefig('figures/qc_pending_delay_1d.svg')

# +
#Submission Stats

#all submissions includes the first submission (that cannot be late) so we need to subtract these 
#out when we only want to talk about re-submissions
resubmission_count = qc_stats_detailed.submissions.sum() - len(qc_stats_detailed)

print(f"There were {resubmission_count} resubmission after returns from ClincialTrials.gov staff")

late_resubs = qc_stats_detailed.total_late_subs.sum()

print(f"{late_resubs} ({round(late_resubs/resubmission_count * 100,1)}%) were more than 25 days late to resubmit")

over_100_spon = len(qc_stats_detailed[sponsor_subs].stack()[qc_stats_detailed[sponsor_subs].stack() > 100])

print(f"{over_100_spon } ({round(over_100_spon/resubmission_count * 100)}%) \
did not resubmit until after more than 100 days")

ever_late = qc_stats_detailed.ever_late_resub.sum()

print(f"The late resubmissions were spread out over {ever_late} ({round(ever_late/resubmission_count * 100,1)}%) \
of trials with complete QC information")

qc_stats_detailed[sponsor_subs].stack().describe()

# +
qc2_bins = np.arange(0,250 + 1, 25)
xlabels = ['0', '25', '50', '75', '100', '125', '150', '175', '200+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(qc_stats_detailed[sponsor_subs].stack(),0,200), hist=True, kde=False, bins=qc2_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,225))
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(qc2_bins[:-2])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days to Resubmission', fontsize=25, labelpad=10)
plt.title("e. Time to Resubmission for Returned QC Results", pad = 20, fontsize = 30)
#plt.savefig('figures/qc_rounds_1e.svg')

# +
#Lets get all trials with full results:

trials_with_results = qc_stats_detailed[(qc_stats_detailed.results_first_posted_date.notnull())].reset_index(drop=True)

results_one_submission = trials_with_results[(trials_with_results.submissions == 1) & (trials_with_results.returns == 0)]

results_multiple_submission = trials_with_results[(trials_with_results.submissions > 1)]

print(f"Results posted after first submission: {round(((len(results_one_submission)/len(trials_with_results)) * 100),2)}%")

#This is the descriptive statistics for trials that had results made available after a single round of review.
results_one_submission.time_to_results_nc.describe()
# -

#This tells us the statistics for all first round submissions whether they led to results or a return
results_multiple_submission.db_sub1_ret1.describe()

#This tells us the statistics for additional rounds of review after the first round
(results_multiple_submission.submissions - 1).describe()

# +
#This gives us the average number of days added per resubmission

num_resubs = trials_with_results.submissions - 1
resub_days = trials_with_results.time_to_results_nc - trials_with_results.db_sub1_ret1

resub_days.sum() / num_resubs.sum()
# -

qc_stats_detailed['never_late_resub'] = np.where((qc_stats_detailed.ever_late_resub == 0) & 
                                                 (qc_stats_detailed.cur_sub_day_outstanding < 25),1,0)

# +
#The outcome here is trials that never had a late resubmission during QC

x_qc = qc_stats_detailed[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 
                          'N/A', 'quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)
y_qc = qc_stats_detailed['never_late_resub'].reset_index(drop=True)

conf = simple_logistic_regression(y_qc,x_qc, cis=.001)
conf

# +
#Use this cell to check crude regression analysis of interest:

crude_x = qc_stats_detailed[['act_flag']].reset_index(drop=True)

simple_logistic_regression(y_qc,crude_x,cis=.001)
# -

qc_rank = create_ranking(qc_stats_detailed, 'never_late_resub', marker=0)
qc_rank_merge = qc_rank.merge(covered_trials, on='sponsor')
qc_rank_merge['prct'] = round((qc_rank_merge['never_late_resub'] / qc_rank_merge['covered_trials']) * 100,2)
qc_rank_merge[qc_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)

# <b>Zarin et al. 2019 Comparison:</b> 
#
# During peer review we were asked to compare our QC findings on first-submission success to Zarin et al. 2019. This piece used a slightly different methodology to examine this compared to ours. Specifically it didn't restrict the analysis to trials with results. This uses a comprable method and looks at the split between industry and non-industry.

# +
just_pending = qc_data[(qc_data.qc_data_status == "Pending") | 
                       (qc_data.qc_data_status == "Currently Canceled")].reset_index(drop=True)

pending_returned = just_pending[just_pending.returns != 0]
print(f"There are {len(pending_returned)} trials that are pending and we know were not successful in round 1")
print(f"{len(pending_returned[pending_returned.ind_spon==1])} of these from industry and \
{len(pending_returned[pending_returned.ind_spon==0])} from non-industry")

print(f"Overall first round compliance, including these trials, was {round((len(results_one_submission)/(len(trials_with_results) + len(pending_returned)) * 100),1)}%")

ind_denom = len(trials_with_results[trials_with_results.ind_spon==1]) + len(pending_returned[pending_returned.ind_spon==1])
ind_num = len(results_one_submission[results_one_submission.ind_spon==1])

non_ind_denom = len(trials_with_results[trials_with_results.ind_spon==0]) + len(pending_returned[pending_returned.ind_spon==0])
non_ind_num = len(results_one_submission[results_one_submission.ind_spon==0])

print(f"For industry sponsors {round((ind_num/ind_denom) * 100,2)}% succeeded on the first try")
print(f"For non-industry sponsors {round((non_ind_num/non_ind_denom) * 100,2)}% succeeded on the first try")
# -

# # Document Analysis

# +
doc_df = df[['nct_id', 'results_due', 'has_results', 'pending_results', 'has_certificate','results_first_submitted_date', 
             'results_first_posted_date','primary_completion_date', 'due_date', 'last_updated_date', 'documents', 'sponsor',
             'act_flag','ind_spon', 'drug_trial', 'phase_var', 'early_phase', 'late_phase', 'N/A', 'quartile_1', 
             'quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

doc_df['has_documents'] = np.where(doc_df.documents.notnull(), 1,0)
doc_df['results_first_submitted_date'] = pd.to_datetime(doc_df['results_first_submitted_date'])
doc_df['primary_completion_date'] = pd.to_datetime(doc_df['primary_completion_date'])
doc_df['due_date'] = pd.to_datetime(doc_df['due_date'])
doc_df['last_updated_date'] = pd.to_datetime(doc_df['last_updated_date'])
doc_df['results_first_posted_date'] = pd.to_datetime(doc_df['results_first_posted_date'])
doc_df.head()

# +
#Describing the population

due_and_docs = len(doc_df[(doc_df.results_due == 1) & (doc_df.has_documents == 1)])
due_docs_reported = len(doc_df[(doc_df.results_due == 1) & (doc_df.has_documents == 1) & (doc_df.has_results == 1)])
results_no_docs = len(doc_df[(doc_df.results_due == 1) & (doc_df.has_documents == 0) & (doc_df.has_results == 1)])
no_results_docs = len(doc_df[(doc_df.results_due == 1) & (doc_df.has_documents == 1) & (doc_df.has_results == 0) ])
check = len(doc_df[(doc_df.results_due == 1) & (doc_df.has_documents == 0) & (doc_df.has_results == 0) & (doc_df.pending_results==1)])
results_due = doc_df.results_due.sum()
print('{} Trials are due to report results, and therefore should have uploaded a protocol and SAP'.format(results_due))
print('Of these {} due trials have any documents, {} due trials have both documents and results'.format(due_and_docs,due_docs_reported))
print('{} due trials have documents but no results, {} have no documents but results'.format(no_results_docs, results_no_docs))
check
# -

has_docs_df = doc_df[['nct_id', 'documents']][doc_df.has_documents == 1].reset_index(drop=True)

has_docs_ids = has_docs_df.nct_id.to_list()

#this makes each document it's own row with nct_id as the index
dfs_list = []
import ast
has_docs = has_docs_df.copy()
has_docs['documents'] = has_docs['documents'].apply(ast.literal_eval)
for index, row in has_docs.iterrows():
    if isinstance(has_docs['documents'][index], list):
        l = len(has_docs['documents'][index])
        ix = [has_docs['nct_id'][index]] * l
        interim_df = pd.DataFrame(has_docs['documents'][index], index = ix)
        dfs_list.append(interim_df)
    else:
        interim_df = pd.DataFrame(has_docs['documents'][index], index = [has_docs['nct_id'][index]])
        dfs_list.append(interim_df)

# +
#Further processing
nct_index_df = pd.concat(dfs_list, sort=True)
nct_index_df = nct_index_df.reset_index(level=0)
nct_index_df.rename(columns= {nct_index_df.columns[0]: "nct_id"}, inplace=True)

#fixing a small incorrect data point that came up in summary review of data
#(verfified in document https://clinicaltrials.gov/ProvidedDocs/10/NCT01866410/Prot_SAP_000.pdf)
bad_index = nct_index_df.index[nct_index_df['document_date'] == 'January 24, 1014'].tolist()[0]
nct_index_df.at[bad_index,'document_date'] = 'January 24, 2014'

nct_index_df['document_date'] = pd.to_datetime(nct_index_df['document_date'])
nct_index_df.head()

# +
#The first time you run this notebook on a new dataset, you can get the data on when the documents 
#were last updated by importing and running the "history_scrape" function. However, if you are using 
#the shared data from the project or re-running a prior analysis you can just export and save a CSV that 
#you can then re-load.

#If you already have the output from the above exported to CSV, just run this cell pointing to that file
#if it isn't already in the same directory (this will work assuming no changed to the clones repo)

try:
    docs_updates = pd.read_csv(parent + '/data/history_scrape_2020-01-17.csv')
except FileNotFoundError:
    from lib.trial_history import history_scrape
    most_recent_doc_update = history_scrape(tqdm(has_docs_ids), date(2020,1,17))
    docs_updates = pd.DataFrame(most_recent_doc_update)
    docs_updates.to_csv('history_scrape_{}.csv'.format(date(2020,1,17)))
# -

#Cleaning and managing the scraped data as above
bad_index = docs_updates.index[docs_updates['document_date'] == 'January 24, 1014'].tolist()[0]
docs_updates.at[bad_index,'document_date'] = 'January 24, 2014'
docs_updates['upload_date'] = pd.to_datetime(docs_updates['upload_date'])
docs_updates['document_date'] = pd.to_datetime(docs_updates['document_date'])
docs_updates['version_date'] = pd.to_datetime(docs_updates['version_date'])
docs_updates.head()

#For ease of analysis we set dummy dates, for submission dates not scraped, 
#either very far in the past or future. This makes determining the earliest and latest
#submission dates much easier during grouping in the next step
full_docs_df = nct_index_df.merge(docs_updates, on=['nct_id', 'document_date', 'document_type'])
full_docs_df['dummy_date_past'] = pd.to_datetime(-2208988800, unit='s')
full_docs_df['dummy_date_future'] = pd.to_datetime(4102444800, unit='s')


# +
#Getting to 1 line per trial

def f(x):
    d = {}
    d['number_of_documents'] = x.nct_id.count()
    d['num_protocol_docs'] = np.where(x.document_has_protocol == 'Yes',1,0).sum()
    d['num_sap_docs'] = np.where(x.document_has_sap == 'Yes',1,0).sum()
    d['has_protocol'] = np.where(((np.where(x.document_has_protocol == 'Yes',1,0).sum())>0),1,0)
    d['has_sap'] = np.where(((np.where(x.document_has_sap == 'Yes',1,0).sum())>0),1,0)
    d['no_sap'] = np.where((np.where(x.no_sap.notnull(), 1, 0).sum() > 0), 1, 0)
    d['first_protocol_submitted'] = np.where(x.document_has_protocol == 'Yes', x.upload_date,x.dummy_date_future).min()
    d['latest_protocol_submitted'] = np.where(x.document_has_protocol == 'Yes', x.upload_date,x.dummy_date_past).max()
    d['first_sap_submitted'] = np.where(x.document_has_sap == 'Yes', x.upload_date,x.dummy_date_future).min()
    d['latest_sap_submitted'] = np.where(x.document_has_sap == 'Yes', x.upload_date,x.dummy_date_past).max()
    return pd.Series(d)

grouped = full_docs_df.groupby('nct_id').apply(f).reset_index()

#Now we can easily replace those far in the future/past dates with nulls
grouped.loc[grouped['latest_protocol_submitted'] == '1900-01-01', 'latest_protocol_submitted'] = pd.NaT
grouped.loc[grouped['latest_sap_submitted'] == '1900-01-01', 'latest_sap_submitted'] = pd.NaT
grouped.loc[grouped['first_protocol_submitted'] == '2100-01-01', 'first_protocol_submitted'] = pd.NaT
grouped.loc[grouped['first_sap_submitted'] == '2100-01-01', 'first_sap_submitted'] = pd.NaT

# +
#Bringing in additional data we need for further analysis

more_cols = ['nct_id', 'results_due', 'has_results', 'pending_results', 'primary_completion_date', 'due_date', 
             'results_first_submitted_date', 'results_first_posted_date', 'last_updated_date', 'ind_spon', 'drug_trial', 
             'phase_var', 'sponsor', 'act_flag', 'early_phase', 'late_phase', 'N/A', 'quartile_1', 'quartile_2', 'quartile_3', 
             'quartile_4']
merged = doc_df[more_cols].merge(grouped, how='left', on='nct_id')
# -

#Data cleaning
merged.number_of_documents.fillna(0, inplace=True)
merged.num_protocol_docs.fillna(0, inplace=True)
merged.num_sap_docs.fillna(0, inplace=True)
merged.has_protocol.fillna(0, inplace=True)
merged.has_sap.fillna(0, inplace=True)
merged.no_sap.fillna(0, inplace=True)
merged['prot_after_completion'] = np.where(merged.latest_protocol_submitted > merged.primary_completion_date, 1, 0)
merged['sap_after_completion'] = np.where(merged.latest_sap_submitted > merged.primary_completion_date, 1, 0)
merged.head()

# +
#filters

due_reported_filt = ((merged.results_due == 1) & (merged.has_results == 1))

due_unreported_filt = (merged.results_due == 1) & (merged.has_results == 0)

# +
#Building various counts to describe the population below

all_due = len(merged[(merged.results_due == 1)])

due_results = len(merged[due_reported_filt])

prot_and_sap = len(merged[due_reported_filt & (merged.has_protocol == 1) & (merged.has_sap == 1)])

just_prot = len(merged[due_reported_filt & (merged.has_protocol == 1) & (merged.has_sap == 0)])

due_results_new_prot = len(merged[(due_reported_filt & (merged.has_protocol == 1) & (merged.prot_after_completion == 1))])

due_results_new_sap = len(merged[(due_reported_filt & (merged.has_sap == 1) & (merged.sap_after_completion == 1))])

just_sap = len(merged[due_reported_filt & (merged.has_protocol == 0) & (merged.has_sap == 1)])

no_sap_statement = len(merged[due_reported_filt & ((merged.has_protocol == 1) & (merged.has_sap == 0) & (merged.no_sap == 1))])

due_results_no_docs = len(merged[due_reported_filt & (merged.has_protocol == 0) & (merged.has_sap == 0)])

docs_accounted = len(merged[due_reported_filt & ((merged.has_protocol == 1) | (merged.has_sap == 1))])

due_no_results = len(merged[due_unreported_filt])

due_no_results_docs = len(merged[due_unreported_filt & ((merged.has_protocol == 1) | (merged.has_sap == 1))])

due_no_resuls_prot_sap = len(merged[due_unreported_filt & ((merged.has_protocol == 1) & (merged.has_sap == 1))])

due_no_results_prot = len(merged[due_unreported_filt & ((merged.has_protocol == 1) & (merged.has_sap == 0))])

due_no_results_sap = len(merged[due_unreported_filt & ((merged.has_protocol == 0) & (merged.has_sap == 1))])

prot_after_comp = len(merged[due_reported_filt & (merged.has_protocol == 1) & (merged.prot_after_completion == 1)])

sap_after_comp = len(merged[due_reported_filt & (merged.has_sap == 1) & (merged.sap_after_completion == 1)])
# -

print(f'''
{due_results} trials are due and have public results available (meaning they have completed QC review); 
{prot_and_sap} ({round(prot_and_sap/due_results * 100,1)}%) of these have a protocol and sap available. 
An additional {no_sap_statement} registered trials have a protocol available but the sponsor declared the \
study has no SAP in their trial record meaning {docs_accounted} ({round(docs_accounted/due_results * 100, 1)}%) \
trials have accounted for all required documentation. 
Among the {due_no_results} due trials without results available, \
{due_no_results_docs} ({round(due_no_results_docs/due_no_results*100,1)}%) have some form of documentation \
available: {due_no_resuls_prot_sap} have a protocol and SAP, {due_no_results_prot} have just a protocol, \
and {due_no_results_sap} has just a SAP.
{prot_after_comp} ({round(prot_after_comp/docs_accounted * 100,1)}%) of the trials with a protocol and \
{sap_after_comp} ({round(sap_after_comp/len(merged[due_reported_filt & (merged.has_sap == 1)]) * 100,1)}%) of \
those with a SAP first submitted or updated these documents after trial completion.
''')

# +
just_due_results = merged[(merged.has_results == 1) & (merged.results_due == 1)].reset_index(drop=True)

just_due_results['docs_accounted'] = np.where((just_due_results.has_protocol == 1) | 
                                              (just_due_results.has_sap == 1),1,0)

x_docs = just_due_results[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 
                           'quartile_3', 'quartile_4']].reset_index(drop=True)
y_docs = just_due_results.docs_accounted.reset_index(drop=True)

conf = simple_logistic_regression(y_docs,x_docs, cis=.001)
conf

# +
#Use this cell to check crude regression analysis of interest:

crude_x = just_due_results[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_docs,crude_x,cis=.001)
# -

docs_rank = create_ranking(just_due_results, 'docs_accounted', marker = 0)
docs_rank_merge = docs_rank.merge(covered_trials, on='sponsor')
docs_rank_merge['prct'] = round((docs_rank_merge['docs_accounted'] / docs_rank_merge['covered_trials']) * 100,2)
docs_rank_merge[docs_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(12)


