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
import pandas as pd
import numpy as np
from datetime import date
from tqdm.auto import tqdm
import seaborn as sns
from zipfile import ZipFile

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

# If the processed data files exists (for covered and all trials), it will read it, if not it will create it from the raw data. See files in the `lib` folder and function docstrings for details of functions. The GitHub will always contains the processed data files but interested users should also be able to re-create them from the raw data as needed. The process used to archive the raw data in our storage format is detailed here:
# https://github.com/ebmdatalab/clinicaltrials-act-converter
#
# The processed data is included in this repository in the `data` folder. The raw data is too large to easily store on GitHub so we load it in from Dropbox storage when needed. If the Dropbox link ever fails, you can also download a copy of the raw data from here:
# https://doi.org/10.6084/m9.figshare.12789902
#
# You can then unzip and import/run `get_data_local` instead of `get_data` on the CSV locally to get the processed dataset.

# +
try:
    df = pd.read_csv(parent + '/data/applicable_trials_2021-01-18.csv')
    
    #This file is zipped for easier storage on GitHub.
    zip_file = ZipFile(parent + '/data/all_trials_2021-01-18.csv.zip')
    df2 = pd.read_csv(zip_file.open('all_trials_2021-01-18.csv'))
    del zip_file
    
except FileNotFoundError:
    old_fda = parent + '/data/fdaaa_regulatory_snapshot.csv'
    
    #This data is the full ClinicalTrials.gov dataset for 18 Jan 2021.
    #Due to size, this is not in our GitHub repo, but stored on Dropbox 
    #You should also be able to download the raw data using this URL
    path = 'https://www.dropbox.com/s/awlhqwjtkzp6t4b/clinicaltrials_raw_clincialtrials_json_2021-01-18.csv.zip?dl=1'

    from lib.data_functions import fda_reg, get_data

    fda_reg_dict = fda_reg(old_fda)
    lines = get_data(path, '2021-01-18')

    #headers is just the list of header names to save space here
    from lib.final_df import make_row, make_dataframe, headers

    #Just pACTs/ACTs
    df = make_dataframe(tqdm(lines), fda_reg_dict, headers, act_filter=True, scrape_date = date(2021,1,18))
    
    #Everything on CT.gov
    df2 = make_dataframe(tqdm(lines), fda_reg_dict, headers, act_filter=False, scrape_date = date(2021,1,18))
    
    #We won't need this anymore so deleting to save some memory
    del lines
    
    #Uncomment this to save as a csv as appropriate
    #df.to_csv(parent + '/data/applicable_trials_2021-01-18.csv', index=False)
    #df2.to_csv(parent + '/data/all_trials_2021-01-18.csv', index=False)

# +
#creating the sponsor size variable for regressions

#Getting counts of each sponsor across all of ClinicalTrials.gov
#Making a single column and dummies
group = df2[['nct_id', 'sponsor']].groupby('sponsor', as_index = False).count()
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

analysis_cols = ['act_flag', 'ind_spon', 'drug_trial', 'early_phase', 'late_phase', 'N/A', 
                 'quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']

#Importing functions created for analysis
from lib.analysis_functions import crosstab, simple_logistic_regression, create_ranking, get_count, get_prcts

# # Overall Cohort Description

# +
#Describing full data
total = len(df2)
all_applicable = len(df)
acts = df.act_flag.sum()
pacts = df.included_pact_flag.sum()
results_due = df.results_due.sum()
due_reported = len(df[(((df.has_results == 1) | (df.pending_results == 1)) & (df.results_due == 1))])
results_all = df.has_results.sum() + df.pending_results.sum()
df['reported_late'] = np.where(((df.results_due == 1) & (df.due_date < df.first_results_submission_any) & 
                                df.first_results_submission_any.notnull()), 1, 0)
df['compliant_reported'] = np.where((df.results_due == 1) & (df.reported_late == 0) & ((df.has_results == 1) | (df.pending_results == 1)), 1, 0)
late_results = df.reported_late.sum()


print(
    f'''As of 18 January 2021, there are {total} trials and
{all_applicable} publicly identifiable applicable trials covered by the law.
{acts} ({round(acts/all_applicable * 100,1)}%) of these are identifiable as ACTs 
and {pacts} ({round(pacts/all_applicable * 100,1)}%) as pACTs.
{results_due} ({round(results_due/all_applicable * 100 ,1)}%) are due to report results.
{results_all} ({round(results_all/all_applicable * 100 ,1)}%) of the entire cohort 
and {due_reported} ({round(due_reported/results_due * 100 ,1)}%) of the due cohort have any results submitted.
{due_reported - late_results} ({round(((due_reported - late_results)/results_due) * 100, 1)}%) of the due trials submitted their results on time.    
    '''
)
# -


#Run this function to get reporting compliance crosstabs for any variable of interest 
crosstab(df[df.results_due == 1], 'compliant_reported', 'act_flag')

# +
#Values for Table 1

summary = {}

for a in analysis_cols:
    cross = crosstab(df[df.results_due == 1], 'compliant_reported', a)
    summary[a] = get_prcts(cross)
    
pd.DataFrame(summary).T

# +
#use this function to get the counts of values for any variable in the dataset
#Can use this on any study population throughout the analysis

#Example
get_count(df, 'sponsor_quartile')
# -

# ## **Note:** 
#
# **The below analyses contain some additional descriptive data for each area (e.g., additional figures, sponsor-level compliance information) not included in the published work due to space limits of the format. This remains in the notebook for any interested parties and to support N. DeVito's doctoral thesis which may draw on this data.**

# # Registration - Prospective and >21 Days Late

# The FDAAA 2007 requires that all covered trials are registered within 21 days of their start date, that is the date in which the first participant is enrolled in the study.

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
print('{} trials out of {} ({}%) covered trials were registered on time the legal definition'.format(
    len(pr_df[pr_df.legal_reg == 1]), len(pr_df), round(len(pr_df[pr_df.legal_reg == 1])/len(pr_df) * 100,2)))

print('{} trials out of {} ({}%) covered trials were registered prospectively'.format(
    len(pr_df[pr_df.pros_reg == 1]), len(pr_df), round(len(pr_df[pr_df.pros_reg == 1])/len(pr_df) * 100,2)))

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
plt.show()
#plt.savefig('figures/late_registration_1a.svg')
# -

#Can use this function to get crosstabs for covariate of interest
crosstab(pr_df, 'legal_reg','act_flag')

# +
#Values for Table 1

summary = {}

for a in analysis_cols:
    cross = crosstab(pr_df, 'legal_reg', a)
    summary[a] = get_prcts(cross)
    
pd.DataFrame(summary).T
# -

x_reg = pr_df[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 
               'quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)
y_reg = pr_df['legal_reg'].reset_index(drop=True)

# +
#Use this cell to check crude regression analysis of interest:

crude_x = pr_df[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_reg,crude_x,cis=.001)

# +
#Adjusted model for legal registration

conf = simple_logistic_regression(y_reg,x_reg,cis=.001)
conf

# +
#Here we are measuring late registrations so set the "legal_reg" markerer to 0

reg_rank = create_ranking(pr_df, 'legal_reg', marker=0)
#r_top_10_prct = reg_rank.legal_reg.quantile(.95)
reg_rank_merge = reg_rank.merge(covered_trials, on='sponsor')
reg_rank_merge['prct'] = round((reg_rank_merge['legal_reg'] / reg_rank_merge['covered_trials']) * 100,2)

#Check beyond top 10 to make sure no ties
reg_rank_merge[reg_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)

# +
comp_by_year = pr_df[['study_first_submitted_date', 'legal_reg']].groupby(pr_df.study_first_submitted_date.dt.year).agg(['sum', 'count'])

comp_by_year['prct_comp'] = round((comp_by_year['legal_reg']['sum'] / comp_by_year['legal_reg']['count']) * 100,2)

reg_trends = comp_by_year[(comp_by_year.index >= 2009) & (comp_by_year.index <= 2020)]['prct_comp']

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
plt.show()
#plt.savefig('figures/reg_trends.svg')
# -

# # Last Verified Date

# The Final Rule states that all covered trials are required to verify their data once a year. Here we examine how many trials, registered for more than a year, had verified within the last calendar year.

# +
cols = ['nct_id', 'has_results', 'pending_results', 'primary_completion_date', 'completion_date', 'study_first_posted_date', 
        'last_verified_date', 'last_updated_date', 'sponsor', 'act_flag', 'ind_spon', 'drug_trial', 'phase_var', 
        'early_phase', 'late_phase', 'N/A', 'quartile_1', 'quartile_2', 'quartile_3', 'quartile_4']
update_dataset = df[cols].reset_index(drop=True)

update_dataset['scrape_date'] = date(2021,1,18)

# +
#Logic for exclusion
print('We start with all applicable trials: {}'.format(len(df)))

#We exclude trials that were first posted to ClinicalTrials.gov within the last year 
#as they don't have a full year of follow-up
exclude_under_a_year = update_dataset.study_first_posted_date >= pd.Timestamp(2020,1,18)
print("Exclude {} for starting within the last year (since 18 Jan 2020)".format(len(update_dataset[exclude_under_a_year])))
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
#Our data is from 18 Jan 2021 meaning verifications older than 18 January 2020 are officially out of date. 
#However, verifications are usually only provided in "Month Year" format with no date. As such, they are defaulted
#to the beginning of the month. Conservatively, we will treat 1 January 2019 as our cutoff.

# Late Verification = 0, Currently Verified = 1
cohort['comp_veri'] = np.where(cohort.last_verified_date >= pd.Timestamp(2020,1,1), 1,0)
cohort['late_veri'] = np.where(cohort.last_verified_date < pd.Timestamp(2020,1,1), 1,0)
late_veri = len(cohort[cohort.comp_veri == 0])
prct_late = round(cohort.late_veri.sum()/len(cohort)*100,1)
print('{} of {} ({}%) of eligible trials are overdue to verify their records'.format(len(cohort)-late_veri, len(cohort), 100-prct_late))   

# +
#describing the days late for unverified trials

cohort['verification_due'] = cohort.last_verified_date + pd.DateOffset(years=1)
cohort['days_late'] = np.where(cohort.comp_veri == 0, (pd.Timestamp(2021,1,1) - cohort.verification_due) / pd.Timedelta('1 day'), 0)
cohort[cohort['comp_veri'] == 0].days_late.describe()

# +
late_with_update = len(cohort[(cohort.comp_veri == 0) & (cohort.last_updated_date > pd.Timestamp(2020,1,18))])

print('{} trials with a late verification updated since 1 January 2019'.format(late_with_update))
print('This is {}% of the currently late trials'.format(round(late_with_update/late_veri * 100,2)))

# +
ver_bins = np.arange(0,1100 + 1, 100)
xlabels = ['0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1000+']

fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
ax.set_axisbelow(True)
ax.grid(zorder=0)
sns.distplot(np.clip(cohort[cohort['comp_veri'] == 0].days_late,0,1000), hist=True, kde=False, bins=ver_bins, ax=ax,
             hist_kws = {'zorder':10}).set(xlim=(0,1100))
plt.xticks(ver_bins)
ax.set_xticklabels(xlabels)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel('# of Trials', fontsize=25, labelpad=10)
plt.xlabel('Days Late', fontsize=25, labelpad=10)
plt.title("b. Days Late to Verify Trial Data", pad = 20, fontsize = 30)
plt.show()
#plt.savefig('figures/last_verified_1b.svg')
# -

#Can crosstab any variable here.
crosstab(cohort, 'comp_veri', 'act_flag')

# +
#For Table 1
summary = {}

for a in analysis_cols:
    cross = crosstab(cohort, 'comp_veri', a)
    summary[a] = get_prcts(cross)
    
pd.DataFrame(summary).T
# -

y_veri = cohort.comp_veri
x_veri = cohort[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 
                 'quartile_3', 'quartile_4']].reset_index(drop=True)

# +
#Use this cell for crude OR of interest by changeing the value of crude_x

crude_x = cohort[['late_phase', 'N/A']].reset_index(drop=True)

simple_logistic_regression(y_veri,crude_x,cis=.001)

# +
#Outcome here is having a current verification date - adjusted

simple_logistic_regression(y_veri,x_veri, cis=.001)

# +
#Getting an overview of late verifications as a percent of all covered trials

veri_rank = create_ranking(cohort, 'late_veri')
#v_top_10_prct = veri_rank.late_veri.quantile(.95)
veri_rank_merge = veri_rank.merge(covered_trials, on='sponsor')
veri_rank_merge['prct'] = round((veri_rank_merge['late_veri'] / veri_rank_merge['covered_trials']) * 100,2)

veri_rank_merge[veri_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)

# +
#And here we restrict it to only the percent of those in the population that could have a late verification

only_veri_cohort = cohort[['sponsor', 'late_veri']].groupby('sponsor', as_index=False)['late_veri'].agg(['sum','count'])
only_veri_cohort['prct'] = round((only_veri_cohort['sum'] / only_veri_cohort['count']) * 100,2)
merged_veri = only_veri_cohort.merge(covered_trials, on='sponsor')
merged_veri[merged_veri.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)
# -

only_veri_cohort.sort_values(by='sum', ascending=False).head(11)

# # Certificate Analysis

# Sponsors of trials covered under FDAAA can seek delays to the deadline to seek results under certain circumstances. The Final Rule specified that these certificates must be requested prior to when the results would otherwise become does (i.e., a year from primary completion).

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

print('As of 18 Jan 2021, {} ({}%) trials had recieved Certificates of Delay out of {} applicable trials'
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
certificate['scrape_date'] = pd.Timestamp(2021,1,18)
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
plt.show()
#plt.savefig('figures/late_certificate_1c.svg')
# -

#Can crosstab any variable here
crosstab(certificate, 'on_time_cert', 'act_flag')

# +
#For Table 1
summary = {}

for a in analysis_cols:
    cross = crosstab(certificate, 'on_time_cert', a)
    summary[a] = get_prcts(cross)
    
pd.DataFrame(summary).T
# -

x_cert = certificate[['act_flag', 'ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 'quartile_3', 
                      'quartile_4']].reset_index(drop=True)
y_cert = certificate['on_time_cert'].reset_index(drop=True)

# +
#Use this cell to check crude regression analysis of interest:

crude_x = certificate[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_cert,crude_x,cis=.001)

# +
#Outcome here is having an on-time certificate

simple_logistic_regression(y_cert,x_cert, cis=.001)
# -

cert_rank = create_ranking(certificate, 'late_cert')
#c_top_10_prct = cert_rank.late_cert.quantile(.95)
cert_rank_merge = cert_rank.merge(covered_trials, on='sponsor')
cert_rank_merge['prct'] = round((cert_rank_merge['late_cert'] / cert_rank_merge['covered_trials']) * 100,2)
cert_rank_merge[cert_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).reset_index(drop=True).head(11)

only_cert = certificate[['sponsor', 'late_cert']].groupby('sponsor', as_index=False)['late_cert'].agg(['sum','count'])
only_cert['prct'] = round((only_cert['sum'] / only_cert['count']) * 100,2)
merged_cert = only_cert.merge(covered_trials, on='sponsor')
merged_cert[merged_cert.covered_trials >= 50].sort_values(by='prct', ascending=False).head(11)

merged_cert[(merged_cert.covered_trials >= 50) & (merged_cert['count'] >= 5)].sort_values(by='prct', ascending=False).head(15)

# # Document Analysis

# The Final Rule stipulates that a protocol and statistical analysis plan for covered trials are required to be reported alongside results for trials covered under the FDAAA 2007.

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

# +
has_docs_df = doc_df[['nct_id', 'documents']][doc_df.has_documents == 1].reset_index(drop=True)

has_docs_ids = has_docs_df.nct_id.to_list()

# +
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

#fixing incorrect data points that came up in summary review of data
#(verfified in document https://clinicaltrials.gov/ProvidedDocs/10/NCT01866410/Prot_SAP_000.pdf)
#(and in document https://clinicaltrials.gov/ProvidedDocs/42/NCT03241342/Prot_SAP_000.pdf)
bad_index = nct_index_df.index[nct_index_df['document_date'] == 'January 24, 1014'].tolist()[0]
nct_index_df.at[bad_index,'document_date'] = 'January 24, 2014'
bad_index = nct_index_df.index[nct_index_df['document_date'] == 'April 10, 1018'].tolist()[0]
nct_index_df.at[bad_index,'document_date'] = 'April 10, 2018'

nct_index_df['document_date'] = pd.to_datetime(nct_index_df['document_date'])

# +
#The first time you run this notebook on a new dataset, you can get the data on when the documents 
#were last updated by importing and running the "history_scrape" function. However, if you are using 
#the shared data from the project or re-running a prior analysis you can just export and save a CSV that 
#you can then re-load.

#If you already have the output from the above exported to CSV, just run this cell pointing to that file
#if it isn't already in the same directory (this will work assuming no changed to the cloned repo)

try:
    docs_updates = pd.read_csv(parent + '/data/history_scrape_2021-01-18.csv')
except FileNotFoundError:
    from lib.trial_history import history_scrape
    most_recent_doc_update = history_scrape(tqdm(has_docs_ids), date(2021,1,18))
    docs_updates = pd.DataFrame(most_recent_doc_update)
    docs_updates.to_csv('history_scrape_{}.csv'.format(date(2021,1,18)))
# -

#Cleaning and managing the scraped data as above
bad_index = docs_updates.index[docs_updates['document_date'] == 'January 24, 1014'].tolist()[0]
docs_updates.at[bad_index,'document_date'] = 'January 24, 2014'
bad_index = docs_updates.index[docs_updates['document_date'] == 'April 10, 1018'].tolist()[0]
docs_updates.at[bad_index,'document_date'] = 'April 10, 2018'
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

#The number due to report results
all_due = len(merged[(merged.results_due == 1)])

#The number of these that have results
due_results = len(merged[due_reported_filt])

#Pending results should have results eventually (but we can't assess right now)
pending = len(merged[due_unreported_filt & (merged.pending_results == 1)])

#Due, results fully available, and has a protocol and a SAP
prot_and_sap = len(merged[due_reported_filt & (merged.has_protocol == 1) & (merged.has_sap == 1)])

#Due, results fully available, and has a protocol, a SAP, or a proactive statement that no SAP exists
prot_sap_or_no_sap_stmt = len(merged[due_reported_filt & (merged.has_protocol == 1) 
                                     & ((merged.has_sap == 1) | (merged.no_sap == 1))])

#Due, results, prot, sap unaccounted
prot_unaccounted_sap = len(merged[due_reported_filt & (merged.has_protocol == 1) 
                                     & ((merged.has_sap == 0) & (merged.no_sap == 0))])

#Due, results, sap accounted, no prot
sap_unaccounted_prot = len(merged[due_reported_filt & (merged.has_protocol == 0) 
                                     & ((merged.has_sap == 1) | (merged.no_sap == 1))])


#total due and unreported
due_unreported = len(merged[due_unreported_filt])

#No results, but has some form of documents available/accounted for
unreported_any_docs = len(merged[due_unreported_filt & ((merged.has_protocol == 1) | ((merged.has_sap == 1) | (merged.no_sap == 1)))])
# -

print(f'''There are {due_results} trials that are due to report and have subsequently completed clinicaltrials.gov \
quality control meaning results are fully posted. Of these, \
{prot_and_sap} ({round((prot_and_sap/due_results) * 100, 2)}%) have their protocol and sap included in their record. \
An additional {prot_sap_or_no_sap_stmt-prot_and_sap} trials have proactively declared they have no SAP meaning \
{prot_sap_or_no_sap_stmt} ({round((prot_sap_or_no_sap_stmt/due_results) * 100, 2)}%) have all documents and results \
fully accounted for. \
Among trials without results, {unreported_any_docs} ({round((unreported_any_docs/due_unreported)* 100, 2)}%) have \
any form of documentation available. \
''')

# +
just_due_results = merged[(merged.has_results == 1) & (merged.results_due == 1)].reset_index(drop=True)

just_due_results['docs_accounted'] = np.where((just_due_results.has_protocol == 1) & 
                                              ((just_due_results.has_sap == 1) | 
                                               (just_due_results.no_sap == 1)),1,0)
# -

crosstab(just_due_results, 'docs_accounted', 'act_flag')

# +
#For Table 1
summary = {}

for a in analysis_cols:
    cross = crosstab(just_due_results, 'docs_accounted', a)
    summary[a] = get_prcts(cross)
    
pd.DataFrame(summary).T
# -

#The regression doesn't converge with 'act_flag' included as it is perfectly predictive, so this is removed
x_docs = just_due_results[['ind_spon', 'drug_trial', 'late_phase', 'N/A', 'quartile_2', 
                           'quartile_3', 'quartile_4']].reset_index(drop=True)
y_docs = just_due_results.docs_accounted.reset_index(drop=True)

# +
#Use this cell to check crude regression analysis of interest:

crude_x = just_due_results[['quartile_2', 'quartile_3', 'quartile_4']].reset_index(drop=True)

simple_logistic_regression(y_docs,crude_x,cis=.001)

# +
#Adjusted regression

simple_logistic_regression(y_docs,x_docs, cis=.001)
# -

docs_rank = create_ranking(just_due_results, 'docs_accounted', marker = 0)
docs_rank_merge = docs_rank.merge(covered_trials, on='sponsor')
docs_rank_merge['prct'] = round((docs_rank_merge['docs_accounted'] / docs_rank_merge['covered_trials']) * 100,2)
docs_rank_merge[docs_rank_merge.covered_trials >= 50].sort_values(by='prct', ascending=False).head(12)

due_results_spon = df[(df.results_due == 1) & (df.has_results == 1)][['sponsor', 'results_due', 'has_results']].groupby('sponsor', as_index=False).sum()

just_due_spon = docs_rank_merge.merge(due_results_spon, how='left', on='sponsor')
just_due_spon['new_prct'] = round((just_due_spon['docs_accounted'] / just_due_spon['results_due']) * 100,2)
just_due_spon[just_due_spon.covered_trials >= 50].sort_values(by='new_prct', ascending=False).head(14)


