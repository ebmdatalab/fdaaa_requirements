# Assessments of compliance with various requirements of the FDAAA 2007

## Overview

This repository contains everything you need to replicate our analysis of various FDAAA 2007 requirements. In this project we assessed requirements for timely registration, annual data verification, seeking reporting delays on-time, and reporting documents alongside results. This repository is designed to work as a Docker container but should also work outside of Docker assuming the environment described in `requirements.txt`.

## Data Sources

Each working day we download the full data from ClinicalTrials.gov as part of our [FDAAA TrialsTracker](https://fdaaa.trialstracker.net/) project. The data is [available in XML format](https://clinicaltrials.gov/ct2/resources/download) that we convert to JSON strings. We store these in CSV format, delimited by the `þ` character for ease of use with tools like BigQuery, however they are also able to be parsed as ndjson files. The code for that downloading and processing is located as part of our TrialsTracker ["clinicaltrials-act-converter" repo](https://github.com/ebmdatalab/clinicaltrials-act-converter). Additional code for the FDAAA TrialsTracker is located [here](https://github.com/ebmdatalab/clinicaltrials-act-tracker).

Adapting the code used to identify applicable trials for the TrialsTracker, we are able to take the raw data of the entirety of ClinicalTrials.gov on a given day and convert it to CSVs with the relevant data necessary for the analysis. The raw data file for this analysis is too big to easily manage in the repo. You can access it [here](https://doi.org/10.6084/m9.figshare.12789902) or [here](https://www.dropbox.com/s/awlhqwjtkzp6t4b/clinicaltrials_raw_clincialtrials_json_2021-01-18.csv.zip?dl=0). The processed data that drives the analysis is available in the repository. We are happy to freely share any additional full archives of ClinicalTrials.gov from our database. Please email us at [ebmdatalab@phc.ox.ac.uk](mailto:ebmdatalab@phc.ox.ac.uk) and we can discuss the best way to get you the data.

## Data Processing and Analysis

### *notebooks*

The `notebooks` directory contains the Jupyter Notebook with all the code for the project. 

### *Figures*

All figures from the notebook are available in the `Figures` directory in vector (.svg) formats.

### *lib*

The `lib` directory contains .py files with functions to import for the processing and analysis of the data.

### *Data*

Files necessary for both the raw data processing and the overall analysis:

>`fdaaa_regulatory_snapshot.csv` is our archive of the old "is_fda_regulated" field from ClinicalTrials.gov used in our pACT identification logic. This data is taken from the 5 January 2017 archive of ClinicalTrials.gov available from the [Clinical Trials Transformation Initiative](https://aact.ctti-clinicaltrials.org/snapshots). It helps us to conservatively identify pACTs.

>`applicable_trials_2021-01-18.csv` is the main processed dataset for the study filtered to only include ACTs/pACTs.

>`all_trials_2021-01-18.csv.zip` is the processed data for the entire ClinicalTrials.gov dataset which we need for certain part of the analysis.

>`history_scrape_2021-01-18.csv` is some data scraped from the ClinicalTrials.gov archive site about when documents were first uploaded.

Additional files and directories in the repository are for use with Docker as described below.

## How to view the notebooks and use the repository with Docker

The analysis Notebooks live in the `notebooks/` folder (with an `ipynb` extension). You can most easily view them [on nbviewer](https://nbviewer.jupyter.org/github/ebmdatalab/fdaaa_trends/tree/master/notebooks/), though looking at them in Github should also work.

The repository has also been set up to run in Docker to ensure a compatible environment. While the notebook should be able to run in the current directory without Docker (assuming the environment specified in `requirements.txt`) you can follow the directions in the `Developers.md` file to clone this repository and run any code of interest within a Docker container on your machine.

## How to cite

You can cite the accompanying paper in [*JAMA Internal Medicine*](https://doi.org/10.1001/jamainternmed.2021.2036)

DeVito NJ, Goldacre B. Evaluation of Compliance With Legal Requirements Under the FDA Amendments Act of 2007 for Timely Registration of Clinical Trials, Data Verification, Delayed Reporting, and Trial Document Submission. JAMA Intern Med 2021; published online May 24. DOI:10.1001/jamainternmed.2021.2036.
