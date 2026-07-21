# AMP-PD — Parkinson's UPDRS progression from CSF proteomics

Dataset Description
The goal of this competition is to predict the course of Parkinson's disease (PD) using protein abundance data. The complete set of proteins involved in PD remains an open research question and any proteins that have predictive value are likely worth investigating further. The core of the dataset consists of protein abundance values derived from mass spectrometry readings of cerebrospinal fluid (CSF) samples gathered from several hundred patients. Each patient contributed several samples over the course of multiple years while they also took assessments of PD severity.

This is a time-series code competition: you will receive test set data and make predictions with a time-series API. See the evaluation_details.txt for details.

Files
train_peptides.csv Mass spectrometry data at the peptide level. Peptides are the component subunits of proteins.

visit_id - ID code for the visit.
visit_month - The month of the visit, relative to the first visit by the patient.
patient_id - An ID code for the patient.
UniProt - The UniProt ID code for the associated protein. There are often several peptides per protein.
Peptide - The sequence of amino acids included in the peptide. See this table for the relevant codes. Some rare annotations may not be included in the table. The test set may include peptides not found in the train set.
PeptideAbundance - The frequency of the amino acid in the sample.
train_proteins.csv Protein expression frequencies aggregated from the peptide level data.

visit_id - ID code for the visit.
visit_month - The month of the visit, relative to the first visit by the patient.
patient_id - An ID code for the patient.
UniProt - The UniProt ID code for the associated protein. There are often several peptides per protein. The test set may include proteins not found in the train set.
NPX - Normalized protein expression. The frequency of the protein's occurrence in the sample. May not have a 1:1 relationship with the component peptides as some proteins contain repeated copies of a given peptide.
train_clinical_data.csv

visit_id - ID code for the visit.
visit_month - The month of the visit, relative to the first visit by the patient.
patient_id - An ID code for the patient.
updrs_[1-4] - The patient's score for part N of the Unified Parkinson's Disease Rating Scale. Higher numbers indicate more severe symptoms. Each sub-section covers a distinct category of symptoms, such as mood and behavior for Part 1 and motor functions for Part 3.
upd23b_clinical_state_on_medication - Whether or not the patient was taking medication such as Levodopa during the UPDRS assessment. Expected to mainly affect the scores for Part 3 (motor function). These medications wear off fairly quickly (on the order of one day) so it's common for patients to take the motor function exam twice in a single month, both with and without medication.
supplemental_clinical_data.csv Clinical records without any associated CSF samples. This data is intended to provide additional context about the typical progression of Parkinsons. Uses the same columns as train_clinical_data.csv.

example_test_files/ Data intended to illustrate how the API functions. Includes the same columns delivered by the API (ie no updrs columns).

public_timeseries_testing_util.py A file for running custom API tests.

## Data (already staged in your workspace, `/workspace`)
- `train_clinical_data.csv` — training patients: `visit_id`, `patient_id`, `visit_month`, `updrs_1..4` (targets).
- `train_proteins.csv`, `train_peptides.csv` — full CSF mass-spec features (NPX / peptide abundance) for all visits.
- `test.csv` — `visit_id`, `patient_id`, `visit_month` for held-out patients (predict their UPDRS).
- `task_descriptor.txt` — original competition description.

## Specialized library to use — REQUIRED
Use **LightAutoML** (`TabularAutoML`, pre-installed) for the multi-target regression after pivoting proteins/peptides to per-visit features. You MAY `pip install tsfresh` for time-series features. Do not hand-roll a single boosting model.

## Training discipline — REQUIRED
Validate SMAPE on held-out patients, keep the best model, print `VALIDATION_SCORE=<smape>`.

## Submission — REQUIRED
Write **`/workspace/submission.csv`** with columns:
- `visit_id` — from `test.csv`
- `updrs_1`, `updrs_2`, `updrs_3`, `updrs_4` — predicted scores

Scoring: **SMAPE** across the four UPDRS targets (lower is better).
