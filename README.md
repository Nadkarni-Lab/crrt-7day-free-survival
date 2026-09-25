# Temporal XGBoost Model for Predicting 7-Day Survival in Patients on CRRT
This repository contains the code used to develop and evaluate an XGBoost temporal prediction model for 7-day survival and CRRT treatment status in critically ill adult patients receiving Continuous Renal Replacement Therapy (CRRT). The model utilizes rolling-windows during feature extraction to capture longitudinal trends in clinical, laboratory, and treatment-related variables. It also incorporates static patient characteristics. This framework enabled dynamic prediction using both a patient's baseline characteristics and their clinical course during CRRT treatment.

## Development Study Cohort
The study cohort was derived from a prospective, multicenter observation study across five academic medical centers in North America on the practices of Continuous Renal Replacement Therapy across five years. The dataset included critically ill adult patients who received CRRT during their intensive care unit (ICU) stay. *[Rewa et al.](https://doi.org/10.1016/j.xkme.2023.100641)*

### Cohort Construction
**Inclusion and Exclusion Criteria**
* Include any patient who received CRRT during ICU stay. 
* Exclude patients with evidence of chronic dialysis dependency, when determinable from available data.
* Exclude patients with end-stage renal disease (ESRD), identified using ICD-9 code 585.6 or ICD-10 code N18.6 prior to CRRT initiation.
* Exclude patients with a total CRRT treatment duration of less than 1 day, after standardization of dates (see [Date Standardization](#date-standardization) for more details).

**Cohort Construction Procedure**
1. Identify patients who received CRRT.
2. For each patient, select the first hospitalization during which CRRT was administered.
3. Identify all CRRT treatment episodes within the selected hospitalization.
4. Determine the start and end dates for each CRRT treatment episode using institutional documentation and available longitudinal data sources.
5. Identify the first CRRT treatment episode lasting more than 24 hours (based on standardized treatment dates and times - see [Date Standardization](#date-standardization) for more detail)
6. Identify the ICU admission and discharge associated with the first qualifying CRRT treatment episode.

## Outcome Definition
The outcome was if the patient achieved CRRT-free survival 7 days after CRRT initiation.

### CRRT-Free Survival
A patient was considered to have achieved CRRT-free survival after 7 days if all of the following conditions were met:
1. The patient was no longer receiving CRRT by Day 7:
    - CRRT stop date occurred before the start of Day 7.
    - Patient did not resume CRRT during the same ICU admission.
2. The patient survived through Day 7:
    - Death date occurred on or after Day 8, or no death was recorded.

### Not CRRT-Free/Non-Survival
A patient was considered not achieving CRRT-free survival at 7 days if either of the following was true:
1. Ongoing CRRT treatment at Day 7
2. Death before the end of Day 7

## Pre-Processing and Cleaning Data
Read `preprocessing_instructions.md` within `docs/` for **feature** pre-processing and cleaning.

### Date Standardization
All timestamps were standardized to calendar dates by removing the time component to ensure consistency across the dataset. For example, a CRRT initiation timestamp of May 5, 12:30 was recorded as May 5 (May 5 00:00). This standardization was applied consistently across all patients and data sources.
