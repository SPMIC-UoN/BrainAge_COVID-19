# Brain Age Prediction Study - Data and Code Repository

This repository contains the data and code necessary to replicate the figures and findings presented in our paper on the impact of the COVID-19 pandemic on brain ageing: [A. R. Mohammadi-Nejad, M. Craig, E. Cox, X. Chen, R. G. Jenkins, S. Francis, S. N. Sotiropoulos, D. P. Auer, “Brains Under Stress: Unravelling the Effects of the COVID-19 Pandemic on Brain Ageing”, medrxiv, 2024][paper-medrxiv-link].

## Data Description
The provided dataset is a text file containing data for 1,336 participants. Each row represents a participant, and the columns include various attributes related to their chronological age, brain age predictions, and socio-demographic factors. Below is a detailed description of each column:
- AgeT0: The chronological age of the participants at their first scan date.
- AgeT1: The chronological age of the participants at their second scan date.
- Group: Categorisation of participants based on the time of their second scan. The categories are:
  - `Pandemic-COVID-19`: Participants scanned before and after the pandemic, who contracted COVID-19.
  - `Pandemic-No COVID-19`: Participants scanned before and after the pandemic, who did not contract COVID-19.
  - `No Pandemic`: Participants scanned twice before the pandemic.
    *(Details of the categorization process are available in the paper.)*
- Gender: The sex of the participants, with possible values Male or Female.
- AgeGapT0_gm: The predicted brain age gap at the first time point, estimated based on the Grey Matter (GM) model.
- AgeGapT1_gm: The predicted brain age gap at the second time point, estimated based on the Grey Matter (GM) model.
- Norm_Delta_gm: The rate of change in the brain age gap between the two scans, normalized and estimated using the Grey Matter (GM) model.
- AgeGapT0_wm: The predicted brain age gap at the first time point, estimated based on the White Matter (WM) model.
- AgeGapT1_wm: The predicted brain age gap at the second time point, estimated based on the White Matter (WM) model.
- Norm_Delta_wm: The rate of change in the brain age gap between the two scans, normalized and estimated using the White Matter (WM) model.
- Health_Categ: The health score of participants, categorized into three levels as a deprivation index: `Low`, `Medium`, and `High`.
- Empl_Categ: The employment score of participants, categorized into three levels as a deprivation index: `Low`, `Medium`, and `High`.
- House_Categ: The housing score of participants, categorized into three levels as a deprivation index: `Low`, `Medium`, and `High`.
- Incom_Categ: The income score of participants, categorized into three levels as a deprivation index: `Low`, `Medium`, and `High`.
- Educ_Categ: The education score of participants, categorized into three levels as a deprivation index: `Low`, `Medium`, and `High`.

## Code Description

The repository includes Python code that reads the provided dataset and generates the figures presented in the paper. This code is designed to facilitate the replication of our results and to help other researchers in understanding and extending our work.

## Usage

1. **Data Preparation**: Ensure the dataset is correctly formatted and located in the directory where the code will be executed.
2. **Running the Code**:
   1. Execute the provided Python scripts to generate the figures.
   2. The code will automatically read the dataset and produce the necessary visualizations.
3. **Customization**:
   1. Modify the code as needed to explore different aspects of the data.
   2. Adapt it for related research questions.

By following these instructions, you can reproduce the key figures from our study and further investigate the impact of the COVID-19 pandemic on brain ageing.

<!-- References -->

[paper-medrxiv-link]: https://www.medrxiv.org/content/10.1101/2024.07.22.24310790v1



