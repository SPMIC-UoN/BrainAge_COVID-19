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
    *(Details of the categorisation process are available in the paper.)*
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

The repository includes a Python script (`Age_predict.py`) that reads the provided dataset and generates the figures presented in the paper. This code is designed to facilitate the replication of our results and to help other researchers understand and extend our work.

### Running the Code:
   - Execute the provided Python script to generate the figures.
   - The code will automatically read the dataset and produce the necessary visualizations, including:
     - **Fig. 1e**: Stability of the brain age predictive model across two scans.
     - **Fig. 2**: Effect of COVID-19 and its pandemic on brain ageing.
     - **Fig. 3a**: Impact of SARS-CoV-2 infection and the COVID-19 pandemic on accelerated brain ageing.
     - **Fig. 3b**: Role of sex in brain ageing during the pandemic (a combination of 6 different figures embedded in a single figure).
     - **Figs. 4b-d**: Influence of socio-demographic factors on brain ageing during the COVID-19 pandemic (each figure is produced by combining 6 different figures).
     - **Fig. 5**: Impact of COVID-19 on cognitive performance across rates of change in brain age gap.

By following these instructions, you can reproduce the key figures from our study and further investigate the impact of the COVID-19 pandemic on brain ageing.

<!-- References -->

[paper-medrxiv-link]: https://www.medrxiv.org/content/10.1101/2024.07.22.24310790v1



