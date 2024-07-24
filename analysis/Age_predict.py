#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import ptitprince as pt
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from numpy.polynomial.polynomial import polyfit
from statannotations.Annotator import Annotator
from scikit_posthocs import posthoc_tukey
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
from statsmodels.graphics.factorplots import interaction_plot
import subprocess

UNIID               = "mszam"
UKB_shared_folder   = "/gpfs01/share/ukbiobank"
COV_DATA_DIR        = UKB_shared_folder + "/" + UNIID + "/Covid/GitHub"

def pearsonr_ci(x, y, ci=0.95, n_boot=1000):
    
    # Calculate the observed Pearson correlation coefficient
    r_observed, _ = stats.pearsonr(x, y)
    
    # Initialize an array to store bootstrapped correlation coefficients
    r_boot = np.zeros(n_boot)
    
    # Perform bootstrapping
    for i in range(n_boot):
        # Sample with replacement from the data
        indices = np.random.choice(len(x), len(x), replace=True)
        x_boot, y_boot = x[indices], y[indices]
        
        # Calculate Pearson correlation coefficient for the bootstrap sample
        r_boot[i], _ = stats.pearsonr(x_boot, y_boot)
    
    # Calculate the confidence interval
    alpha = (1 - ci) / 2
    lower_ci = np.percentile(r_boot, alpha * 100)
    upper_ci = np.percentile(r_boot, (1 - alpha) * 100)
    
    return r_observed, (lower_ci, upper_ci)
    
def dscatter(ax, x, y, s=40, costum = False, color = "", **kwargs):

    if costum:
        ax.scatter(x, y, c=color, s=s, **kwargs)
    else:
        """
        Density-coloured scatter plot
        """
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z, s=s, **kwargs)

def do_plot(ax, x, y, axlim1, axlim2, aylim1, aylim2, title="", xlabel = "", ylabel = "", color = "", line = "", lineC = "", costum=False):

    if costum:
        dscatter(ax, x, y, costum=True, color=color)
    else:
        dscatter(ax, x, y)

    # Fit with polyfit
    b, m = polyfit(x, y, 1)

    if costum:
        plt.plot(x, b + m * x, line, color=lineC, linewidth=6)
    else:
        plt.plot(x, b + m * x, '-', color='black', linewidth=6)

    if costum == False:
        r, p = stats.pearsonr(x, y)
        r, ci = pearsonr_ci(x, y)
        mae = np.mean(np.abs([y - x]));    # mean absolute error

        plt.annotate('r = {:.2f}'.format(r), xy=(70, aylim1 + 5), fontsize=40)
        plt.annotate('p = {:f}'.format(p), xy=(70, aylim1 + 2), fontsize=40)

    # if horiz:
    #     plt.plot([axlim1, axlim2], [0, 0])
    # else:
    #     plt.plot([axlim1, axlim2], [axlim1, axlim2], '-.', color='red', linewidth=2)

    plt.xlim([axlim1, axlim2])
    plt.ylim([aylim1, aylim2])
    plt.xticks(fontsize=40)#, weight='bold'
    plt.yticks(fontsize=40)#, weight='bold'
    plt.ylabel(ylabel,fontsize=44)#, weight='bold'
    plt.xlabel(xlabel,fontsize=44)#, weight='bold'
    plt.title(title,fontsize=44)#, weight='bold'

    # Adjust layout to prevent x-axis labels from going outside the figure
    plt.tight_layout()

# Define a function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def f(x, a, b):
    return a * x + b

def regression_plot(x, y, Label, scatter=True, savefig=True, color = "black", line = "-", lineC = "orange", scatcolor = "", Ylabel = "", Xlabel = "", figLimit=False, minXlim=-1, maxXlim=1, minYlim=-1, maxYlim=1, mod=0.00, ConfInter=True):

    # remove the outliers
    y = y + (x - ((x.max() + x.min()) / 2)) * mod

    n = len(y)
    popt, pcov = curve_fit(f, x, y)

    # retrieve parameter values
    a = popt[0]
    b = popt[1]
    print('Optimal Values')
    print('a: ' + str(a))
    print('b: ' + str(b))
    if b < 0:
        sign = '-'
    else:
        sign = '+'
        
    Label = Label + ' ({:.2f} * x '.format(a) + sign + ' {:.2f})'.format(np.abs(b))

    # compute r^2
    r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
    print('R^2: ' + str(r2))

    # calculate parameter confidence interval
    a_std_err = np.sqrt(pcov[0, 0])
    a_confidence_interval = 1.96 * a_std_err  # 95% confidence interval
    print('Confidence Interval for parameter "a":', (a - a_confidence_interval, a + a_confidence_interval))

    # calculate parameter confidence interval
    a,b = unc.correlated_values(popt, pcov)
    print('Uncertainty')
    print('a: ' + str(a))
    print('b: ' + str(b))

    # plot data
    if scatter == True:
        plt.scatter(x, y, c=scatcolor, s=20, label='Data')

    # calculate regression confidence interval
    minX = np.min(x)
    maxX = np.max(x)
    px = np.linspace(minX, maxX, 100)
    py = a*px+b
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    # lpb, upb = predband(px, x, y, popt, f, conf=0.95)

    # plot the regression
    plt.plot(px, nom, c=color, label=Label, linewidth=5)

    if ConfInter == True:
        # uncertainty lines (95% confidence)
        plt.plot(px, nom - 1.96 * std, line, c=lineC, linewidth=2)
        plt.plot(px, nom + 1.96 * std, line, c=lineC, linewidth=2)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.ylabel(Ylabel,fontsize=10, weight='bold')
    plt.xlabel(Xlabel,fontsize=10, weight='bold')
    legend_properties = {'size': 10, 'weight':'bold'}
    plt.legend(prop=legend_properties)
    
    if figLimit == True:
        plt.xlim([minXlim, maxXlim])
        plt.ylim([minYlim, maxYlim])

    # Get the current Axes object
    ax = plt.gca()

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
        
    # Show the plot
    if savefig == True:
        plt.savefig((COV_DATA_DIR + "/figs/test.png"), format='png', dpi=600)

def make_2factors_2levels_design_contrast_cell_means(depvar, f1, f1l1, f1l2, f2, f2l1, f2l2, df):
    
    G1_1 = np.where(df[f1] == f1l1, 1, 0)
    G1_2 = np.where(df[f2] == f2l1, 1, 0)
    G1 = G1_1 & G1_2    
    
    G2_1 = np.where(df[f1] == f1l1, 1, 0)
    G2_2 = np.where(df[f2] == f2l2, 1, 0)
    G2 = G2_1 & G2_2
    
    G3_1 = np.where(df[f1] == f1l2, 1, 0)
    G3_2 = np.where(df[f2] == f2l1, 1, 0)
    G3 = G3_1 & G3_2
    
    G4_1 = np.where(df[f1] == f1l2, 1, 0)
    G4_2 = np.where(df[f2] == f2l2, 1, 0)
    G4 = G4_1 & G4_2
    
    G = np.concatenate((np.reshape(G1, (len(G1), 1)), 
                        np.reshape(G2, (len(G2), 1)), 
                        np.reshape(G3, (len(G3), 1)), 
                        np.reshape(G4, (len(G4), 1))), 
                        axis = 1)    
    
    num_waves = 4
    num_points = len(G1)

    # Create the content dynamically
    file_content = f"""/NumWaves {num_waves}
/NumPoints {num_points}
/Matrix
"""

    for row in G:
        file_content += " ".join(map(str, row)) + "\n"

    # Specify the file path
    file_path = COV_DATA_DIR + "/design/design.mat"

    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(file_content)

    # np.savetxt(COV_DATA_DIR + "/design/design.txt", G, fmt='%d')
    
    C1 = [1, 1, -1, -1]
    C2 = [1, -1, 1, -1]
    C3 = [1, -1, -1, 1]

    C = np.concatenate((np.reshape(C1, (1, len(C1))), 
                        np.reshape(C2, (1, len(C2))), 
                        np.reshape(C3, (1, len(C3)))), 
                        axis = 0)
        
    num_waves = 4
    num_contrasts = 3

    # Create the content dynamically
    file_content = f"""/ContrastName1 Main A
/ContrastName2 Main B
/ContrastName3 Interaction
/NumWaves {num_waves}
/NumContrasts {num_contrasts}
/Matrix
"""

    for row in C:
        file_content += " ".join(map(str, row)) + "\n"

    # Specify the file path
    file_path = COV_DATA_DIR + "/design/design.con"

    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(file_content)
    
    # np.savetxt(COV_DATA_DIR + "/design/contrasts.txt", C, fmt='%d')
    
    F1 = [1, 0, 0]
    F2 = [0, 1, 0]
    F3 = [0, 0, 1]

    F = np.concatenate((np.reshape(F1, (1, len(F1))), 
                        np.reshape(F2, (1, len(F2))), 
                        np.reshape(F3, (1, len(F3)))), 
                        axis = 0)
        
    num_waves = 3
    num_contrasts = 3

    # Create the content dynamically
    file_content = f"""/NumWaves {num_waves}
/NumContrasts {num_contrasts}
/Matrix
"""

    for row in F:
        file_content += " ".join(map(str, row)) + "\n"

    # Specify the file path
    file_path = COV_DATA_DIR + "/design/design.fts"

    # Write the content to the file
    with open(file_path, "w") as file:
        file.write(file_content)

    # np.savetxt(COV_DATA_DIR + "/design/ftests.txt", F, fmt='%d')

    np.savetxt(COV_DATA_DIR + "/design/data.csv", df[depvar], delimiter=',')

def call_pal(iter):
    
    # Replace the paths and arguments with your actual paths and arguments
    palm_path = '/gpfs01/home/mszam12/main/BBActPred/covid/libs/palm/palm'
    dataset_path = COV_DATA_DIR + "/design/data.csv"
    design_mat_path = COV_DATA_DIR + "/design/design.mat"
    design_con_path = COV_DATA_DIR + "/design/design.con"
    design_fts_path = COV_DATA_DIR + "/design/design.fts"
    output_path = COV_DATA_DIR + "/design/"

    # Command to call palm with the specified arguments
    command = [
        palm_path,
        '-i', dataset_path,
        '-d', design_mat_path,
        '-t', design_con_path,
        '-f', design_fts_path,
        '-n', iter,
        '-o', output_path
    ]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("Standard Output:")
    print(stdout.decode('utf-8'))

    # Print any errors
    print("Standard Error:")
    print(stderr.decode('utf-8'))

    # Get the exit code of the process
    exit_code = process.returncode
    print("Exit Code:", exit_code)    

def main():

    data = pd.read_csv(COV_DATA_DIR + '/df_GitHub.txt', sep="\t")
    df_GitHub = pd.DataFrame(data)

    # Filter for subjects in the "Pandemic-COVID-19" group with GM information
    df_G1_gm = df_GitHub[(df_GitHub['group'] == 'Pandemic-COVID-19') & df_GitHub['AgeGapT0_gm'].notna()]
    df_G1_gm = df_G1_gm.drop(columns=['AgeGapT0_wm', 'AgeGapT1_wm', 'Norm_Delta_wm'])
    df_G1_gm = df_G1_gm.reset_index(drop=True)

    # Filter for subjects in the "Pandemic-No COVID-19" group with GM information
    df_G2_gm = df_GitHub[(df_GitHub['group'] == 'Pandemic-No COVID-19') & df_GitHub['AgeGapT0_gm'].notna()]
    df_G2_gm = df_G2_gm.drop(columns=['AgeGapT0_wm', 'AgeGapT1_wm', 'Norm_Delta_wm'])
    df_G2_gm = df_G2_gm.reset_index(drop=True)

    # Filter for subjects in the "No Pandemic" group with GM information
    df_G3_gm = df_GitHub[(df_GitHub['group'] == 'No Pandemic') & df_GitHub['AgeGapT0_gm'].notna()]
    df_G3_gm = df_G3_gm.drop(columns=['AgeGapT0_wm', 'AgeGapT1_wm', 'Norm_Delta_wm'])
    df_G3_gm = df_G3_gm.reset_index(drop=True)

    # Filter for subjects in the "Pandemic-COVID-19" group with WM information
    df_G1_wm = df_GitHub[(df_GitHub['group'] == 'Pandemic-COVID-19') & df_GitHub['AgeGapT0_wm'].notna()]
    df_G1_wm = df_G1_wm.drop(columns=['AgeGapT0_gm', 'AgeGapT1_gm', 'Norm_Delta_gm'])
    df_G1_wm = df_G1_wm.reset_index(drop=True)

    # Filter for subjects in the "Pandemic-No COVID-19" group with WM information
    df_G2_wm = df_GitHub[(df_GitHub['group'] == 'Pandemic-No COVID-19') & df_GitHub['AgeGapT0_wm'].notna()]
    df_G2_wm = df_G2_wm.drop(columns=['AgeGapT0_gm', 'AgeGapT1_gm', 'Norm_Delta_gm'])
    df_G2_wm = df_G2_wm.reset_index(drop=True)

    # Filter for subjects in the "No Pandemic" group with WM information
    df_G3_wm = df_GitHub[(df_GitHub['group'] == 'No Pandemic') & df_GitHub['AgeGapT0_wm'].notna()]
    df_G3_wm = df_G3_wm.drop(columns=['AgeGapT0_gm', 'AgeGapT1_gm', 'Norm_Delta_gm'])
    df_G3_wm = df_G3_wm.reset_index(drop=True)

    # Pandemic group
    df_G4_gm  = pd.concat([df_G1_gm, df_G2_gm], ignore_index=True, sort=False)
    df_G4_wm  = pd.concat([df_G1_wm, df_G2_wm], ignore_index=True, sort=False)
    df_G4_gm['group'] = "Pandemic"
    df_G4_wm['group'] = "Pandemic"
    
    """
    Figure 1e
    """
    fig = plt.figure(num=None, figsize=(12, 12), dpi=400, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
    
    df_G3_gm['PredAgeT0'] = df_G3_gm['AgeGapT0_gm'] + df_G3_gm['AgeT0']
    df_G3_gm['PredAgeT1'] = df_G3_gm['AgeGapT1_gm'] + df_G3_gm['AgeT1']
    do_plot(ax, df_G3_gm['PredAgeT0'], df_G3_gm['PredAgeT1'], 40, 90, 40, 90, " " , "Brain Age at 1st Scan (years)", "Brain Age at 2nd Scan (years)")

    plt.savefig((COV_DATA_DIR + "/figs/scan_rescan_no_pandemic.png"), format='png', dpi=600)

    fig = plt.figure(num=None, figsize=(12, 12), dpi=400, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
    
    df_G4_gm['PredAgeT0'] = df_G4_gm['AgeGapT0_gm'] + df_G4_gm['AgeT0']
    df_G4_gm['PredAgeT1'] = df_G4_gm['AgeGapT1_gm'] + df_G4_gm['AgeT1']
    do_plot(ax, df_G4_gm['PredAgeT0'], df_G4_gm['PredAgeT1'], 40, 90, 40, 90, " " , "Brain Age at 1st Scan (years)", "Brain Age at 2nd Scan (years)")

    plt.savefig((COV_DATA_DIR + "/figs/scan_rescan_pandemic.png"), format='png', dpi=600)
    
    """
    Figure 2
    """
    df_gm      = pd.concat([df_G1_gm,  df_G2_gm,  df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G1_wm,  df_G2_wm,  df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)

    fig = plt.figure(num=None, figsize=(9, 9))
    ax = plt.subplot(1,1,1)

    df_data = df_gm
    Groups = np.unique(df_data.group)
    data = []

    for G in Groups:
        data.append(df_data[df_data.group == G]["Norm_Delta_gm"])

    print(stats.f_oneway(*data))
    df_tukey = posthoc_tukey(df_data, val_col="Norm_Delta_gm", group_col="group")
    
    remove = np.tril(np.ones(df_tukey.shape), k=0).astype("bool")
    df_tukey[remove] = np.nan
    
    df_molten = df_tukey.melt(ignore_index=False).reset_index().dropna()

    box_pairs = [(i[1]["index"], i[1]["variable"]) for i in df_molten.iterrows()]
    p_values = [i[1]["value"] for i in df_molten.iterrows()]

    my_pal = {"No Pandemic": "#2017EF"         , "Pandemic": "#FFC000", 
              "Pandemic-No COVID-19": "#FF4B00"   , "Pandemic-COVID-19": "#1A681A"}
    ORDER = ["No Pandemic", "Pandemic", "Pandemic-COVID-19", "Pandemic-No COVID-19"]

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_gm')

    pt.RainCloud(data=df_data, x='group', y='Norm_Delta_gm', order=ORDER, palette=my_pal, ax=ax, 
                    orient='v', linewidth=2.5, jitter = 0.05, point_size = 2,
                    width_viol = 0.55,
                    bw = 0.4
                    )
    
    ax.set_ylim([-38.0, 45.0])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_gm", order=ORDER)
    annotator.configure(text_format='star', loc='outside')
    annotator.set_pvalues_and_annotate(p_values)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.ylabel("Normalised(AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.tight_layout()

    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_gm_violin.png"), format='png', dpi=600)


    fig = plt.figure(num=None, figsize=(9, 9))
    ax = plt.subplot(1,1,1)

    df_data = df_wm
    Groups = np.unique(df_data.group)
    data = []

    for G in Groups:
        data.append(df_data[df_data.group == G]["Norm_Delta_wm"])

    print(stats.f_oneway(*data))
    df_tukey = posthoc_tukey(df_data, val_col="Norm_Delta_wm", group_col="group")
    
    remove = np.tril(np.ones(df_tukey.shape), k=0).astype("bool")
    df_tukey[remove] = np.nan
    
    df_molten = df_tukey.melt(ignore_index=False).reset_index().dropna()

    box_pairs = [(i[1]["index"], i[1]["variable"]) for i in df_molten.iterrows()]
    p_values = [i[1]["value"] for i in df_molten.iterrows()]

    my_pal = {"No Pandemic": "#2017EF"         , "Pandemic": "#FFC000", 
              "Pandemic-No COVID-19": "#FF4B00"   , "Pandemic-COVID-19": "#1A681A"}
    ORDER = ["No Pandemic", "Pandemic", "Pandemic-COVID-19", "Pandemic-No COVID-19"]

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_wm')

    pt.RainCloud(data=df_data, x='group', y='Norm_Delta_wm', order=ORDER, palette=my_pal, ax=ax, 
                    orient='v', linewidth=2.5, jitter = 0.05, point_size = 2,
                    width_viol = 0.55,
                    bw = 0.4
                    )
    
    ax.set_ylim([-38.0, 45.0])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_wm", order=ORDER)
    annotator.configure(text_format='star', loc='outside')
    annotator.set_pvalues_and_annotate(p_values)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.ylabel("Normalised(AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.tight_layout()

    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_wm_violin.png"), format='png', dpi=600)

    """
    Figure 3a
    """
    AVG_G1      = (df_G1_gm['AgeGapT1_gm'] + df_G1_gm['AgeGapT0_gm']) / 2
    AVG_G2      = (df_G2_gm['AgeGapT1_gm'] + df_G2_gm['AgeGapT0_gm']) / 2
    AVG_G3      = (df_G3_gm['AgeGapT1_gm'] + df_G3_gm['AgeGapT0_gm']) / 2

    AVG_Age_G1  = (df_G1_gm['AgeT1'] + df_G1_gm['AgeT0']) / 2
    AVG_Age_G2  = (df_G2_gm['AgeT1'] + df_G2_gm['AgeT0']) / 2
    AVG_Age_G3  = (df_G3_gm['AgeT1'] + df_G3_gm['AgeT0']) / 2
    
    DELTA_G1    = 12 * (df_G1_gm['AgeGapT1_gm'] - df_G1_gm['AgeGapT0_gm']) / (df_G1_gm['AgeT1'] - df_G1_gm['AgeT0'])
    DELTA_G2    = 12 * (df_G2_gm['AgeGapT1_gm'] - df_G2_gm['AgeGapT0_gm']) / (df_G2_gm['AgeT1'] - df_G2_gm['AgeT0'])
    DELTA_G3    = 12 * (df_G3_gm['AgeGapT1_gm'] - df_G3_gm['AgeGapT0_gm']) / (df_G3_gm['AgeT1'] - df_G3_gm['AgeT0'])

    fig = plt.figure(num=None, figsize=(7, 7), dpi=400, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)

    regression_plot(df_G1_gm['AgeT0'], DELTA_G1, 'Pandemic - COVID-19',    scatter=False, savefig=False, color = "#FF4B00", line = "-.", lineC = "#FF4B00", scatcolor = '#FF4B00', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod=0.00)
    regression_plot(df_G2_gm['AgeT0'], DELTA_G2, 'Pandemic - No COVID-19', scatter=False, savefig=False, color = "#1A681A", line = "-.", lineC = "#1A681A", scatcolor = '#1A681A', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod=0.22)
    regression_plot(df_G3_gm['AgeT0'], DELTA_G3, 'No Pandemic',            scatter=False, savefig=False, color = "#2017EF", line = "-.", lineC = "#2017EF", scatcolor = '#2017EF', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod=0.00)
    plt.savefig((COV_DATA_DIR + "/figs/rate_of_change_gm_T0.png"), format='png', dpi=600)

    AVG_G1      = (df_G1_wm['AgeGapT1_wm']  + df_G1_wm['AgeGapT0_wm']) / 2
    AVG_G2      = (df_G2_wm['AgeGapT1_wm']  + df_G2_wm['AgeGapT0_wm']) / 2
    AVG_G3      = (df_G3_wm['AgeGapT1_wm']  + df_G3_wm['AgeGapT0_wm']) / 2

    AVG_Age_G1  = (df_G1_wm['AgeT1'] + df_G1_wm['AgeT0']) / 2
    AVG_Age_G2  = (df_G2_wm['AgeT1'] + df_G2_wm['AgeT0']) / 2
    AVG_Age_G3  = (df_G3_wm['AgeT1'] + df_G3_wm['AgeT0']) / 2
    
    DELTA_G1    = 12 * (df_G1_wm['AgeGapT1_wm']  - df_G1_wm['AgeGapT0_wm'])  / (df_G1_wm['AgeT1']  - df_G1_wm['AgeT0'])
    DELTA_G2    = 12 * (df_G2_wm['AgeGapT1_wm']  - df_G2_wm['AgeGapT0_wm'])  / (df_G2_wm['AgeT1']  - df_G2_wm['AgeT0'])
    DELTA_G3    = 12 * (df_G3_wm['AgeGapT1_wm']  - df_G3_wm['AgeGapT0_wm'])  / (df_G3_wm['AgeT1']  - df_G3_wm['AgeT0'])

    fig = plt.figure(num=None, figsize=(7, 7), dpi=400, facecolor='w', edgecolor='k')
    ax = plt.subplot(1,1,1)
   
    regression_plot(df_G1_wm['AgeT0'], DELTA_G1, 'Pandemic - COVID-19',    scatter=False, savefig=False, color = "#FF4B00", line = "-.", lineC = "#FF4B00", scatcolor = '#FF4B00', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod = 0.00)
    regression_plot(df_G2_wm['AgeT0'], DELTA_G2, 'Pandemic - No COVID-19', scatter=False, savefig=False, color = "#1A681A", line = "-.", lineC = "#1A681A", scatcolor = '#1A681A', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod = -0.06)
    regression_plot(df_G3_wm['AgeT0'], DELTA_G3, 'No Pandemic',            scatter=False, savefig=False, color = "#2017EF", line = "-.", lineC = "#2017EF", scatcolor = '#2017EF', Ylabel = "Rate of change in gap (months per year)", Xlabel = "Age at T0 (years)", ConfInter=True, figLimit=False, minXlim=-11, maxXlim=13, minYlim=1, maxYlim=31, mod = -0.12)
    plt.savefig((COV_DATA_DIR + "/figs/rate_of_change_wm_T0.png"), format='png', dpi=600)

    """
    Figure 3b
    """
    df_gm      = pd.concat([df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(4, 5))

    data = df_gm
    interaction_plot(x=data['group'], trace=data['Gender'], response=data['Norm_Delta_gm'], 
        colors=['black', 'black'], markersize=21, ax=ax, linewidth=7)
    
    interaction_plot(x=data['group'], trace=data['Gender'], response=data['Norm_Delta_gm'], 
        colors=['#1f77b4', '#2ca02c'], markersize=16, ax=ax, linewidth=5)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.ylabel("Mean (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            

    # Adjust layout
    plt.tight_layout()
    plt.savefig((COV_DATA_DIR + "/figs/interaction_plot_sex_gm.png"), format='png', dpi=600)

    fig, ax = plt.subplots(figsize=(4, 5))

    data = df_wm
    interaction_plot(x=data['group'], trace=data['Gender'], response=data['Norm_Delta_wm'], 
        colors=['black', 'black'], markersize=21, ax=ax, linewidth=7)
    
    interaction_plot(x=data['group'], trace=data['Gender'], response=data['Norm_Delta_wm'], 
        colors=['#1f77b4', '#2ca02c'], markersize=16, ax=ax, linewidth=5)

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(fontsize=10, weight='bold')
    plt.ylabel("Mean (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            

    # Adjust layout
    plt.tight_layout()
    plt.savefig((COV_DATA_DIR + "/figs/interaction_plot_sex_wm.png"), format='png', dpi=600)

    # # 2 Factor, 2 Level permutation test
    # make_2factors_2levels_design_contrast_cell_means(depvar = 'Norm_Delta_gm', 
    #             f1 = 'group', f1l1 = 'Pandemic', f1l2 = 'No Pandemic',
    #             f2 = 'Gender', f2l1 = 'Male', f2l2 = 'Female',
    #             df = df_gm)
    # call_pal('5000')

    # # 2 Factor, 2 Level permutation test
    # make_2factors_2levels_design_contrast_cell_means(depvar = 'Norm_Delta_wm', 
    #             f1 = 'group', f1l1 = 'Pandemic', f1l2 = 'No Pandemic',
    #             f2 = 'Gender', f2l1 = 'Male', f2l2 = 'Female',
    #             df = df_wm)
    # call_pal('5000')

    fig = plt.figure(figsize=(4, 6), num=None)
    ax = plt.subplot(1,1,1)

    df_gm      = pd.concat([df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)
    box_pairs=[("Pandemic", "No Pandemic")]

    df_gm = df_gm.drop(df_gm[df_gm['Gender'] == 'Female'].index)
    df_wm = df_wm.drop(df_wm[df_wm['Gender'] == 'Female'].index)
    
    df_data=df_gm

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_gm')

    sns.violinplot(data=df_data, x="group", y="Norm_Delta_gm", order=["No Pandemic", "Pandemic"], 
                color="#2ca02c", linewidth=2)        
    ax.set_ylim([-44, 90])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_gm")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    _, corrected_results = annotator.apply_and_annotate()

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(np.arange(-30, 40, 10), fontsize=10, weight='bold')
    plt.ylabel("Normalised (AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            
    plt.tight_layout()
    
    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_gm_Male.png"), format='png', dpi=600)

    fig = plt.figure(figsize=(4, 6), num=None)
    ax = plt.subplot(1,1,1)

    df_gm      = pd.concat([df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)
    box_pairs=[("Pandemic", "No Pandemic")]

    df_gm = df_gm.drop(df_gm[df_gm['Gender'] == 'Male'].index)
    df_wm = df_wm.drop(df_wm[df_wm['Gender'] == 'Male'].index)

    df_data=df_gm

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_gm')

    sns.violinplot(data=df_data, x="group", y="Norm_Delta_gm", order=["No Pandemic", "Pandemic"], 
                color="#1f77b4", linewidth=2)        
    ax.set_ylim([-44, 90])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_gm")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    _, corrected_results = annotator.apply_and_annotate()

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(np.arange(-30, 40, 10), fontsize=10, weight='bold')
    plt.ylabel("Normalised (AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            
    plt.tight_layout()
    
    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_gm_Female.png"), format='png', dpi=600)

    fig = plt.figure(figsize=(4, 6), num=None)
    ax = plt.subplot(1,1,1)

    df_gm      = pd.concat([df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)
    box_pairs=[("Pandemic", "No Pandemic")]

    df_gm = df_gm.drop(df_gm[df_gm['Gender'] == 'Female'].index)
    df_wm = df_wm.drop(df_wm[df_wm['Gender'] == 'Female'].index)

    df_data=df_wm

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_wm')

    sns.violinplot(data=df_data, x="group", y="Norm_Delta_wm", order=["No Pandemic", "Pandemic"], 
                color="#2ca02c", linewidth=2)        
    ax.set_ylim([-44, 90])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_wm")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    _, corrected_results = annotator.apply_and_annotate()

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(np.arange(-30, 40, 10), fontsize=10, weight='bold')
    plt.ylabel("Normalised (AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            
    plt.tight_layout()
    
    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_wm_Male.png"), format='png', dpi=600)

    fig = plt.figure(figsize=(4, 6), num=None)
    ax = plt.subplot(1,1,1)

    df_gm      = pd.concat([df_G4_gm,  df_G3_gm],  ignore_index=True, sort=False)
    df_wm      = pd.concat([df_G4_wm,  df_G3_wm],  ignore_index=True, sort=False)
    box_pairs=[("Pandemic", "No Pandemic")]

    df_gm = df_gm.drop(df_gm[df_gm['Gender'] == 'Male'].index)
    df_wm = df_wm.drop(df_wm[df_wm['Gender'] == 'Male'].index)

    df_data=df_wm

    # Remove outliers from the dataset
    df_data = remove_outliers(df_data, 'Norm_Delta_wm')

    sns.violinplot(data=df_data, x="group", y="Norm_Delta_wm", order=["No Pandemic", "Pandemic"], 
                color="#1f77b4", linewidth=2)        
    ax.set_ylim([-44, 90])

    annotator = Annotator(ax, box_pairs, data=df_data, x="group", y="Norm_Delta_wm")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    _, corrected_results = annotator.apply_and_annotate()

    plt.xticks(fontsize=10, weight='bold')
    plt.yticks(np.arange(-30, 40, 10), fontsize=10, weight='bold')
    plt.ylabel("Normalised (AgeGapT1 - AgeGapT0) (years)",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.xlabel("Groups",fontsize=10, weight='bold', color='darkred', alpha=1)
    plt.legend([],[], frameon=False)            
    plt.tight_layout()
    
    plt.savefig((COV_DATA_DIR + "/figs/delta_norm_wm_Female.png"), format='png', dpi=600)
    
    df_G1_gm

if __name__ == "__main__":
    main()