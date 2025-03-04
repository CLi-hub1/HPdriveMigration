import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ## T-test plot, SIfigures
control = pd.read_excel(r'filepath\SIfigureS9_Ttest_control.xlsx')
treat = pd.read_excel(r'filepath\SIfigureS9_Ttest_treatment.xlsx')
treat = treat.groupby('GEOID').mean().reset_index()

co = control.iloc[:, 8:15]
tr = treat.iloc[:, 5:12]

fig, axs = plt.subplots(nrows=1, ncols=7, figsize=(30,5), dpi=600)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24


for i in range(7):
    col_co = co.iloc[:, i]
    col_tr = tr.iloc[:, i]

    # Two-sample T-test
    t_stat, p_val = ttest_ind(col_co, col_tr)

    # Check significance level
    significance = ''
    if p_val < 0.01:
        significance = '***'
    elif p_val < 0.05:
        significance = '**'
    elif p_val < 0.1:
        significance = '*'

    # Bar plot
    sns.barplot(data=pd.DataFrame({'Control': col_co, 'Treatment': col_tr}), ax=axs[i])
    axs[i].errorbar(x=[0, 1], y=[col_co.mean(), col_tr.mean()], yerr=[col_co.std(), col_tr.std()], fmt='o', color='black')
    axs[i].text(0.15, 70.95, f' {significance}P = {p_val:.4f}', color='black', fontsize=14) ## ha='center', va='top',
    # axs[i].set_title(f' {significance}P = {p_val:.4f}')
    axs[i].set_ylim(0, 80)

plt.tight_layout()
plt.show()



# Mediating effect under different conditions
HP = 'MP'
PCP = pd.read_excel(r'filepath\Fig4_publicimpactonCP.xlsx',sheet_name='Sheet1')
excel_file = r'filepath\fig3&SIfigS78_results.xlsx'
dfs = pd.read_excel(excel_file, sheet_name=None)

# Iterate over each sheet and output the sheet name and the first few rows of the DataFrame
for sheet_name, df in dfs.items():
    print(f"Sheet Name: {sheet_name}")
    print(df.head())  # Output the first few rows of the DataFrame, adjust as needed
    if HP in sheet_name:
        Plag = df
        PCP_filter = PCP.dropna(subset=['mark'])
        Plag_filter = Plag[Plag['Pvalue'] < 0.1]

        Plag_filter = Plag_filter[~Plag_filter['name'].str.startswith('R')]
        Plag_filter['name'] = HP + Plag_filter['name']

        PCP_filter['coef'] = PCP_filter['coef']/10 ## Change in the number of climate policies for every ten percent increase in the number of people holding the same opinion
        Plag_filter['value'] = Plag_filter['value'] ## Change in the number of migrations for every additional climate policy (in units of 1000 people)

        # Create an empty DataFrame to store the calculation results
        result_df = pd.DataFrame(columns=['typesP', 'Cong', 'PonM', 'ConP', 'multiply','group'])

        # Iterate through the 'name' column of Plag_filter
        for group in Plag_filter['group']:
            # group = 'Nettotal'
            mach_plag = Plag_filter[Plag_filter['group'] == group]
            for name in mach_plag['name']:
                # Find all matching y values
                # name = 'APTechnologicalInfrastructural'
                matching_y_values = PCP_filter[PCP_filter['y'] == name]

                for x_value in matching_y_values['x']:
                    # Calculate the product
                    # x_value = 'humancaused'
                    PonM = mach_plag.loc[mach_plag['name'] == name, 'value'].values[0]
                    ConP = matching_y_values.loc[matching_y_values['x'] == x_value, 'coef'].values[0]
                    multiply_values = PonM * ConP

                    # Create a temporary DataFrame containing the calculation results
                    temp_df = pd.DataFrame({
                        'typesP': [name],
                        'Cong': [x_value],
                        'PonM': [PonM],
                        'ConP': [ConP],
                        'multiply': [multiply_values],
                        'group': [group]
                    })

                    result_df = pd.concat([result_df, temp_df], ignore_index=True)

        # Print the final results
        df_unique = result_df.drop_duplicates(subset=['multiply'])
        # Reorder
        typesP_order = [''+HP+'TechnologicalInfrastructural', ''+HP+'Institutional', ''+HP+'BehavioralCultural', ''+HP+'NatureBased']
        group_order = ['Netdiff', 'Netsame', 'Nettotal']
        df_unique['typesP'] = pd.Categorical(df_unique['typesP'], categories=typesP_order, ordered=True)
        df_unique['group'] = pd.Categorical(df_unique['group'], categories=group_order, ordered=True)
        df_unique = df_unique.sort_values(by=['typesP', 'group'])
        # Export the results
        df_unique.to_excel(r'filepath\fig4bc&SItables10-12\result_'+sheet_name+'.xlsx', index=False)