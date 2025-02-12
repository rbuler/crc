# %%
import yaml
import utils
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dataset import CRCDataset
from sklearn.metrics import confusion_matrix
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

np.random.seed(config['seed'])


def perform_statistical_tests(df: pd.DataFrame,
                              col_type: str,
                              column: str) -> str:
    results = {}
    if col_type == 'cat':
        contingency_table = pd.crosstab(df['N_binary'], df[column])
        contingency_table_entries = contingency_table.values
        contingency_table_entries_condition = contingency_table_entries > 5
        contingency_table_entries_condition = contingency_table_entries_condition.sum()
        # print(contingency_table_entries_condition)
        if contingency_table_entries_condition < contingency_table.size:
            _, p_value = stats.fisher_exact(contingency_table)
        else:
            _, p_value, _, _ = stats.chi2_contingency(contingency_table)
        proportions = contingency_table.apply(lambda r: r/r.sum(), axis=1)
        results['proportions'] = f"N-negative: {contingency_table.iloc[0, 1]} ({proportions.iloc[0, 1]*100:.2f}%), N-positive: {contingency_table.iloc[1, 1]} ({proportions.iloc[1, 1]*100:.2f}%) for {contingency_table.columns.values[1]}"
    elif col_type == 'con':
        group1 = df[df['N_binary'] == "0"][column].astype(float)
        group2 = df[df['N_binary'] == "1"][column].astype(float)
        _, p_normal1 = stats.shapiro(group1)
        _, p_normal2 = stats.shapiro(group2)
        if p_normal1 > 0.05 and p_normal2 > 0.05:
            _, p_var = stats.levene(group1, group2)
            if p_var > 0.05:
                _, p_value = stats.ttest_ind(group1, group2)
            else:
                _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        else:
            _, p_value = stats.mannwhitneyu(group1, group2)
        mean_sd_group1 = f"{group1.mean():.2f} ({group1.std():.2f})"
        mean_sd_group2 = f"{group2.mean():.2f} ({group2.std():.2f})"
        results['mean_sd'] = f"N-negative: {mean_sd_group1}, N-positive: {mean_sd_group2}"
    results['p_value'] = p_value
    return results



if __name__ == '__main__':

    columns_to_analyze = {
        "Płeć": "cat", 
        "Wiek w momencie dx": "con", 
        "Masa ciała przed rozpoczęciu leczenia": "con", 
        "Wzrost": "con", 
        "BMI": "con", 
        "Aspiryna (czy przyjmuje) tak/nie": "cat", # 2 cat and x
        # "Palący": "cat", # 3 categories
        "Cukrzyca (ta/nie)": "cat", 
        "OSAS": "cat", 
        "Migotanie przedsionków (tak/nie)": "cat", 
        "Choroba niedokrwienna serca (tak/nie)": "cat",
        "Nadciśnienie tętnicze (tak/nie)": "cat", 
        "przebyty zawał (tak/nie)": "cat", 
        "przebyty udar (tak/nie)": "cat", 
        "stan po interwencji na naczyniach wieńcowych (tak/nie)": "cat",
        "stan po pomostowaniu naczyń wieńcowych (tak/nie)": "cat",
        "IBD": "cat", # 2 cat with and x 
        "Sterydy (czy przyjmuje) (tak/nie) (obecnie lub w ciagu ostatnich 3 miesięcy)": "cat", # 2 cat and x
        "Immunosupresja (czy przyjmuje) (tak/nie) (obecnie lub w ciagu ostatnich 3 miesięcy)" : "cat" # 2 cat and x
    }    
    root = config['dir']['root']
    clinical_data = config['dir']['clinical_data']
    dataset = CRCDataset(root,
                         clinical_data=clinical_data,
                         config=config,
                         transform=None,
                         save_new_masks=False)
    # ------------------------
    
    selected_classes = ['lymph_node_positive', 'lymph_node_negative']
    df = dataset.radiomic_features[dataset.radiomic_features['class_name'].isin(selected_classes)]
    df.iloc[:, df.columns.get_loc('patient_id')] = df['patient_id'].astype(int)
    df = df.sort_values(by='patient_id').reset_index(drop=True)
    
    counts_per_patient = df.groupby('patient_id')['class_name'].value_counts()

    # Iterate through the grouped data and print the results
    for patient_id, class_counts in counts_per_patient.groupby(level=0):
        counts_str = ", ".join([f"{class_name}: {count}" for class_name, count in class_counts.items()])
        for class_name, count in class_counts.items():
            dataset.clinical_data.loc[dataset.clinical_data['Nr pacjenta'] == patient_id, class_name[1]] = count
    dataset.clinical_data[
        ['lymph_node_positive', 'lymph_node_negative']] = dataset.clinical_data[
            ['lymph_node_positive', 'lymph_node_negative']].fillna(0).astype(int)
    dataset.update_clinical_data()

    # columns_to_select = ["Nr pacjenta", "wmN", "pN", "lymph_node_positive", "lymph_node_negative", "wmNlymph_node_positive_overnoding", "pNlymph_node_positive_overnoding",
    #                      "Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane",
    #                      "wmNLiczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane_overnoding"]
    # subset = dataset.clinical_data[columns_to_select]


    # columns_to_select = ["Nr pacjenta", "wmN"]
    # subset = dataset.clinical_data[columns_to_select]
    subset = dataset.clinical_data
    subset.rename(columns={"Nr pacjenta": "patient_id"}, inplace=True)
    subset["N_binary"] = subset["wmN"].apply(lambda x: "1" if x != "0" else "0")
    # subset["T_binary"] = subset["wmT"].apply(lambda x: 1 if x != "0" else 0)


    # select only patients that have already have images
    # ids = []
    # for i in range(len(dataset)):
    #     ids.append(int(dataset.get_patient_id(i)))
    # # bad quality/invalid annotations
    # temp_to_drop = [140, 139, 138, 136, 132, 129, 128, 123, 120, 115, 113, 110, 108, 107, 102, 101, 99, 98,
    #                 96, 88, 86, 75, 70, 61, 49, 48, 37, 26, 21, 8, 3, 2]
    
    # ids = list(set(ids) - set(temp_to_drop))
    # subset = subset[subset['patient_id'].isin(ids)]

    # sorted by patient_id
    # new_df = subset.merge(df, how='inner', on='patient_id')
#TODO wmN 0 vs other  ALBO dla T
    
    # remove x
    subset = subset.dropna()
    for key in columns_to_analyze.keys():
        print(f"Results for {key}:", end=" ")
        if columns_to_analyze[key] == "cat":
            # print(subset[key].value_counts())
            subset.loc[:, key] = subset[key].apply(lambda x: "0" if str(x).lower() in ['x', 'tia'] else x)
        results = perform_statistical_tests(subset, columns_to_analyze[key], key)
        if columns_to_analyze[key] == "cat":
            print(subset[key].value_counts())
        print(f"{results}")

    y_true = subset['pN']
    y_pred = subset['wmN']

    labels = sorted(subset['pN'].unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Radiologist Diagnosis")
    plt.ylabel("Pathological Diagnosis")
    plt.title("Radiologist Staging vs. Pathological Staging")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='wmT', data=subset, palette='viridis')
    plt.title('Distribution of wmT')
    plt.xlabel('wmT')
    plt.ylabel('Count')
    wmT_order = sorted(subset['wmT'].unique())
    plt.xticks(ticks=range(len(wmT_order)), labels=wmT_order)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='wmN', data=subset, palette='viridis')
    plt.title('Distribution of wmN')
    plt.xlabel('wmN')
    plt.ylabel('Count')
    wmN_order = sorted(subset['wmN'].unique())
    plt.xticks(ticks=range(len(wmN_order)), labels=wmN_order)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
