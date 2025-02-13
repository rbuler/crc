# %%
import yaml
import utils
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
                              column: str,
                              group: str) -> str:
    results = {}
    if col_type == 'cat':
        contingency_table = pd.crosstab(df[group], df[column])
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
        group1 = df[df[group] == "0"][column].astype(float)
        group2 = df[df[group] == "1"][column].astype(float)
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

def plot_confusion_matrix(subset: pd.DataFrame, t_or_n: str) -> None:
    if t_or_n == 'T':
        y_true = subset['pT']
        y_pred = subset['wmT']
        # labels = sorted(subset['pT'].unique())
        # labels = [f"T{label}" for label in labels]
        labels = ['T0', 'T1', 'T2', 'T3', 'T4', 'T4a', 'T4b']
    elif t_or_n == 'N':
        y_true = subset['pN']
        y_pred = subset['wmN']
        # labels = sorted(subset['pN'].unique())
        # labels = [f"N{label}" for label in labels]
        labels = ['N0', 'N1a', 'N1b', 'N2a', 'N2b']


    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Radiologist")
    plt.ylabel("Pathological")
    if t_or_n == 'T':
        plt.title("Tumour Staging: Radiologist vs. Pathological")
    elif t_or_n == 'N':
        plt.title("Node Staging: Radiologist vs. Pathological")
    plt.yticks(rotation=0)
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().xaxis.set_ticks_position('none')
    hatch_patterns = {
        'overstaging': "/",
        'understaging': "\\"
    }

    ax = plt.gca()
    num_labels = len(labels)

    for i in range(num_labels):
        for j in range(num_labels):
            if j > i:  # Over-staging (above diagonal)
                rect = patches.Rectangle((j, i), 1, 1,
                                            fill=False,
                                            hatch=hatch_patterns['overstaging'],
                                            edgecolor='black',
                                            linewidth=0,
                                            alpha=0.1)
                ax.add_patch(rect)
            elif j < i:  # Under-staging (below diagonal)
                rect = patches.Rectangle((j, i), 1, 1,
                                            fill=False,
                                            hatch=hatch_patterns['understaging'],
                                            edgecolor='black',
                                            linewidth=0,
                                            alpha=0.1)
                ax.add_patch(rect)
    plt.show()


def plot_staging_distribution(subset: pd.DataFrame, column: str, label_prefix: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    order = sorted(subset[column].unique())
    ax = sns.countplot(x=column, data=subset, palette='viridis', order=order)
    order = [f"{label_prefix}{label}" for label in order]
    plt.xticks(ticks=range(len(order)), labels=order)
    plt.xlabel(title)
    plt.ylabel('', rotation=0)
    plt.yticks([])
    sns.despine(top=True, bottom=True, left=True, right=True)
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().xaxis.set_ticks_position('none')
    total = len(subset)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = 100 * count / total
        ax.annotate(f'{count} ({percentage:.1f}%)', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.show()

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
    subset['T_binary'] = subset['wmT'].apply(lambda x: "1" if x != "0" else "0")


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

    subset = subset.dropna()
    for key in columns_to_analyze.keys():
        print(f"Results for {key}:", end=" ")
        if columns_to_analyze[key] == "cat":
            # print(subset[key].value_counts())
            subset.loc[:, key] = subset[key].apply(lambda x: "0" if str(x).lower() in ['x', 'tia'] else x)
        results = perform_statistical_tests(subset, columns_to_analyze[key], key, "N_binary")
        if columns_to_analyze[key] == "cat":
            print(subset[key].value_counts())
        print(f"{results}")

    plot_confusion_matrix(subset, 'T')
    plot_staging_distribution(subset, 'wmT', 'T', 'Tumour Staging (R)')
    plot_staging_distribution(subset, 'pT', 'T', 'Tumour Staging (P)')
    plot_confusion_matrix(subset, 'N')
    plot_staging_distribution(subset, 'wmN', 'N', 'Node Staging (R)')
    plot_staging_distribution(subset, 'pN', 'N', 'Node Staging (P)')

# %%