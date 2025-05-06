import yaml
import utils
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from network import MILNetwork, train_net, test_net
from reduce_dim_features import icc_select_reproducible
from reduce_dim_features import select_best_from_clusters
from dataset import CRCDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utils import generate_mil_bags, summarize_bags


logger = logging.getLogger(__name__)
logger_radiomics = logging.getLogger("radiomics")
logging.basicConfig(level=logging.ERROR)


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# %%
if __name__ == '__main__':
    
    dataset = CRCDataset(root_dir=config['dir']['root'],
                         clinical_data_dir=config['dir']['clinical_data'],
                         nii_dir=config['dir']['nii_images'],
                         dcm_dir=config['dir']['dcm_images'],
                         config=config,
                         transform=None,
                         save_new_masks=False)
    
    selected_classes = ['lymph_node_positive', 'lymph_node_negative']
    
    df = dataset.radiomic_features[dataset.radiomic_features['class_name'].isin(selected_classes)]
    df['patient_id'] = df['patient_id'].astype(int)
    df = df.sort_values(by='patient_id').reset_index(drop=True)
    
    counts_per_patient = df.groupby('patient_id')['class_name'].value_counts()
    
    for patient_id, class_counts in counts_per_patient.groupby(level=0):
        counts_str = ", ".join([f"{class_name}: {count}" for class_name, count in class_counts.items()])
        for class_name, count in class_counts.items():
            dataset.clinical_data.loc[dataset.clinical_data['patient_id'] == patient_id, class_name[1]] = count
    
    dataset.clinical_data[selected_classes] = dataset.clinical_data[selected_classes].fillna(0).astype(int)
    # dataset.update_clinical_data()

    # columns_to_select = ["Nr pacjenta", "wmN", "pN", "lymph_node_positive", "lymph_node_negative", "wmNlymph_node_positive_overnoding", "pNlymph_node_positive_overnoding",
    #                      "Liczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane",
    #                      "wmNLiczba zaznaczonych ww chłonnych, 0- zaznaczone ale niepodejrzane_overnoding"]
    columns_to_select = ["patient_id", "wmN"]
    subset = dataset.clinical_data[columns_to_select]
    # subset.rename(columns={"Nr pacjenta": "patient_id"}, inplace=True)

    # select only patients that have already have images
    ids = []
    for i in range(len(dataset)):
        ids.append(int(dataset.get_patient_id(i)))
    # bad quality/invalid annotations
    # temp_to_drop = [140, 139, 138, 136, 132, 129, 128, 123, 120, 115, 113, 110, 108, 107, 102, 101, 99, 98,
    #                 96, 88, 86, 75, 70, 61, 49, 48, 37, 26, 21, 8, 3, 2]
    
    # ids = list(set(ids) - set(temp_to_drop))
    subset = subset[subset['patient_id'].isin(ids)]

    # sorted by patient_id
    new_df = subset.merge(df, how='inner', on='patient_id')
    
    binary_labels = new_df[new_df.columns[3]].map({selected_classes[0]: 0, selected_classes[1]: 1})
    features = new_df[new_df.columns[5:]]

    # normalize features
    # TODO 
    # normalize feature using only training set statistics
    features = (features - features.mean()) / features.std()
    #
    #
    # exclusion of nonreproducible features (in case of multiple bin widths)
    if config['radiomics']['multiple_binWidth']['if_multi']:
        bin_widths = config['radiomics']['multiple_binWidth']['binWidths']
        selected_features_icc = icc_select_reproducible(features=features,
                                                        labels=binary_labels,
                                                        threshold=0.75,
                                                        bin_widths=bin_widths)
        reproducible_features = features[selected_features_icc]
    else:
        reproducible_features = features
    logger.info(f"{reproducible_features.shape=}")

    # selection of the most relevant features using LassoCV
    features_prior_to_selection = reproducible_features.loc[:, reproducible_features.nunique() > 1]
    logger.info(f"Removing columns with uninformative values (single unique value):\n\t{features_prior_to_selection.shape=}")
    features_prior_to_selection = features_prior_to_selection.T.drop_duplicates().T
    logger.info(f"Removing duplicate columns:\n\t{features_prior_to_selection.shape=}")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_prior_to_selection)
    scaled_df = pd.DataFrame(features_scaled, columns=features_prior_to_selection.columns)
    lasso_cv = linear_model.LassoCV(cv=5, random_state=config['seed'])
    lasso_cv.fit(features_scaled, binary_labels)
    best_alpha = lasso_cv.alpha_
    logger.info(f"Best alpha: {best_alpha}")
    lasso_coefficients_cv = lasso_cv.coef_
    selected_features_cv_names = features_prior_to_selection.columns[lasso_coefficients_cv != 0]
    selected_features_df = scaled_df[selected_features_cv_names]

    # selection of most representative features for each cluster
    features_df = select_best_from_clusters(features=selected_features_df,
                                   n_clusters=5,
                                #    num_features=selected_features_df.shape[1],  # <---------------------
                                   num_features=20,  # <---------------------
                                   random_state=config['seed'])
    logger.info(f"{features_df.shape=}")
    features = torch.tensor(features_df.values, dtype=torch.float32)
    binary_labels = torch.tensor(binary_labels.values, dtype=torch.long)

    # %%


    X = features.detach().cpu().numpy()
    y = binary_labels.detach().cpu().numpy()

    classifiers = {
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(penalty='l2', solver='liblinear', class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM (Linear)": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = []

    for name, clf in classifiers.items():
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_val)

            acc = accuracy_score(y_val, y_pred)
            roc = roc_auc_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred)

            results.append({
                "Model": name,
                "Fold": fold,
                "Accuracy": acc,
                "ROC AUC": roc,
                "F1 Score": f1
            })

            print(f"\n{name} (Fold {fold}):")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  ROC AUC:  {roc:.3f}")
            print(f"  F1 Score: {f1:.3f}")

    df = pd.DataFrame(results)

    model_name_map = {
        "MLP": "MLP",
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "SVM (Linear)": "SVM-L",
        "SVM (RBF)": "SVM-R",
        "k-NN": "kNN",
    }
    df["Model"] = df["Model"].map(model_name_map)

    sns.set(style="white", context="talk", font_scale=0.9)
    custom_palette = sns.color_palette("muted")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    metrics = ["Accuracy", "ROC AUC", "F1 Score"]
    titles = ["Accuracy", "ROC AUC", "F1 Score"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.boxplot(
            data=df,
            x="Model",
            y=metric,
            ax=axes[i],
            palette=custom_palette,
            linewidth=1,
            fliersize=2,
        )
        axes[i].set_title(title, fontsize=16, weight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis='x', rotation=30)
        sns.despine(ax=axes[i], top=True, right=True, left=False, bottom=False)
    # for ax in axes:
    #     ax.tick_params(axis='x', rotation=45)
    for ax in axes:
        ax.set_ylim(0.5, 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


    # %%
    # bags = generate_mil_bags(new_df, patient_col='patient_id', features=features, instance_label_col='class_name', bag_label_col='wmN')


    # bag_ids = [bag['patient_id'] for bag in bags]
    # bag_labels = [bag['bag_label'].item() for bag in bags]
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # all_histories = []
    # all_metrics = []
    # all_conf_matrices = []

    # for train_index, test_ids in skf.split(bag_ids, bag_labels):
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        
    #     train_ids = np.array(bag_ids)[train_index]
    #     train_labels = np.array(bag_labels)[train_index]
    #     _, valid_ids = next(sss.split(train_ids, train_labels))
    #     valid_ids = train_ids[valid_ids]
    #     train_ids = np.setdiff1d(train_ids, valid_ids)


    #     train_bags = [bag for bag in bags if bag['patient_id'] in train_ids]
    #     valid_bags = [bag for bag in bags if bag['patient_id'] in valid_ids]
    #     test_bags = [bag for bag in bags if bag['patient_id'] in test_ids]

    #     train_positive, train_negative = summarize_bags(train_bags)
    #     valid_positive, valid_negative = summarize_bags(valid_bags)
    #     test_positive, test_negative = summarize_bags(test_bags)
        
    #     def collate_fn(batch):
    #         return batch

    #     train_loader = DataLoader(train_bags, batch_size=1, shuffle=True, collate_fn=collate_fn)
    #     valid_loader = DataLoader(valid_bags, batch_size=1, shuffle=False, collate_fn=collate_fn)
    #     test_loader = DataLoader(test_bags, batch_size=1, shuffle=False, collate_fn=collate_fn)

    #     print(f"Training set: {train_positive} positive bags, {train_negative} negative bags")
    #     print(f"Validation set: {valid_positive} positive bags, {valid_negative} negative bags")
    #     print(f"Test set: {test_positive} positive bags, {test_negative} negative bags")

    #     print(f"Number of training batches: {len(train_loader)}")
    #     print(f"Number of validation batches: {len(valid_loader)}")
    #     print(f"Number of test batches: {len(test_loader)}")

    #     model = MILNetwork(input_dim=features.size(1), hidden_dim=64)
    #     criterion = nn.BCELoss()
    #     # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #     model, history = train_net(model=model,
    #                             criterion=criterion,
    #                             optimizer=optimizer,
    #                             train_loader=train_loader,
    #                             valid_loader=valid_loader,
    #                             num_epochs=4000,
    #                             patience=5)
        
    #     all_histories.append(history)
    #     metrics, conf_matrix = test_net(model, test_loader)
    #     all_metrics.append(metrics)
    #     all_conf_matrices.append(conf_matrix)
    #     print()

    # # Aggregate histories, metrics, and confusion matrices
    # aggregated_history = {
    #     'epoch': [],
    #     'train_acc': [],
    #     'val_acc': [],
    #     'train_loss': [],
    #     'val_loss': []
    # }
    # for key in aggregated_history.keys():
    #     for history in all_histories:
    #         aggregated_history[key].extend(history[key])

    # aggregated_metrics = {
    #     'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
    #     'precision': np.mean([m['precision'] for m in all_metrics]),
    #     'recall': np.mean([m['recall'] for m in all_metrics]),
    #     'f1_score': np.mean([m['f1_score'] for m in all_metrics])
    # }

    # aggregated_conf_matrix = np.sum(all_conf_matrices, axis=0)

    # # Plotting aggregated history
    # sns.set_style(style="whitegrid")
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # sns.lineplot(x=aggregated_history["epoch"], y=aggregated_history["train_acc"], label="Train Accuracy", ax=axes[0])
    # sns.lineplot(x=aggregated_history["epoch"], y=aggregated_history["val_acc"], label="Validation Accuracy", ax=axes[0])
    # axes[0].set_title("Accuracy Comparison")
    # axes[0].set_xlabel("Epoch")
    # axes[0].set_ylabel("Accuracy")
    # sns.lineplot(x=aggregated_history["epoch"], y=aggregated_history["train_loss"], label="Train Loss", ax=axes[1])
    # sns.lineplot(x=aggregated_history["epoch"], y=aggregated_history["val_loss"], label="Validation Loss", ax=axes[1])
    # axes[1].set_title("Loss Comparison")
    # axes[1].set_xlabel("Epoch")
    # axes[1].set_ylabel("Loss")
    # plt.tight_layout()
    # plt.show()

    # # Plotting aggregated confusion matrix
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(aggregated_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
    #             xticklabels=["Negative", "Positive"], 
    #             yticklabels=["Negative", "Positive"])
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Aggregated Confusion Matrix")
    # plt.show()

# %%
