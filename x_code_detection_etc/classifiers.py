import yaml
import utils
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
parser = utils.get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']

class Classifier():
    def __init__(self, X, y, patient_ids, classifier_name=None):
        self.X = X
        self.y = y
        self.patient_ids = patient_ids
        self.clasifier_name = classifier_name

        self.classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
            "SVM": SVC(kernel='rbf', probability=True, random_state=seed),  # RBF kernel is commonly used for radiomics
            "Logistic Regression": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=seed),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=seed)
        }

        if self.clasifier_name is not None:
            self.classifier = self.classifiers[self.clasifier_name]
        else:
            raise ValueError("Please specify a classifier from the following list: Random Forest, SVM, "
                             +"Logistic Regression, k-NN, Naive Bayes, Gradient Boosting, XGBoost")


        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.patient_ids)
        self.X_train, self.X_test = self.standardize_features(X_train=self.X_train, X_test=self.X_test)


    def split_data(self, patient_ids):
        unique_ids = list(set(patient_ids))
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=seed)
        
        train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_ids]
        test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_ids]
        
        X_train = self.X[train_indices]
        X_test = self.X[test_indices]
        y_train = self.y[train_indices]
        y_test = self.y[test_indices]
        
        return X_train, X_test, y_train, y_test

    def standardize_features(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.X_train)
        X_test = scaler.transform(self.X_test)
        return X_train, X_test

    def train_classifier(self):
        clf = self.classifier
        print(f"Training {self.clasifier_name}...")
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"{self.clasifier_name} Accuracy: {acc:.4f}")
        print(classification_report(self.y_test, y_pred))
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
# %%
