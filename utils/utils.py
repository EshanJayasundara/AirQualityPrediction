import logging
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

global_rand_state = 66
global_map_ = {'Moderate':2, 'Good':3, 'Hazardous':0, 'Poor':1}


# Initialize logging and save to a file
def initiate_logger(loc="mlflow_api.log"):
    LOG_FILE = loc
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_FILE, filemode='w')


def load_raw_dataset(file_path: str, target: str) -> List[pd.DataFrame]:
    """
    Function to load raw datasets from a csv file

    parameters,
    file_name(str): path to the csv file without the extention(.csv)
    target(str): name of the target column
    
    returns,
    list of two pd.DataFrame objects(X, y)
    """

    data = pd.read_csv(f"{file_path}.csv")
    X = data.drop(columns=[target])
    y = data[target]

    logging.info("Successfully load the raw dataset.")

    return X, y


def fill_nulls(
        X:pd.DataFrame, 
        y:pd.Series, 
        target:str, 
        feature_null_threshold:int, 
        target_null_threshold:int
        ) -> List[pd.DataFrame]:
    """
    Function to fill null values in features and target separately

    parameters,
    X(pd.DataFrame): input features
    y(pd.Series): target
    target(str): name of the target column
    feature_null_threshold(int): between 0 and 100 
    target_null_threshold(int): between 0 and 100 
    
    returns,
    list of two pd.DataFrame objects(X, y)
    """
    null_cache = {}
    null_cache["features"] = {}
    null_cache["features"]["numerical_means"] = {}
    null_cache["features"]["categorical_modes"] = {}
    null_cache["target"] = {}
    null_cache["target"]["numerical_mean"] = {}
    null_cache["target"]["categorical_mode"] = {}

    columns = X.select_dtypes(include=['number']).columns

    logging.info("Checking null values in features, X.")
    for col in columns:
        if X[col].isna().sum() / X[col].shape[0] * 100 > feature_null_threshold:
            logging.warning(f"Feature `{col}` contain null values greater than `{feature_null_threshold}%`, drop the column.")
            X = X.drop(columns=[col])
        elif X[col].isna().sum() / X[col].shape[0] * 100 == 0:
            logging.info(f"Feature `{col}` does not contain null values.")
            pass
        else:
            logging.warning(f"Feature `{col}` does not contain null values greater than `{feature_null_threshold}%`, treating all null values with mean for numerical data.")
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode())
        if pd.api.types.is_numeric_dtype(X[col]):
            null_cache["features"]["numerical_means"][f"{col}"] = X[col].mean()
        else:
            null_cache["features"]["categorical_modes"][f"{col}"] = X[col].mode()[0]
    
    logging.info("Checking null values in target y.")
    if y.isna().sum() / y.shape[0] * 100 > target_null_threshold:
        logging.error(f"Target column ({target}) contain null values greater than `{target_null_threshold}%`.")
        raise ValueError("Target column highly contain null values.")
    elif X[col].isna().sum() / y.shape[0] * 100 == 0:
        logging.info(f"Feature `{target}` does not contain null values.")
        pass
    else:
        logging.warning(f"Feature `{target}` does not contain null values greater than `{target_null_threshold}%`, treating all null values with mean for numerical data.")
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode())
    if pd.api.types.is_numeric_dtype(y):
        null_cache["target"]["numerical_mean"][f"{target}"] = y.mean()
    else:
        null_cache["target"]["categorical_mode"][f"{target}"] = y.mode()[0]

    logging.info("Null imputation complete")

    return X, y, null_cache


def filter_outliers(
        X:pd.DataFrame, 
        y:pd.Series, 
        method:str, 
        skip:List[str]=['PM2.5', 'PM10'],
        outlier_threshold:int=5 
        ) -> pd.DataFrame:
    """
    Function to fill null values in features and target separately

    parameters,
    X(pd.DataFrame): input features
    y(pd.Series): target
    method(int): method to remove the outliers(iqr or z-score)
    outlier_threshold(int): maximum outlier percentage allowed for each feature
    skip(List[str]): skip these columns from outlier removal to reduce the data loss
    
    returns,
    pd.DataFrame object(X) containeg the outlier removed dataset
    """
    outlier_cache = {}
    outlier_cache[f"{method}"] = {}

    if method not in ["iqr", "z-score"]:
        raise ValueError("Invalid method. Choose 'iqr' or 'z-score'.")
    
    non_outliers_mask = pd.Series(True, index=X.index)          
    match method:
        case "iqr":
            logging.info(f"Using IQR method to remove outliers.")
            Q3 = X.describe().loc["75%", :]
            Q1 = X.describe().loc["25%", :]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_cache[f"{method}"]["lower_bound"] = {}
            outlier_cache[f"{method}"]["upper_bound"] = {}
            for col in X.select_dtypes("number").columns:
                if col in skip:
                    logging.info(f"Skipping outlier removal for `{col}` feature.")
                    continue
                outlier_cache[f"{method}"]["skip_columns"] = skip
                outlier_cache[f"{method}"]["lower_bound"][f"{col}"] = lower_bound[col]
                outlier_cache[f"{method}"]["upper_bound"][f"{col}"] = upper_bound[col]
                # print(X[(X[col] < lower_bound[col]) | (X[col] > upper_bound[col])][col].shape)
                col_outliers_mask = ((X[col] < lower_bound[col]) | (upper_bound[col] < X[col]))
                outlier_percentage = col_outliers_mask.sum() / X.shape[0] * 100
                # print("op:", outlier_percentage)
                logging.info(f"IQR: outlier percentage in {col} feature is {outlier_percentage}.")
                if outlier_percentage < outlier_threshold:
                    logging.info(f"Doing Outlier removal.")
                    non_outliers_mask &= ~col_outliers_mask
                else:
                    logging.error(f"Serious issue with outliers. Please manualy check `{col}` feature.")
                    raise ValueError(f"Serious issue with outliers. Please manualy check `{col}` \
                                     feature. If outliers removed significant amount of data will \
                                     be lost. If you need to skip this column doing the outlier \
                                     removal based on iqr add it to skip list.")
        case "z-score":
            z_threshold = 3 # Default Z-score threshold (adjust if needed)
            outlier_cache[f"{method}"]["skip_columns"] = skip
            logging.info(f"Using Z-score method with threshold ±{z_threshold}.")
            outlier_cache[f"{method}"]["z_threshold"] = z_threshold
            outlier_cache[f"{method}"]["mean"] = {}
            outlier_cache[f"{method}"]["std"] = {}
            for col in X.select_dtypes("number").columns:
                mean = X[col].mean()
                std = X[col].std()
                if col in skip:
                    continue
                if std == 0:
                    logging.info(f"Skipping {col} (zero standard deviation).")
                    continue
                outlier_cache[f"{method}"]["mean"][f"{col}"] = mean
                outlier_cache[f"{method}"]["std"][f"{col}"] = mean
                z_scores = (X[col] - mean) / std
                col_outliers_mask = (abs(z_scores) > z_threshold)
                outlier_percentage = col_outliers_mask.mean() * 100
                logging.info(f"Z-score: Outlier percentage in {col}: {outlier_percentage:.2f}%")
                if outlier_percentage < outlier_threshold:
                    non_outliers_mask &= ~col_outliers_mask
                else:
                    raise ValueError(
                        f"Outlier percentage ({outlier_percentage:.2f}%) in `{col}` exceeds threshold. "
                        f"Add `{col}` to the skip list if needed."
                    )
    X_filtered = X[non_outliers_mask]
    y_filtered = y[non_outliers_mask]
    
    logging.info("Outlier removing completed.")
    
    return X_filtered, y_filtered, outlier_cache


def is_categorical_vars(X:pd.DataFrame) -> bool:
    """
    Function to check if the feature variables have categorical variables

    parameters,
    X(pd.DataFrame): input features
    
    returns,
    boolean value True/False
    """
    return True if X.select_dtypes('object').columns.tolist() else False


def encode_categorical_vars(X:pd.DataFrame) -> pd.DataFrame:
    """
    Function to encode the categorical feature variables into numeric values

    parameters,
    X(pd.DataFrame): input features
    
    returns,
    pd.DataFrame containing numeric features
    """
    # You may add new parameters as you wish
    
    encode_cat_feat_cache = {} # fill this cache as needed

    if not is_categorical_vars(X=X):
        return X, encode_cat_feat_cache
    
    # Code your categorical variable handling logic here
    ### start

    ### end

    return X, encode_cat_feat_cache


def is_categorical_tar(y:pd.Series) -> bool:
    """
    Function to check if the target variable is a categorical variable

    parameters,
    y(pd.Series): target
    
    returns,
    boolean value True/False
    """
    return True if y.dtype == 'O' else False


def encode_categorical_tar(y:pd.Series, map_:Dict) -> pd.Series:
    """
    Function to encode the target variable into numeric values

    parameters,
    y(pd.Series): target
    map(Dict): logic to map categorical to numerical

    returns,
    pd.Series containing numeric target
    """
    encode_cat_tar_cache = {}

    if not is_categorical_tar(y=y):
        return y, encode_cat_tar_cache
    
    encode_cat_tar_cache["map"] = map_
    return y.map(map_), encode_cat_tar_cache


def remove_correlated_duplicates(
        X:pd.DataFrame, 
        y:pd.Series, 
        target:str, 
        correlation_threshold:float=0.9,  
        enable_heatmap:bool=False
        ) -> pd.DataFrame:
    """
    Function to remove highly correlated feature vectors based on the logic,
    If the correlation of two distinct variables is greater than `correlation_threshold`,
    remove the one with lower correlation with the target variable.

    parameters,
    X(pd.DataFrame): input features
    y(pd.Series): target
    target(str): name of the target column
    correlation_threshold(float): between 0 and 1 
    enable_heatmap(bool): True/False
    
    returns,
    pd.DataFrame object(X) with no highly correlated features
    """
    correlation_cache = {}

    logging.info("Removing highly correlated features started.")
    correlation_matrix = pd.concat([X, y], axis=1).corr().abs()
    if enable_heatmap:
        plt.title("Correlation Matrix Before Removing Features")
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            cols = correlation_matrix.drop(target, axis=1).columns
            if cols[i] == cols[j]:
                continue
            if correlation_matrix.loc[cols[i], cols[j]] > correlation_threshold:
                # print(correlation_matrix.loc[i, j])
                if correlation_matrix.loc[target, cols[i]] < correlation_matrix.loc[target, cols[j]]:
                    X = X.drop(columns=[cols[i]], axis=1)
                else:
                    X = X.drop(columns=[cols[j]], axis=1)
    correlation_matrix = pd.concat([X, y], axis=1).corr().abs()
    if enable_heatmap:
        plt.title("Correlation Matrix After Removing Features")
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()

    correlation_cache["selected_features"] = X.columns.tolist()
    logging.info("Removing highly correlated features successfully finished.")

    return X, correlation_cache


# not used
def save_preprocessed_dataset(X:pd.DataFrame, y:pd.Series, location:str) -> None:
    """
    Function to save the preprocessed dataset

    parameters,
    X(pd.DataFrame): input features
    y(pd.Series): target
    location(str): location to save the preprocessed dataset

    returns,
    None
    """
    logging.info(f"Saving preprocessed data to {location}.")

    pd.concat([X, y], axis=1).to_csv(f"{location}.csv")

    logging.info(f"Preprocessed data saved to {location}.")


def data_preprocessing(location: str, target: str, random_state: int, feature_null_threshold: int=90, target_null_threshold: int=50) -> List[pd.DataFrame]:
    """
    Function to preprocess datasets from a csv file

    parameters,
    file_name(str): path to the csv file without the extention(.csv)
    target(str): name of the target column
    random_state(int): every time the shuffle happens in a similar manner
    feature_null_threshold(int): between 0 and 100 
    target_null_threshold(int): between 0 and 100 
    
    returns,
    list of two pd.DataFrame objects(X, y)
    """
    caches = []

    X, y = load_raw_dataset(file_path=location, target=target)
    
    init_feat_cache = {}
    init_feat_cache["initial_features"] = X.columns.tolist()
    caches.append(init_feat_cache)

    X, y, null_cache = fill_nulls(X=X, y=y, target=target, feature_null_threshold=feature_null_threshold, target_null_threshold=target_null_threshold)
    print(null_cache)
    caches.append(null_cache)

    X, y, outlier_cache = filter_outliers(X=X, y=y, method='z-score', skip=['PM2.5', 'PM10'])
    print(outlier_cache)
    caches.append(outlier_cache)

    X, encode_cat_feat_cache= encode_categorical_vars(X=X)
    print(encode_cat_feat_cache)
    caches.append(encode_cat_feat_cache)

    # following thing only happens for classification
    y, encode_cat_tar_cache = encode_categorical_tar(y=y, map_=global_map_) # see the top of the notebook where the global_map_ defined
    print(encode_cat_tar_cache)
    caches.append(encode_cat_tar_cache)

    X, correlation_cache = remove_correlated_duplicates(X=X, y=y, target='Air Quality', enable_heatmap=False)
    print(correlation_cache)
    caches.append(correlation_cache)

    print(X.shape, y.shape)

    return X, y, caches


def scale_features(X_train:pd.DataFrame, X_test:pd.DataFrame, scaler:object) -> List[pd.DataFrame]:
    """
    Function to sclae the features and \
    converting the resulting np.arry back to a pd.DataFrame.

    parameters,
    scaler(object): scaler object StandardScaler/MinMaxScaler/RobustScaler from sklearn
    X_train(pd.DataFrame): input train features
    X_test(pd.DataFrame): input test features
    
    returns,
    list of two pd.DataFrame objects containing the scaled features separately for train and test sets
    """
    logging.info("Feature scaling started.")

    columns = X_train.columns
    train_index = X_train.index
    test_index = X_test.index

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train, columns=columns, index=train_index)
    X_test_df = pd.DataFrame(X_test, columns=columns, index=test_index)

    logging.info("Feature scaling finished.")

    return X_train_df, X_test_df

def plot_model_performances(
        X_train_scaled:pd.DataFrame, 
        y_train:pd.Series, 
        models:Dict[str, BaseEstimator], 
        colors: List[str], 
        random_state:int, 
        ) -> None:
    """
    Function to plot different metrics for classification model.

    parameters,
    model(List[BaseEstimator]): list of estimators from sklearn
    X_train_scaled(pd.DataFrame): input train features
    y_train(pd.Series): target vector
    colors(List[str]): colors for the graphs
    random_state(int): every time the shuffle happens in a similar manner
    
    returns,
    None
    """
    logging.info("Model comparison started.")

    X_train_, X_val_, y_train_, y_val_ = train_test_split(
                                                X_train_scaled, 
                                                y_train, test_size=0.1, 
                                                shuffle=True, 
                                                random_state=random_state
                                            )
    # feature_importances = []
    for model_name, model in models.items():
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_val_)
        report = classification_report(y_true=y_val_, y_pred=y_pred, output_dict=True)

        x = [k for k in report.keys() if k.isnumeric()]

        if len(colors) != len(x):
            logging.error(f"len(colors) == number of classes should be True. {len(colors)} != {len(x)}")
        assert len(colors) == len(x), "len(colors) == number of classes should be True."

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))  # Adjusted figure size
        fig.suptitle(f"Classification Metrics for {model_name}", fontsize=16)

        for i, metric_key in enumerate(["precision", "recall", "f1-score", "support"]):
            y = [metrics[metric_key] for key, metrics in report.items() if key.isdigit()]

            ax[i].set_title(f"{metric_key}")
            ax[i].bar(x, y, color=colors)
            ax[i].set_xlabel("Class")
            ax[i].set_ylabel(metric_key)
        
        plt.text(x=-0.2, y=-40, s="support: number of instences per class")
        plt.show()

    logging.info("Model comparison finished.")


def select_features_elbow_method(X_train_scaled:pd.DataFrame, y_train:pd.Series, best_model: BaseEstimator, plot_enable:bool=True) -> List[str]:
    """
    Function to select the features based on best performing model.

    parameters,
    model(List[BaseEstimator]): list of estimators from sklearn
    X_train_scaled(pd.DataFrame): input train features
    y_train(pd.Series): target vector
    colors(List[str]): colors for the graphs
    plto_enable(bool): True/False
    
    returns,
    list of string selected column names
    """
    # Fit the model
    best_model.fit(X_train_scaled, y_train)

    # Extract feature importance for tree-based or linear models
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).flatten()
    else:
        # Use permutation importance as a fallback
        from sklearn.inspection import permutation_importance
        result = permutation_importance(best_model, X_train_scaled, y_train, n_repeats=5, random_state=42)
        importances = result.importances_mean

    # Sort features by importance in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = X_train_scaled.columns[sorted_indices]

    # Apply elbow method to find the optimal number of features
    from kneed import KneeLocator
    kneedle = KneeLocator(
        range(1, len(sorted_importances) + 1), 
        sorted_importances, 
        curve="convex", 
        direction="decreasing",
        )
    optimal_num_features = kneedle.elbow

    # Select top features based on the elbow point
    selected_features = sorted_features[:optimal_num_features].tolist()

    # Plot the knee graph
    if plot_enable:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(sorted_importances) + 1), sorted_importances, marker='o', linestyle='-', label="Feature Importance")
        if optimal_num_features:
            plt.axvline(optimal_num_features, color='r', linestyle='--', label=f'Elbow Point ({optimal_num_features})')
            plt.scatter(optimal_num_features, sorted_importances[optimal_num_features - 1], color='red', s=100, label='Elbow')
        plt.xlabel("Number of Features")
        plt.ylabel("Feature Importance")
        plt.title("Elbow Method for Feature Selection")
        plt.legend()
        plt.show()

    return selected_features, sorted_features


def select_features_metric_method(
        X_train_scaled: pd.DataFrame, 
        y_train: pd.Series, 
        best_model: BaseEstimator, 
        random_state: int, 
        sorted_feature_names_based_on_feature_importance: List[str], 
        selector: str="accuracy",
        plot_enable: bool=True
        ) -> List[str]:
    """
    Function to select features based on evaluation matric for classification \
    given the sorted feature names in desending order based on feature importance.
    
    parameters:
    X_train_scaled(pd.DataFrame): input features 
    y_train(pd.Series): target vector
    best_model(BaseEstimator): selected best model at early step
    random_state(int): every time the shuffle happens in a similar manner
    sorted_feature_names_based_on_feature_importance(List[str]): second output of `select_features` function
    selector(str): select features based on this metric
    plot_enable(bool): enable plots True/False
    
    Note:
    `sorted_feature_names_based_on_feature_importance` array named as `sfnbfi` ->
    [column name of the most important feature, ..., column name of the least important feature]

    returns:
    List[str] containing selected feature names
    """
    logging.info(f"Feature selection using {selector}. Given the sorted feature names in desending order based on feature importance.")

    def find_maxima_coordinates(x_vec:List, y_vec:List):
        """
        Method to find the coordinate at maxima
        """
        assert len(x_vec) == len(y_vec), f"Lenght of x and y vectors should be the same.\
            But given len(x_vec)={len(x_vec)} and len(y_vec)={len(y_vec)}"
        y_max_ = -np.inf
        x_at_max_y = 0
        for x, y in zip(x_vec, y_vec): 
            if y_max_ < y:
                y_max_ = y
                x_at_max_y = x
        return (x_at_max_y, y_max_)


    X_train_, X_val_, y_train_, y_val_ = train_test_split(
                                                X_train_scaled, 
                                                y_train, test_size=0.1, 
                                                shuffle=True, 
                                                random_state=random_state
                                            )
    
    reports = {}
    sfnbfi = sorted_feature_names_based_on_feature_importance
    for i in range(1, len(sfnbfi)+1):
        best_model.fit(X_train_[sfnbfi[0: i]], y_train_)
        y_pred = best_model.predict(X_val_[sfnbfi[0: i]])
        reports[i] = classification_report(y_true=y_val_, y_pred=y_pred, output_dict=True)
    
    x_corrdinates_at_maxima = []
    
    if selector == "accuracy":
        accuracies = {}
        for i, report in reports.items():
            accuracies[i] = report['accuracy']

        acc_x = list(accuracies.keys())
        acc_y = list(accuracies.values())
        acc_maxima_coordinate = find_maxima_coordinates(acc_x, acc_y)
        # print("maximum accuracy is at corrdinate:", acc_maxima_coordinate)
        x_corrdinates_at_maxima.append(acc_maxima_coordinate[0])
        if plot_enable:
            plt.figure(figsize=(7, 4))
            plt.plot(acc_x, acc_y, linewidth=2)
            plt.ylim(0, 1)
            plt.title(f"accuracy against number of selected features based on importance")
            plt.legend([f"maximum accuracy is at corrdinate: ({acc_maxima_coordinate[0]:.2f}, {acc_maxima_coordinate[1]:.2f})"])
            plt.ylabel("accuracy")
            plt.xlabel("number of features selected")
            plt.grid(visible=True)
            plt.show(block=False)
    elif selector == "f1-score" or \
        selector == "precision" or \
            selector == "recall":
        dic = {}
        for i, report in reports.items():
            dic[i] = {key: value for key, value in report.items() if key.isnumeric()}

        classes = sorted(y_train.unique().tolist())
        # print(classes)
        if plot_enable:
            plt.figure(figsize=(7, 4))
        legend = []
        for cls in classes:
            # print(cls)
            x = []
            y = []
            n_features_selected = len(sfnbfi)
            for n_feature, metrics in dic.items():
                x.append(n_feature)
                f1_score = metrics[f"{cls}"][selector]
                y.append(f1_score) # Prefer to use this because it's a combination of precision and recall
            maxima_coordinate = find_maxima_coordinates(x, y)
            # print(f"maximum f1-score of `class {cls}` is at corrdinate:", maxima_coordinate)
            x_corrdinates_at_maxima.append(maxima_coordinate[0])
            if plot_enable:
                legend.append(f"Class {cls} maxima coordinate: ({maxima_coordinate[0]:.2f}, {maxima_coordinate[1]:.2f})")
                plt.plot(x, y, linewidth=2)
        if plot_enable:
            plt.title(f"{selector} against number of selected features based on importance")
            plt.ylim(0, 1)
            plt.legend(legend)
            plt.ylabel(selector)
            plt.xlabel("number of features selected")
            plt.grid(visible=True)
            plt.show(block=False)
    else:
        raise ValueError(f"Invalid value for selector. {selector} is not in selectors list")
    
    selected_features = sfnbfi[:max(x_corrdinates_at_maxima)]
    logging.info(f"Feature selection based on importance and {selector} completed.")
    return selected_features.tolist()
        

def preprocess_production(X_req_:List[List[float]], pickle_loc:str="caches.pkl") -> pd.DataFrame:
    import pickle
    # load the previously saved caches
    with open(pickle_loc, "rb") as f:
        caches: List[Dict] = pickle.load(f)
        f.close()

    # dont change this order 
    init_feat_cache, null_cache, outlier_cache, encode_cat_feat_cache, encode_cat_tar_cache, correlation_cache, scaler_cache, feature_selection_elbow_cache, feature_selection_metric_cache = caches

    # import pprint
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(feature_selection_metric_cache)

    # construct the dataframe
    X_req = pd.DataFrame(X_req_, columns=init_feat_cache["initial_features"])

    # fill categorical null values
    if is_categorical_vars(X=X_req):
        for col in list(null_cache["features"]["categorical_modes"].keys()):
            X_req[col] = X_req[col].fillna(null_cache["features"]["categorical_modes"][f"{col}"])

    # fill numerical null values
    for col in list(null_cache["features"]["numerical_means"].keys()):
        X_req[col] = X_req[col].fillna(null_cache["features"]["numerical_means"][f"{col}"])

    import warnings
    # detect outliers
    if 'iqr' in list(outlier_cache.keys()):
        flag = pd.Series(True, index=X_req.index)
        for col in list(outlier_cache["iqr"]["lower_bound"].keys()):
            flag &= ((X_req[col] < outlier_cache["iqr"]["lower_bound"][f"{col}"]) & (outlier_cache["iqr"]["upper_bound"][f"{col}"] < X_req[col]))
        ilocs = np.arange(0, X_req.shape[0], 1)[flag.values]
        # print(ilocs)
        if flag.sum() != 0:
            warnings.warn("Potential outliers are found in input features at following row numbers:")
            print("Potential outliers are found in input features at following row numbers:", ilocs)
            logging.warning("Potential outliers are in input features.")

    if 'z-score' in list(outlier_cache.keys()):
        flag = pd.Series(True, index=X_req.index)
        for col in list(outlier_cache["z-score"]["mean"].keys()):
            flag &= (((X_req[col] - outlier_cache["z-score"]["mean"][f"{col}"]) /  outlier_cache["z-score"]["std"][f"{col}"]).abs() > outlier_cache["z-score"]["z_threshold"])
        ilocs = np.arange(0, X_req.shape[0], 1)[flag.values]
        # print(ilocs)
        if flag.sum() != 0:
            warnings.warn("Potential outliers are found in input features at following row numbers:")
            print("Potential outliers are found in input features at following row numbers:", ilocs)
            logging.warning("Potential outliers are in input features.")

    # categorical feature encoding
    if is_categorical_vars(X=X_req):
        print(encode_cat_feat_cache)
        # code here: the logic for handle categorical features 
        ### start

        ### end

    # select features based on correlation
    X_req = X_req[correlation_cache["selected_features"]]

    X_req_scaled = scaler_cache["scaler"].transform(X_req)

    X_req_scaled_df = pd.DataFrame(X_req_scaled, index=X_req.index, columns=X_req.columns)

    # since the elbow method is not good I'vs choosed the metric method for feature selection
    selector_method = "feature_selection_metric_method" # or feature_selection_elbow_method
    match selector_method:
        case "feature_selection_metric_method":
            X_final_df = X_req_scaled_df[feature_selection_metric_cache["selected_features"]]
        case "feature_selection_elbow_method":
            X_final_df = X_req_scaled_df[feature_selection_elbow_cache["selected_features"]]

    return X_final_df