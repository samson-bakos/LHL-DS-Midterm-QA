import os
import pandas as pd
import warnings
from sklearn.base import TransformerMixin, BaseEstimator
import pickle


def encode_tags(df):
    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low
    counts of tags to keep cardinality to a minimum.

    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag

    return df


def get_dataframe(directory):
    df = pd.DataFrame()
    json_files = [file for file in os.listdir(directory) if file.endswith(".json")]
    for file in json_files:
        file_path = os.path.join(directory, file)
        json_subfile = pd.read_json(file_path)
        json_subdf = pd.json_normalize(json_subfile["data"]["results"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.concat([df, json_subdf], ignore_index=True)
    return df


class ColumnSelector(TransformerMixin):
    def __init__(self, column_list):
        self.column_list = column_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[col for col in self.column_list if col in X.columns]]


class DropMissingValues(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            X = X[~X[col].isna()]
        return X


class FillMissingValues(TransformerMixin):
    def __init__(self, fill_values_dict):
        self.fill_values_dict = fill_values_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col, fill_value in self.fill_values_dict.items():
            X[col] = X[col].fillna(fill_value)
        return X


class TypeTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Replace 'condo' with 'condos' in 'description.type' column
        X["description.type"] = X["description.type"].replace("condo", "condos")
        # Perform one-hot encoding for 'description.type' and 'description.sub_type' columns
        X = pd.get_dummies(X, columns=["description.type", "description.sub_type"])
        return X


class MergeAndImputeTransformer(TransformerMixin):
    def __init__(self, input_df_path):
        self.input_df_path = input_df_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Read the second dataframe from the specified directory
        input_df = pd.read_csv(self.input_df_path)

        # Merge the dataframes
        merged_df = X.merge(input_df, how="left", on="location.address.city")

        # Impute missing values with the mean
        merged_df["city_mean_sold_price"].fillna(
            merged_df["city_mean_sold_price"].mean(), inplace=True
        )

        # Drop the 'location.address.city' column
        merged_df.drop(columns=["location.address.city"], inplace=True)

        return merged_df


class TagsEncoder(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No need to collect unique tags in fit
        return self

    def transform(self, X):
        # Create an empty list to collect all tags
        all_tags = []

        # Iterate over each sublist in the 'tags' column
        for sublist in X["tags"]:
            if isinstance(sublist, list):
                all_tags.extend(sublist)

        # Create DataFrame of binary columns for each unique tag
        tag_df = pd.get_dummies(
            pd.DataFrame(all_tags, columns=["tags"]),
            columns=["tags"],
            prefix="",
            prefix_sep="",
        )

        # Concatenate tag_df and original feature set
        X_encoded = pd.concat([X.drop("tags", axis=1), tag_df], axis=1)

        return X_encoded


class PretrainedMinMaxScale(TransformerMixin):
    def __init__(self, scaler_path):
        self.scaler_path = scaler_path
        self.scaler = None

    def fit(self, X, y=None):
        # Load the scaler from the pickle file
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        return self

    def transform(self, X):
        # Transform the current dataframe using the loaded scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=X.columns, index=X.index
        )
        return X_scaled


class PredictionsFromModel(TransformerMixin, BaseEstimator):
    def __init__(self, model_path):
        self.model_path = model_path

    def fit(self, X, y=None):
        # No fitting necessary
        return self

    def transform(self, X):
        # Load the pickled model
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        # Make predictions
        predictions = model.predict(X)

        return X, predictions
