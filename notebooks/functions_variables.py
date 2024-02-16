import os
import pandas as pd
import warnings
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split


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


class TagsEncoder(TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Determine tags to include based on threshold
        self.tags_to_include = []
        tag_counts = {}
        for sublist in X["tags"]:
            if isinstance(sublist, list):
                for tag in sublist:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        total_count = len(X["tags"])
        for tag, count in tag_counts.items():
            if count / total_count >= self.threshold:
                self.tags_to_include.append(tag)
        return self

    def transform(self, X):
        # Create a DataFrame for tags with zeros
        tag_df = pd.DataFrame(0, index=X.index, columns=self.tags_to_include)

        # Fill in tags
        for tag in self.tags_to_include:
            tag_df[tag] = X["tags"].apply(
                lambda x: 1 if isinstance(x, list) and tag in x else 0
            )

        # Concatenate tag_df and original feature set
        X_encoded = pd.concat([X, tag_df], axis=1)

        # Drop the original 'tags' column
        X_encoded.drop("tags", axis=1, inplace=True)

        return X_encoded


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


class TrainTestSplitter:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform train-test split
        train_df, test_df = train_test_split(X, test_size=0.2, random_state=42)
        return train_df, test_df
