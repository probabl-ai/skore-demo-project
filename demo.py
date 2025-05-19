# %%
import re
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector


import skrub
from skrub import tabular_learner, DropCols, MinHashEncoder, TableVectorizer, AggJoiner, TableReport
from skrub import _selectors as s

import skore
from skore import EstimatorReport
from skore_remote_project.project.project import Project

# %% [markdown]
# Fetch the dataset. We will use the credit fraud dataset from skrub.

# %%
dataset = skrub.datasets.fetch_credit_fraud()
df_products = dataset.products
df_target = dataset.baskets
# %%
skrub.TableReport(df_products)
# %%
skrub.TableReport(df_target)
# %% 
products_grouped = df_products.groupby("basket_ID").agg(list)
# %%
products_flatten = []
for col in products_grouped.columns:
    cols = [f"{col}{idx}" for idx in range(24)]
    products_flatten.append(pd.DataFrame(products_grouped[col].to_list(), columns=cols))
products_flatten = pd.concat(products_flatten, axis=1)
# line below in skrub docs, but what for?
# products_flatten.insert(0, "basket_ID", products_grouped.index)
# %% 

# %%
# for the sake of this demo, we will use a small subset of the data.
products_flatten = products_flatten.sample(1000, random_state=1)
df_target = df_target.sample(1000, random_state=1)
X_train, X_test, y_train, y_test = skore.train_test_split(products_flatten, df_target["fraud_flag"], random_state=1)

# %% [markdown]
# Simpler is better.
# Let's do a simple baseline.

# %% 
# keep only the columns with the most information
def drop_empty_cols(df):
    # for each columns, check if more than 50% of the values are not null
    # if so, keep the column
    cols_to_keep = []
    for col in df.columns:
        if df[col].isnull().sum() / len(df) < 0.5:
            cols_to_keep.append(col)
    
    reduced_df = np.empty(shape=(len(df), len(cols_to_keep)), dtype=object)
    for i, col in enumerate(cols_to_keep):
        for j, val in enumerate(df[col]):
            reduced_df[j, i] = val
    return reduced_df

feature_selection_func = FunctionTransformer(drop_empty_cols)

feature_selection = ColumnTransformer(
    transformers=[
        ('extract_year', feature_selection_func, products_flatten.columns),
    ],
    remainder='drop'
)

# %%
baseline = tabular_learner('classification')
baseline.steps.insert(0, ('feature_selection', feature_selection))
baseline_report = EstimatorReport(baseline, X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test)
baseline_report.help()

# %%
# create project
project = Project(name="project demo", tenant="Probabl")
# %%
project.put("baseline", baseline_report)

# %% 
baseline_report.metrics.report_metrics()

# %% [markdown]
# A lot of information is lost in the feature selection step.
# Let's try to play a bit with the columns and keep some.
# %% 
# %% [markdown]
# The column with most information is `model`. 
# It seems possible to extract the year of the device. 
# Rationale: we would expect that the newer the device, the more likely it is to be used for fraud.
# %%
def extract_year_regex(x_str):
    if type(x_str) == str:
        extracted = re.findall(r"(20\d{2})", x_str)
        if len(extracted) == 1:
            # if several years are found, it might mean that the regex is uncorrect - let's not use it
            return extracted[0]
    return None

def identity_and_extract_mean_year(models):
    # construct object dtype array with two columns
    features = np.empty(shape=(len(models), len(models.columns)+1), dtype=object)
    for i, item in enumerate(models.values):
        print(item)
        features[i, 0] = np.nanmean(np.array([extract_year_regex(x) for x in item]))
    for j, col in enumerate(models.columns):
        for i, val in enumerate(models[col]):
            features[i, j+1] = val
    return features

# %% 
# number of items in the basket
def count_items(items):
    features = np.empty(shape=(len(items), len(items.columns)+1), dtype=object)
    for i, item in enumerate(items.values):
        features[i, 0] = np.count_nonzero(item != None)
    for j, col in enumerate(items.columns):
        for i, val in enumerate(items[col]):
            features[i, j+1] = val
    return features

# %%
# Create a ColumnTransformer to apply the extract_year function to the 'model' column
preprocessor = ColumnTransformer(
    transformers=[
        #('extract_year', FunctionTransformer(identity_and_extract_mean_year), make_column_selector(pattern="model.{1,2}")),
        ('count_items', FunctionTransformer(count_items), make_column_selector(pattern="item.{1,2}")),
    ],
    remainder='passthrough'
)

# %%
baseline_2 = tabular_learner("classification")
baseline_2.steps.insert(0, ('preprocessor', preprocessor))
baseline_report_2 = EstimatorReport(baseline_2, X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test)
# %%
project.put("baseline_2", baseline_report_2)
# %%
baseline_report_2.metrics.report_metrics()
# %%
comp = skore.ComparisonReport([baseline_report, baseline_report_2])
comp.help()
# %%
comp.metrics.report_metrics(pos_label = 1)
# %%


# %% [markdown]
# DEMO PART 2 - after superior review

# %%
from sklearn.model_selection import train_test_split
X, y = df_target[["ID"]], df_target["fraud_flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)
X_train.shape, y_train.shape
# %%

vectorizer = TableVectorizer(
    high_cardinality=MinHashEncoder(),  # encode ["item", "model"]
    specific_transformers=[
        (OrdinalEncoder(), ["make", "goods_code"]),
    ],
)
products_transformed = vectorizer.fit_transform(df_products)
TableReport(products_transformed)

# %% 
# Skrub selectors allow us to select columns using regexes, which reduces
# the boilerplate.
minhash_cols_query = s.glob("item*") | s.glob("model*")
minhash_cols = s.select(products_transformed, minhash_cols_query).columns

agg_joiner = AggJoiner(
    aux_table=products_transformed,
    aux_key="basket_ID",
    main_key="ID",
    cols=minhash_cols,
    operations=["min"],
)
baskets_products = agg_joiner.fit_transform(df_target)
TableReport(baskets_products)

# %%
agg_joiner = make_pipeline(
    AggJoiner(
        aux_table=products_transformed,
        aux_key="basket_ID",
        main_key="ID",
        cols=minhash_cols,
        operations=["min"],
    ),
    AggJoiner(
        aux_table=products_transformed,
        aux_key="basket_ID",
        main_key="ID",
        cols=["make", "goods_code", "cash_price", "Nbr_of_prod_purchas"],
        operations=["sum", "mean", "std", "min", "max"],
    ),
    DropCols(["ID"]),
    HistGradientBoostingClassifier(),
)


# %% 
report_agg_joiner = EstimatorReport( 
    agg_joiner,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %% 
project.put("agg_joiner", report_agg_joiner)

# %% 
report_agg_joiner.metrics.report_metrics()

