# %%
import skore
import skrub
from skrub import tabular_learner
from skore import EstimatorReport
from skore_remote_project.project.project import Project
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import re
import numpy as np

# %% [markdown]
# Fetch the dataset. We will use the credit fraud dataset from skrub.

# %%
dataset = skrub.datasets.fetch_credit_fraud()
df_target = dataset.baskets
df_products = dataset.products
# merge the two dataframes on the basket_id column
df = df_target.merge(df_products, left_on="ID", right_on="basket_ID").drop(columns=["basket_ID", "ID"])

# %% [markdown]
# Let's first have a quick look at the data inside. 
# %% 
skrub.TableReport(df)

# %%
# for the sake of this demo, we will use a small subset of the data.
X = df.drop(columns=["fraud_flag"])
y = df["fraud_flag"]
X = X.sample(1000, random_state=1)
y = y.sample(1000, random_state=1)
X_train, X_test, y_train, y_test = skore.train_test_split(X, y, random_state=1)


# %% [markdown]
# Simpler is better.
# Let's do a simple baseline.

# %%
baseline = tabular_learner('classification')
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

def identity_and_extract_year(models):
    return [{"year": extract_year_regex(model), "model": model} for model in models]

# %%
from sklearn.preprocessing import FunctionTransformer

# Define a FunctionTransformer for the extract_year function
extract_year_transformer = FunctionTransformer(identity_and_extract_year)

# Create a ColumnTransformer to apply the extract_year function to the 'model' column
preprocessor = ColumnTransformer(
    transformers=[
        ('extract_year', extract_year_transformer, 'model'),
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
