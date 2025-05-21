# %% [markdown]
# # Welcome to the demo of Skore!
#
# Let's start by fetching the dataset. We will use the census dataset from openml.
#
# It's a binary classification problem, where the target is whether a person earns more than 50K a year.
#
# https://www.openml.org/search?type=data&sort=runs&id=1590&status=active

# %%
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

# %% [markdown]
# Let's take a look at the data
# in real life, we would do a lot more data exploration.

X.info()

# %%
y.value_counts()

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# %%
import pandas as pd

pd.Series(y_encoded).value_counts()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1)

# %% [markdown]
# Simpler is better.
#
# Let's do a simple baseline.

# %%
from skrub import tabular_learner

baseline = tabular_learner("classification")
baseline

# %%
from skore import EstimatorReport

baseline_report = EstimatorReport(
    baseline,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
baseline_report.help()

# %%
baseline_report.metrics.report_metrics()

# %% [markdown]
# Let's go a bit further in that baseline by optimizing the hyperparameters.

# %%
from sklearn.model_selection import GridSearchCV

tuned_baseline = GridSearchCV(
    estimator=baseline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.01, 0.1, 0.2],
        "histgradientboostingclassifier__max_depth": [1, 3, 5],
        "histgradientboostingclassifier__max_leaf_nodes": [30, 60, 90],
    },
    cv=5,
    n_jobs=-1,
    refit=True,
    scoring="neg_log_loss",
)
tuned_baseline

# %%
tuned_baseline_report = EstimatorReport(
    tuned_baseline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
# %%
tuned_baseline_report.metrics.report_metrics()

# %%
from skore import ComparisonReport

comp = ComparisonReport(
    {"Baseline Model": baseline_report, "Tuned model": tuned_baseline_report}
)
comp.help()

# %%
comp.metrics.report_metrics(pos_label=1, indicator_favorability=True)

# %%
# init for notebook execution
name = "demo"
tenant = "Probabl"

# %% [markdown]
# cell to be executed only when running the script
import argparse

# Parsing arguments when used as a script
parser = argparse.ArgumentParser()
parser.add_argument('--tenant', action="store", dest='tenant', default="Probabl")
parser.add_argument('--name', action="store", dest='name', default="demo")
args = parser.parse_args()
tenant = args.tenant
name = args.name

# %%
# create or connect to project
from skore import Project

project = Project(f"hub://{tenant}/{name}")
# %%
project.put("baseline", baseline_report)
project.put("tuned_baseline", tuned_baseline_report)

# %% [markdown]
# # DEMO PART 2 - after supervisor review
#
# Their request: even more simple baselines: dummy classifier, and a linear model.

# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="prior")
dummy_report = EstimatorReport(
    dummy,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
dummy_report.help()
# %%
dummy_report.metrics.report_metrics()

# %%
project.put("dummy", dummy_report)

# %%
from sklearn.linear_model import LogisticRegression

logistic_report = EstimatorReport(
    tabular_learner(LogisticRegression()),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
logistic_report.help()

# %%
logistic_report.metrics.report_metrics()

# %%
project.put("logistic", logistic_report)


# %% [markdown]
# # DEMO PART 3 - after business stakeholder review
# Their request: why is the feature `sex` not important, while intuitevely it should be?

# %%
# It is possible to fetch a specific report by id from the project.
# The id is available in skore-hub interface, along with other metadata. 
# chosen_report = project.reports.get(id)
chosen_report = logistic_report

# %%
from skrub import TableReport

TableReport(chosen_report.X_train)
