# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

# %% [markdown]
# Fetch the dataset. We will use the census dataset from openml.
# It's a binary classification problem, where the target is whether a person earns more than 50K a year.
# https://www.openml.org/search?type=data&sort=runs&id=1590&status=active

# %%
from sklearn.datasets import fetch_openml

X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
# %%
from skrub import TableReport

TableReport(X)
# %%
y.value_counts()

# %%
y = 1 * (y == ">50K")

# %%
import skore

X_train, X_test, y_train, y_test = skore.train_test_split(X, y, random_state=1)

# %% [markdown]
# Simpler is better.
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

# %%
# create or connect to project
from skore_hub_project.project.project import Project

project = Project(name="project demo - census", tenant="Probabl")
# %%
project.put("baseline", baseline_report)

# %% [markdown]
# Let's go a bit further in that baseline by optimizing the parameters.

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
    tuned_baseline, 
    X_train=X_train, 
    y_train=y_train, 
    X_test=X_test, 
    y_test=y_test
)
# %%
project.put("tuned_baseline", tuned_baseline_report)
# %%
tuned_baseline_report.metrics.report_metrics()
# %%
comp = skore.ComparisonReport([baseline_report, tuned_baseline])
comp.help()
# %%
comp.metrics.report_metrics(pos_label=1)

# %% [markdown]
# DEMO PART 2 - after superior review
