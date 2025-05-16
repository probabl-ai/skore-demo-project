# %%
import skore
import skrub
from skrub import tabular_learner
from skore import EstimatorReport
from skore_remote_project.project.project import Project
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# Fetch the dataset. We will use the traffic violations dataset from skrub.

# %%
dataset = skrub.datasets.fetch_traffic_violations()
df = dataset.traffic_violations

# %% [markdown]
# Let's first have a quick look at the data inside. 
# %% 
skrub.TableReport(df, max_plot_columns=50)

# %%
y.describe()

# %%
# for the sake of this demo, we will use a small subset of the data.
X = dataset.X.sample(1000, random_state=1)
y = dataset.y.sample(1000, random_state=1)
X_train, X_test, y_train, y_test = skore.train_test_split(X, y, random_state=1)


# %% [markdown]
# Simpler is better.
# Let's do a simple baseline.

# %%
baseline = tabular_learner('classification')
baseline_report = EstimatorReport(baseline, X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test)
baseline_report.help()

# %% 
baseline_report.metrics.report_metrics()
# %%
# Whatever other simple model. 

# %%
# create project
project = Project(name="project demo", tenant="Probabl")
# %%
project.put("baseline", baseline_report)

# %%
baseline_2 = tabular_learner(RandomForestClassifier())
baseline_report_2 = EstimatorReport(baseline_2, X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test)
# %%
project.put("baseline_2", baseline_report_2)
# %%
