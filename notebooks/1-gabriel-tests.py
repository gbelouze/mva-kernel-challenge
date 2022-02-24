# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: mva-kernel-challenge
#     language: python
#     name: mva-kernel-challenge
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Visualisation

# %%
import numpy as np
import matplotlib.pyplot as plt

import challenge.data.io as io
import challenge.visualization.visualize as visualize
from challenge.data.paths import x_train, y_train

# %%
x_train = io.xload(x_train)

# %%
visualize.plot(x_train[108])
plt.show()

# %% [markdown]
# ## kernel ridge regression

# %%
from challenge.models.predictor import KernelRidgeRegressor
from challenge.models.kernel import RBF
import challenge.data.paths as paths
import pandas as pd

# %%
xtrain = io.xload(paths.x_train)[:800]
ytrain = io.yload(paths.y_train)[:800]
xval = io.xload(paths.x_train)[800:850]
yval = io.yload(paths.y_train)[800:850]

# %%
rbf = RBF()
ridge = KernelRidgeRegressor(kernel=rbf, lambd=0.01)

# %%
ridge.fit(xtrain, ytrain)

# %%
ypred = ridge.predict(xval)
pd.DataFrame(ypred - yval).describe()

# %%
ypred = ridge.predict(xtrain)
pd.DataFrame(ypred - ytrain).describe()

# %%
