# %% [markdown]
# Import packages and prepare data
# --------------------------------

# %%
import pandas as pd
import numpy as np
import thecannon as tc
import matplotlib.pyplot as plt

from scipy.optimize import Bounds

# %%
# Read in the training labels and model spectra
# These are 1000 samples of labels 'EWT','LMA','N','CHL','CAR','ANT'
# used to modelled spectra at wavelengths 400..1..2500nm with the PROSPECT code
LUC_labels  = pd.read_csv('PROSPECT_input.csv')
LUC_spectra = pd.read_csv('PROSPECT_LUT.csv')

# %% [markdown]
# These data include the following labels:
# 
# - EWT: Equivalent water thickness
# - LMA: Dry matter content (DMC) equivalent
# - CHL: chlorophyll
# - CAR: carotenoids
# - ANT: anthocyanins
# - N: leaf structure parameter (integer, 1 through 4)

# %%
# Prepare input for a complete and restrictred Cannon model
prospect_wavelength = np.arange(400,2501,1)

labels = ['EWT','LMA','N','CHL','CAR','ANT']
prospect_labels = np.array(LUC_labels[labels])

labels_limited = ['EWT','LMA']
prospect_labels_limited = np.array(LUC_labels[labels_limited])

prospect_spectra = np.array(LUC_spectra).T
prospect_spectra_ivar = (100./prospect_spectra)**2.0 # SNR 100

for label in labels:
    print(f"{label:4s}: {np.min(LUC_labels[label]):.6e}, {np.max(LUC_labels[label]):.6e}")

# %%
fig_initial = plt.figure()
ax_std = fig_initial.add_subplot(2, 1, 1)
ax_log = fig_initial.add_subplot(2, 1, 2)
ax_std.boxplot(prospect_labels)
ax_log.boxplot(prospect_labels)
ax_log.set_yscale("log")
ax_std.set_xticks(range(1, 7), "")
ax_log.set_xticks(range(1, 7), labels)


# %% [markdown]
# Run a restricted model
# ----------------------
# 
# So far, we've allowed every label to be completely free in fitting. However, this is unphysical; most of the labels (with the expcetion of `N`) have units weight per unit area, so they have a physical lower limit of 0.
# 
# Note that the theta limits in the `RestrictedCannonModel` are limits on the $`\theta`$ coefficient of the model for that term; they do **not** represent physical limits of the label value (a rookie mistake that the developer made for two weeks). Instead, to get limits on the output labels, one needs to send label bounds through to the `test` function.
# 
# By not specifying any `theta_bounds`, `RestrictedCannonModel` is equivalent to a standard `CannonModel`.

# %%
# Initialise and train the restricted Cannon Model
from thecannon.restricted import RestrictedCannonModel
prospect_model_restricted = RestrictedCannonModel(
    prospect_labels,
    prospect_spectra, prospect_spectra_ivar,
    vectorizer=tc.vectorizer.PolynomialVectorizer(list(labels), 2),dispersion=prospect_wavelength)
# prospect_model_restricted.theta_bounds = {
#     t: (1e-9, None) for t in prospect_model_restricted.vectorizer.human_readable_label_vector.split(" + ")[1:]  # Ignore +1 term
# }
# prospect_model_restricted.theta_bounds["N"] = (1.0, 4.0)

prospect_theta_restricted, prospect_s2_restricted, prospect_metadata_restricted = prospect_model_restricted.train(threads=1)
print(prospect_model_restricted.theta_bounds)

# %%
# %pdb
# Test the label recovery of the same spectra
bnds = ([-1 for l in prospect_model_restricted.vectorizer.label_names], [1000.0 for l in prospect_model_restricted.vectorizer.label_names])
prospect_test_labels_restricted, prospect_test_cov_restricted, prospect_metadata_restricted = prospect_model_restricted.test(
    prospect_spectra, prospect_spectra_ivar, 
    initial_labels=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    op_kwds={"bounds": Bounds(
        lb=[0.1 for label in prospect_model_restricted.vectorizer.label_names], 
        ub=[np.inf for label in prospect_model_restricted.vectorizer.label_names], 
        keep_feasible=[False for label in prospect_model_restricted.vectorizer.label_names])}
    )

# %%
for i in range(prospect_test_labels_restricted.shape[1]):
    print(f"{i}: {np.min(prospect_test_labels_restricted[:, i])}")

# %%
fig_one_to_one_restricted = tc.plot.one_to_one(prospect_model_restricted, prospect_test_labels_restricted)

# %%



