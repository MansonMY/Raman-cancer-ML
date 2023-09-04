# %%
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pybaselines
from pybaselines.polynomial import imodpoly, modpoly
from pybaselines.whittaker import asls, aspls
from scipy.ndimage import uniform_filter1d
import glob as glob

files = glob.glob("./*.txt")
# data selected, x_data is wavelenghth, y_data is Raman intensity
for file in files:
    df_data = pd.read_csv(file, sep = '\t')
    x_data = df_data.iloc[0,70:600]
    y_data = df_data.iloc[1,70:600]
# wavenumver convert from wavelenghth
    Wavenumber = ((10**7)/(785))-((10**7)/(x_data))
# baseline fitting using aspls algorithm from library pybaselines
    lam = 100000
    tol = 1e-3
    max_iter = 20

    fit_2, params_2 = aspls(y_data, lam=lam, tol=tol, max_iter=max_iter)
# baseline removal
    baseline_sub = y_data - fit_2
# result smooth to remove noisty peak here we used 3
    smooth_baseline_sub = uniform_filter1d(baseline_sub, 3)

    fig,ax = plt.subplots(3,1,sharex=True)
    fig.suptitle('A{}'.format(file))
    
    ax[0].plot(Wavenumber, y_data, color = 'blue', label = 'raw data')
    ax[0].plot(Wavenumber, fit_2, color = 'green', label = 'aspls_baseline fitting')
    ax[0].legend(loc='best', framealpha = 0.1)
    ax[1].plot(Wavenumber, baseline_sub, label = 'baseline removal')
    ax[1].legend(loc='upper right',framealpha = 0.1)
    ax[2].plot(Wavenumber,smooth_baseline_sub, label = 'fitting noisy')
    ax[2].legend(loc='upper right', framealpha = 0.1)
    fig.text(0.5, 0, 'Wavenumber(cm$^-$$^1$)', ha='center', fontsize = 20)
    fig.text(0, 0.5, 'Intensity', va='center', rotation='vertical',fontsize = 20)

    plt.subplots_adjust(hspace = 0.3)
    plt.show()
print('done')






    
  

# %%
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pybaselines
from pybaselines.polynomial import imodpoly, modpoly
from pybaselines.whittaker import asls, aspls
from scipy.ndimage import uniform_filter1d
import glob as glob

files = glob.glob("./*.txt")

data_output = pd.DataFrame(dtype = 'float')

for file in files:
    df_data = pd.read_csv(file, sep = '\t')
    x_data = df_data.iloc[0,70:600]
    y_data = df_data.iloc[1,70:600]

    Wavenumber = ((10**7)/(785))-((10**7)/(x_data))
    
    lam = 100000
    tol = 1e-3
    max_iter = 20

    fit_2, params_2 = aspls(y_data, lam=lam, tol=tol, max_iter=max_iter)
    baseline_sub = y_data - fit_2
    smooth_baseline_sub = uniform_filter1d(baseline_sub, 3)
    
    data_output['A{}'.format(file)] = smooth_baseline_sub

data_output['Wavenumber'] = Wavenumber.reset_index(drop=True)
data_output.set_index('Wavenumber',inplace=True)
data_output.columns = list(files)
data_output.to_csv("data_processed.csv")
print(data_output)
#print('done')
# %%
