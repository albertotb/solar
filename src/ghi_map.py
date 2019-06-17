
# coding: utf-8

import pandas as pd
import numpy as np
import math
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from joblib import Parallel, delayed
import torch
import gpytorch

DATA_PATH = '/home/SHARED/SOLAR/data/'


def get_grid(lonlat, decimals=3):

    prec = 10**decimals
    max_min = lonlat.agg([np.max, np.min])

    lat = np.arange(int(math.floor(max_min.loc['amin',  'Latitude']*prec)),
                    int(math.ceil( max_min.loc['amax',  'Latitude']*prec))+1)/prec

    lon = np.arange(int(math.floor(max_min.loc['amin', 'Longitude']*prec)),
                    int(math.ceil( max_min.loc['amax', 'Longitude']*prec))+1)/prec

    xx, yy = np.meshgrid(lon, lat)
    return pd.DataFrame({'lon': xx.ravel(), 'lat': yy.ravel()})


def train_scikit(X_train, y_train, X_test):
    gpr = GaussianProcessRegressor(kernel=RBF(10, (1e-3, 1e-3)), n_restarts_optimizer=10)
    gpr.fit(X_train, y_train)
    return gpr.predict(X_test)

def train_torch(X_train, y_train, X_test, n_epochs=500, verbose=False):
    # Features to torch tensor
    train_x = torch.from_numpy(X_train.values).double()
    # Target to torch tensor
    train_y = torch.from_numpy(y_train.values).double()
    # Test features to torch tensor
    test_x = torch.from_numpy(X_test.values).double()
    ####################################################################
    ## Model specification. Simple GP model (RBFKernel)
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    ####################################################################
    ## Model training
    # Find optimal model hyperparameters
    model.double()
    model.train()
    likelihood.train()
    #
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  
    ], lr=0.1)

    # "Loss" = the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    #
    for i in range(n_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    ####################################################################
    # Make predictions
    # Get into evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #test_x = torch.from_numpy(test_x).double()
        observed_pred = likelihood(model(test_x))
        
    return observed_pred.mean.numpy()

def train_gpr(df, datetime):
    X_train = df[['Longitude', 'Latitude']]
    y_train = df['GHI']

    X_test = get_grid(df[['Longitude', 'Latitude']])
    X_test['GHI'] = train_torch(X_train, y_train, X_test)

    return datetime, X_test.set_index(['lon', 'lat'])


# read minute data and location info
df   = pd.read_pickle(DATA_PATH + 'oahu_min_final.pkl')
info = pd.read_pickle(DATA_PATH + 'info.pkl')

df_long = (df#.iloc[0:1]
             .stack()
             .reset_index('Datetime')
             .join(info[['Longitude', 'Latitude']])
             .rename(columns={0: 'GHI'})
             .dropna())

res = Parallel(n_jobs=8)(delayed(train_gpr)(df, datetime)
                         for datetime, df in df_long.groupby('Datetime'))

df_wide = pd.concat(dict(res)).unstack(level=['lon', 'lat']).sort_index(axis=1)

df_wide.columns = df_wide.columns.droplevel(0)

df_wide.to_pickle(DATA_PATH + 'ghi_map_torch.pkl')
