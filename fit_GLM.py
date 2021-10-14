import argparse
import gc
import glob
import itertools
import logging
import os
import sys
import time
from asyncio import Event
from pathlib import Path
from typing import Tuple

import cv2
import h5py
import ray
import yaml
from ray.actor import ActorHandle
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import shift as imshift
from scipy.optimize import minimize_scalar
from scipy.stats import binned_statistic
from sklearn import linear_model as lm
from sklearn.metrics import mean_poisson_deviance, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from tqdm.auto import tqdm, trange

sys.path.append(str(Path('.').absolute()))
import io_dict_to_hdf5 as ioh5
from format_data import load_ephys_data_aligned
from utils import *

# For typing purposes

# ProgressBar

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

@ray.remote
def do_glm_fit_vis_skl(train_nsp, test_nsp, x_train, x_test, celln, model_type, lag_list, pbar: ActorHandle, bin_length=40, model_dt=.1):

    ##### Format data #####
    nt_glm_lag = len(lag_list)

    # Shift spikes by -lag for GLM fits
    sps_train = train_nsp[:, celln]  # np.roll(train_nsp[:,celln],-lag)
    sps_test = test_nsp[:, celln]  # np.roll(test_nsp[:,celln],-lag)

    if model_type == 'elasticnetcv':
        # lm.RidgeCV(alphas=np.arange(100,10000,1000))) #  #MultiOutputRegressor(lm.Ridge(),n_jobs=-1))
        model = lm.ElasticNetCV(l1_ratio=[.05, .01, .5, .7])
        model.fit(x_train, sps_train)
        sta_all = np.reshape(model.coef_, (nt_glm_lag,)+nks)
        sp_pred = model.predict(x_test)
    elif model_type == 'ridgecv':
        lambdas = 1024 * (2**np.arange(0, 16))
        model = lm.RidgeCV(alphas=lambdas)
        model.fit(x_train, sps_train)
        sta_all = np.reshape(model.coef_, (nt_glm_lag,)+nks)
        sp_pred = model.predict(x_test)
    else:
        #     lambdas = 2048 * (2**np.arange(0,16))
        lambdas = 2**np.arange(0, 16)
        nlam = len(lambdas)
        # Initialze mse traces for regularization cross validation
        msetrain = np.zeros((nlam, 1))
        msetest = np.zeros((nlam, 1))
        pred_all = np.zeros((x_test.shape[0], nlam))
        w_ridge = np.zeros((x_train.shape[-1], nlam))
        w_intercept = np.zeros((nlam, 1))
        # loop over regularization strength
        for l in range(len(lambdas)):
            model = lm.PoissonRegressor(alpha=lambdas[l], max_iter=300)
            # calculate MAP estimate
            model.fit(x_train, sps_train)
            w_ridge[:, l] = model.coef_
            w_intercept[l] = model.intercept_
            pred_all[:, l] = model.predict(x_test)
            # calculate test and training rms error
            # np.mean((sps_train - model.predict(x_train))**2)
            msetrain[l] = mean_poisson_deviance(
                sps_train, model.predict(x_train))
            # np.mean((sps_test - model.predict(x_test))**2)
            msetest[l] = mean_poisson_deviance(sps_test, pred_all[:, l])
        # select best cross-validated lambda for RF
        best_lambda = np.argmin(msetest)
        w = w_ridge[:, best_lambda]
        intercept = w_intercept[best_lambda]
        ridge_rf = w_ridge[:, best_lambda]
        sta_all = np.reshape(w, (nt_glm_lag,)+nks)
        sp_pred = pred_all[:, best_lambda]
    #     model = make_pipeline(StandardScaler(), lm.PoissonRegressor(alpha=lambdas[best_lambda]))
    #     model.fit(x_train,sps_train)
    # predicted firing rate
    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')
                 ) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(
        bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth[bin_length:-bin_length],
                     pred_smooth[bin_length:-bin_length])
    cc_all = cc[0, 1]
    r2_all = r2_score(sp_smooth, pred_smooth)
    pbar.update.remote(1)
    return cc_all, sta_all, sps_test, sp_pred, r2_all


@ray.remote
def do_glm_fit_vismov_skl(train_nsp, test_nsp, x_train, x_test, move_train, move_test, perm, celln, lag_list, pbar: ActorHandle, bin_length=40, model_dt=.05):
    ##### Format data #####
    nt_glm_lag = len(lag_list)
    w_move = np.zeros(move_train.shape[-1])
    xm_train = move_train[:, perm]
    xm_test = move_test[:, perm]
    x_train = np.concatenate((x_train, xm_train), axis=-1)
    x_test = np.concatenate((x_test, xm_test), axis=-1)
    # Shift spikes by -lag for GLM fits
    sps_train = train_nsp[:, celln]  # np.roll(train_nsp[:,celln],-lag)
    sps_test = test_nsp[:, celln]  # np.roll(test_nsp[:,celln],-lag)

    lambdas = 2**np.arange(0, 16)
    nlam = len(lambdas)
    # Initialze mse traces for regularization cross validation
    error_train = np.zeros((nlam, 1))
    error_test = np.zeros((nlam, 1))
    pred_all = np.zeros((x_test.shape[0], nlam))
    w_cv = np.zeros((x_train.shape[-1], nlam))
    w_intercept = np.zeros((nlam, 1))
    # loop over regularization strength
    for l in range(len(lambdas)):
        model = lm.PoissonRegressor(alpha=lambdas[l], max_iter=300)
        # calculate MAP estimate
        model.fit(x_train, sps_train)
        w_cv[:, l] = model.coef_
        w_intercept[l] = model.intercept_
        pred_all[:, l] = model.predict(x_test)
        # calculate test and training rms error
        error_train[l] = mean_poisson_deviance(sps_train, model.predict(
            x_train))  # np.mean((sps_train - model.predict(x_train))**2)
        # np.mean((sps_test - model.predict(x_test))**2)
        error_test[l] = mean_poisson_deviance(sps_test, pred_all[:, l])
    # select best cross-validated lambda for RF
    best_lambda = np.argmin(error_test)
    w = w_cv[:, best_lambda]
    intercept = w_intercept[best_lambda]
    sta_all = np.reshape(w[:-xm_train.shape[-1]], (nt_glm_lag,)+nks)
    w_move[perm] = w[-xm_train.shape[-1]:]
    sp_pred = pred_all[:, best_lambda]

    # predicted firing rate
    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')
                 ) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(
        bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth[bin_length:-bin_length],
                     pred_smooth[bin_length:-bin_length])
    cc_all = cc[0, 1]
    r2_all = r2_score(sp_smooth, pred_smooth)
    pbar.update.remote(1)
    return cc_all, sta_all, sps_test, sp_pred, r2_all, w_move


@ray.remote
def do_glm_fit_mot_skl(train_nsp, test_nsp, move_train, move_test, celln, perms, model_type, pbar: ActorHandle, bin_length=40, model_dt=.05):

    ##### Format data #####
    w_move = np.zeros(move_train.shape[1])
    # Shift spikes by -lag for GLM fits
    sps_train = train_nsp[:, celln]  # np.roll(train_nsp[:,celln],-lag)
    sps_test = test_nsp[:, celln]  # np.roll(test_nsp[:,celln],-lag)

    # Reshape data (video) into (T*n)xN array
    x_train = move_train[:, perms]
    x_test = move_test[:, perms]

    if model_type == 'elasticnetcv':
        # lm.RidgeCV(alphas=np.arange(100,10000,1000))) #  #MultiOutputRegressor(lm.Ridge(),n_jobs=-1))
        model = lm.ElasticNetCV(l1_ratio=[.05, .01, .5, .7])
        model.fit(x_train, sps_train)
        w_move[perms] = model.coef_
        sp_pred = model.predict(x_test)
    elif model_type == 'ridgecv':
        lambdas = 1024 * (2**np.arange(0, 16))
        model = lm.RidgeCV(alphas=lambdas)
        model.fit(x_train, sps_train)
        w_move[perms] = model.coef_
        sp_pred = model.predict(x_test)
    else:
        model = lm.PoissonRegressor(alpha=0, max_iter=300)
        # calculate MAP estimate
        model.fit(x_train, sps_train)
        intercept = model.intercept_
        w_move[perms] = model.coef_
        sp_pred = model.predict(x_test)

    # bin the firing rate to get smooth rate vs time
    sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')
                 ) / (bin_length * model_dt)
    pred_smooth = (np.convolve(sp_pred, np.ones(
        bin_length), 'same')) / (bin_length * model_dt)
    # a few diagnostics
    err = np.mean((sp_smooth-pred_smooth)**2)
    cc = np.corrcoef(sp_smooth[bin_length:-bin_length],
                     pred_smooth[bin_length:-bin_length])
    if cc[0, 1] == np.nan:
        cc_all = 0
    else:
        cc_all = cc[0, 1]
    r2_all = r2_score(sp_smooth[bin_length:-bin_length],
                      pred_smooth[bin_length:-bin_length])
    pbar.update.remote(1)
    return cc_all, w_move, sps_test, sp_pred, r2_all
