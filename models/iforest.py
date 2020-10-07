# -*- coding: utf-8 -*-
"""
Part of the implementation comes from  https://github.com/yzhao062/pyod/

"""

from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector

class IF(BaseDetector):
    """ Isolation Forest 
    
    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.
    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function.
    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
    bootstrap : bool, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.
    max_samples_ : integer
        The actual number of samples
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0, threshold=0):
        super(IF, self).__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination=contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples,nx*ny))
        self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                         max_samples=self.max_samples,
                                         contamination=self.contamination,
                                         max_features=self.max_features,
                                         bootstrap=self.bootstrap,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose)

        self.detector_.fit(X=X, y=None, sample_weight=None)

        # invert decision_scores_. Outliers comes with higher outlier scores.
        self.decision_scores_ = self.detector_.decision_function(X).ravel() * -1
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples,nx*ny))
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        return self.detector_.decision_function(X).ravel() * -1

    