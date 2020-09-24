# -*- coding: utf-8 -*-
"""Using Auto Encoder with Outlier Detection
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from statistics import mean 
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from utils.utility import check_parameter
from utils.utility import weight_init
from utils.stat_models import pairwise_distances_no_broadcast

from .base import BaseDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
        """ Pytorch AutoEncodeur """
        def __init__(self , input_size, hidden_neurons, hidden_activation, dropout_rate, output_activation):
            super().__init__()

            #Input layer
            self.linear=nn.ModuleList([nn.Linear( input_size, hidden_neurons[0])])
            self.hidden_activation = hidden_activation
            self.dropout = nn.Dropout(dropout_rate)
            last_h_neurons = hidden_neurons[0]

            # Additional layers
            for i in range(1,len(hidden_neurons)):
                self.linear.append(nn.Linear(last_h_neurons , hidden_neurons[i]))
                last_h_neurons = hidden_neurons[i]

           # Output layers
            self.linear.append(nn.Linear(last_h_neurons, input_size))
            self.output_activation = output_activation


        def forward(self,w): 
            #Input layer

            out = self.linear[0](w)
            out = self.hidden_activation(out)
            out = self.dropout(out)

            # Additional layers
            for linear1 in self.linear[1:-1]:
                out = linear1(out)
                out = self.hidden_activation(out)
                out = self.dropout(out)

            # Output layers
            out = self.linear[-1](out)
            out = self.output_activation(out)
            return out




class AutoEncoder(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations unsupervisedly. Similar to PCA, AE could be used to
    detect outlying objects in the data by calculating the reconstruction
    errors. 

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32, 32, 64])
        The number of neurons per hidden layers.

    hidden_activation : obj, optional (default=nn.ReLU(True))
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.


    output_activation : obj, optional (default=nn.Sigmoid())
        Activation function to use for output layer.


    loss : str or obj, optional (default=nn.MSELoss())
        String (name of objective function) or objective function.

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.001)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbosity >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
        
     threshold : float, optional (default=1.0)
         It is the ``n_samples * threshold`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.


    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : toch Object
        The underlying AutoEncoder in Keras.
        
    history_: List 
        The AutoEncoder training history.
    """

    def __init__(self, hidden_neurons=None,
                 hidden_activation=nn.ReLU(True), output_activation=nn.Sigmoid(),
                 loss=nn.MSELoss(), optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.001, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, threshold=1.0):
        super(AutoEncoder, self).__init__()
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.threshold = threshold

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _build_model(self):
        return Autoencoder(self.n_features_ * self.n_samples_ ,  self.hidden_neurons_ , self.hidden_activation , self.dropout_rate, self.output_activation)

    # noinspection PyUnresolvedReferences
    def fit(self, X , y):
        """Fit detector

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

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[1], X.shape[2]


        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > (self.n_features_ * self.n_samples_)  :
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_
        
        
        #Loader Torch
        
        w_size=X.shape[1]*X.shape[2]        
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(X).float().view(([X.shape[0],w_size]))
        ) , batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Build AE model & fit with X
        self.model_ = self._build_model().to(device)
        self.model_.apply(weight_init)
        criterion = self.loss.to(device)
        if (self.optimizer == "adam" )  : 
            opt = torch.optim.Adam(params = self.model_.parameters(), lr=self.l2_regularizer)
        if (self.optimizer == "sgd" )  : 
            opt= torch.optim.SGD(params = self.model_.parameters(), lr=self.l2_regularizer, momentum=0.9)

        # Train AE    
        self.history_ = []
        for epoch in range(self.epochs):
            losses=[]
            for [batch] in train_loader:
                opt.zero_grad()
                batch = batch.to(device)
                out = self.model_(batch)
                loss = criterion(out,batch)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.history_.append("Loss at epoch " + str(epoch)+" = "+str(mean(losses)))
            
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
        test_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """ 
        w_size=X.shape[1]*X.shape[2]   
        
        #Loader Torch
        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(X).float().view(([X.shape[0],w_size]))
        ) , batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        results = self.predict(test_loader)
        
        test_scores=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                              results[-1].flatten().detach().cpu().numpy()])

        return test_scores
    
    
    def predict(self, test_loader):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        test_loader : Loader pytorch
            The input samples.

        Returns
        -------
        results : array of tensor (n_samples,windows_size)
            Score for each time window.
        """
        results=[]
        for [batch] in test_loader:
            batch = batch.to(device)
            pred=self.model_(batch)
            results.append(torch.mean((batch-pred)**2,axis=1))
        return results