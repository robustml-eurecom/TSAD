# -*- coding: utf-8 -*-
"""
Part of the implementation comes from  https://github.com/danieltan07/dagmm

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

#class Cholesky(torch.autograd.Function):
#    def forward(ctx, a):
#        l = torch.cholesky(a, False)
#        ctx.save_for_backward(l)
#        return l
#    
#    def backward(ctx, grad_output):
#        l, = ctx.saved_variables
#        linv = l.inverse()
#        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
#            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
#        s = torch.mm(linv.t(), torch.mm(inner, linv))
#        return s
    
    
class Encoder(nn.Module):
        """ Pytorch Encodeur """
        def __init__(self , input_size, hidden_neurons, hidden_activation, dropout_rate = 0):
            super().__init__()

            #Input layer
            self.linear=nn.ModuleList([nn.Linear( input_size, hidden_neurons[0])])
            self.hidden_activation = hidden_activation
            self.dropout = nn.Dropout(dropout_rate)
            last_h_neurons = hidden_neurons[0]

            # Additional layers
            for i in range(1,int((len(hidden_neurons)+1)/2)):
                self.linear.append(nn.Linear(last_h_neurons , hidden_neurons[i]))
                last_h_neurons = hidden_neurons[i]



        def forward(self,w): 
            #Input layer

            out = self.linear[0](w)
            out = self.hidden_activation(out)
            out = self.dropout(out)

            # Additional layers
            for linear1 in self.linear[1:]:
                out = linear1(out)
                out = self.hidden_activation(out)
                out = self.dropout(out)
            return out

class Decoder(nn.Module):
        """ Pytorch Decodeur """
        def __init__(self , input_size, hidden_neurons, hidden_activation, output_activation, dropout_rate = 0):
            super().__init__()

            self.linear=nn.ModuleList()
            self.hidden_activation = hidden_activation
            self.dropout = nn.Dropout(dropout_rate)

            # Additional layers
            for i in range(int(len(hidden_neurons)/2),len(hidden_neurons)-1):
                self.linear.append(nn.Linear(hidden_neurons[i] , hidden_neurons[i+1]))

           # Output layers
            self.linear.append(nn.Linear(hidden_neurons[-1], input_size))
            self.output_activation = output_activation


        def forward(self,w): 

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
    
class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, input_size, hidden_neurons, hidden_activation, dropout_rate, output_activation, n_gmm = 2, latent_dim=3):
        super(DaGMM, self).__init__()

        self.encoder = Encoder(input_size, hidden_neurons, hidden_activation, )
        self.decoder = Decoder(input_size, hidden_neurons, hidden_activation,  hidden_activation)

        layers = []
        layers += [nn.Linear(latent_dim,10)]
        layers += [hidden_activation]        
        layers += [nn.Dropout(dropout_rate)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [output_activation]


        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim,latent_dim))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)

        rec_cosine = nn.functional.cosine_similarity(x, dec, dim=1)
        rec_euclidean = self.relative_euclidean_distance(x, dec)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    



class Dagmm(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations unsupervisedly. Similar to PCA, AE could be used to
    detect outlying objects in the data by calculating the reconstruction
    errors. 

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32, 10, 1, 1, 10, 32, 64])
        The number of neurons per hidden layers.

    hidden_activation : obj, optional (default=nn.Tanh())
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.


    output_activation : obj, optional (default=nn.Softmax(dim=1))
        Activation function to use for output layer.
        
    gmm_k : int, optional (default=4)
        
    lambda_energy : float in (0., 1), optional (default=0.1)
    
    lambda_cov_diag : float in (0., 1), optional (default=0.005)
 
    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.
        
    dropout_rate : float in (0., 1), optional (default=0.5)
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
                 hidden_activation=nn.Tanh(), output_activation=nn.Softmax(dim=1),
                 gmm_k=4 , lambda_energy = 0.1 , lambda_cov_diag = 0.005 , optimizer='adam',
                 epochs=100, batch_size=32,dropout_rate=0.5,
                 l2_regularizer=0.001, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, threshold=1.0):
        super(Dagmm, self).__init__()
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.threshold = threshold
        
        
        
        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 10, 1, 1, 10, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

        

    def _build_model(self):
        return DaGMM(self.n_features_ * self.n_samples_ ,  self.hidden_neurons_ , self.hidden_activation , self.dropout_rate, self.output_activation)

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
                enc, dec, z, gamma = self.model_(batch)
                total_loss, sample_energy, recon_error, cov_diag = self.loss_function(batch, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)
                total_loss.backward()
                opt.step()
                losses.append(total_loss.data.item())
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5)
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
        test_energy=[]
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for [batch] in test_loader:
            batch = batch.to(device)
            enc, dec, z, gamma = self.model_(batch)
            sample_energy, _ = self.compute_energy(z, size_average=False)
            results.append(sample_energy.cpu())
            test_energy.append(sample_energy.data.cpu().numpy())
        test_energy = np.concatenate(test_energy,axis=0)
        self.threshold = np.percentile(test_energy, 100 - 20)
        return results
    
    
    
    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)
        # K
        phi = (sum_gamma / N)
        self.phi = phi.data
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.phi.to(device)
        if mu is None:
            mu = self.mu.to(device)
        if cov is None:
            cov = self.cov.to(device)
        k, D, _ = cov.size()
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + torch.eye(D)*eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            #det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        #det_cov = device(torch.cat(torch.from_numpy(np.array(det_cov))))
        det_cov = torch.from_numpy(np.float32(np.array(det_cov))).to(device)
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        
        return loss, sample_energy, recon_error, cov_diag

