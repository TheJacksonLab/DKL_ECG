# import sys
# ind_job = int(sys.argv[1])-1
# ind_job=5

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(ind_job%4)
# os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import gpytorch
from gpytorch import settings as gpt_settings
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.constraints import GreaterThan, Interval
from scipy.special import gamma
from copy import deepcopy
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from sklearn.mixture import GaussianMixture
gpt_settings.cholesky_jitter._set_value(1e-4,1e-4,None)

# clusters the dkl-projected test data 
# dim_d is the dimension of the latent feature
# lat_tr is the dkl-projected training data
# lat_te is the dkl-projected test data
def lat_gmm(k_neighbors, dim_d, lat_tr, lat_te):
    # average observed noise by the model via the training data
    noise_tr = np.sqrt(np.mean((lat_tr[:,dim_d+1]-lat_tr[:,dim_d+2])**2))
    
    # minimum population needed for gmm cluster to be included in analysis
    n_cutoff = 15
    
    # train gmm on the test set
    gmm = GaussianMixture(n_components=k_neighbors,n_init=10,max_iter=1000).fit(lat_te[:,:dim_d])
    total_pred = gmm.predict(lat_te[:,:dim_d]).flatten()
    # print(clust_pred)
    
    # classify each test point into a cluster
    total_prob = gmm.predict_proba(lat_te[:,:dim_d])
    total_prob = np.array([np.max(total_prob[j,:]) for j in range(len(total_pred))])
    
    # n_pop is the number of clusters with population above cutoff value
    # this generates a correspondence between overall cluster index and index in list of populated clusters
    n_pop = 0 
    clust_arr = np.zeros(k_neighbors)
    for j in range(k_neighbors):
        n_clust = np.sum(total_pred == j)
        if n_clust > n_cutoff:
            clust_arr[n_pop] = j
            n_pop += 1
    
    # clust_arr contains the indices for all clusters to be analyzed
    clust_arr = clust_arr[:n_pop]
    data_clust = np.zeros((n_pop,14))
    for j in range(n_pop):
        # indices within the dataset for populated cluster j
        clust_inds = total_pred == clust_arr[j]
        clust_prob = total_prob[clust_inds]
        clust_weight = np.sum(clust_prob)
        clust_e = lat_te[clust_inds,(dim_d+1):(dim_d+6)]
        clust_neff = clust_weight**2/np.sum(clust_prob**2)
        
        # rescale the observed test energy value by the predicted mean
        # this allows for larger clusters to be made
        clust_e[:,0] = clust_e[:,0] - clust_e[:,1]
        clust_e[:,1] = 0
        
        data_clust[j,0] = np.sum(clust_prob*clust_e[:,0])/clust_weight # weighted mean test energy
        data_clust[j,1] = np.sqrt(np.sum(clust_prob*(clust_e[:,0]-data_clust[j,0])**2)/clust_weight/(clust_neff-1)) # weighted std err of mean
        data_clust[j,2] = data_clust[j,1]*np.sqrt(clust_neff) # weighted stdev of test energy mean
        data_clust[j,3] = data_clust[j,2]/np.sqrt(2*(clust_neff-1)) # weighted std err of test energy stdev
        
        data_clust[j,4] = np.sum(clust_prob*clust_e[:,1])/clust_weight # weighted mean pred energy
        data_clust[j,5] = np.sqrt(np.sum(clust_prob*(clust_e[:,1]-data_clust[j,4])**2)/clust_weight/(clust_neff-1)) #fluct of the pred mean
        data_clust[j,6] = np.sqrt(np.sum(clust_prob*clust_e[:,2]**2)/clust_weight) # weighted mean pred noise
        std_mean = np.sum(clust_prob*clust_e[:,2])/clust_weight
        data_clust[j,7] = np.sqrt(np.sum(clust_prob*(clust_e[:,2]-std_mean)**2)/clust_weight/(clust_neff-1)) # std error of mean pred noise
        
        data_clust[j,8] = clust_neff # overall pred cluster energy width = fluct of pred mean
        data_clust[j,9] = noise_tr # overall mean pred noise (full dataset)
        data_clust[j,10] = np.mean(clust_prob) # distribution of weights within the cluster
        data_clust[j,11] = np.std(clust_prob)
        data_clust[j,12] = np.sum(clust_prob*clust_e[:,3])/clust_weight # variance components
        data_clust[j,13] = np.sum(clust_prob*clust_e[:,4])/clust_weight
        # data_clust[j,14] = clust_neff
        
    
    return data_clust

def init_custom(m):
    if type(m) == torch.nn.Linear:
        bound = 1/np.sqrt(m.in_features)
        torch.nn.init.normal_(m.weight, 0, bound)
        # torch.nn.init.zeros_(m.bias)

# DKL neural network structure
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, 80))
        self.add_module('drop1', torch.nn.BatchNorm1d(80))
        self.add_module('relu1', torch.nn.ELU())
        self.add_module('linear2', torch.nn.Linear(80,40))
        self.add_module('drop2', torch.nn.BatchNorm1d(40))
        self.add_module('relu2', torch.nn.ELU())
        self.add_module('linear3', torch.nn.Linear(40, 20))
        self.add_module('drop3', torch.nn.BatchNorm1d(20))
        self.add_module('relu3', torch.nn.ELU())
        self.add_module('linear4', torch.nn.Linear(20, 10))
        self.add_module('drop4', torch.nn.BatchNorm1d(10))
        self.add_module('relu4', torch.nn.ELU())
        self.add_module('linear5', torch.nn.Linear(10, out_dim))
        self.add_module('drop5', torch.nn.BatchNorm1d(out_dim))
        self.add_module('relu5', torch.nn.ELU())
        self.apply(init_custom)
        # self.add_module('relu5', torch.nn.Sigmoid())

# DKL output -> mean in latent space
class VariationalMean(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(VariationalMean, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, data_dim))
        # self.add_module('relu1', torch.nn.Tanh())
        # self.add_module('linear2', torch.nn.Linear(data_dim, data_dim))
        self.apply(init_custom)
        with torch.no_grad():
            self[0].weight = torch.nn.Parameter(torch.eye(data_dim))
        # self.add_module('relu5', torch.nn.Sigmoid())

# DKL output -> variance in latent space
class VariationalVar(torch.nn.Sequential):
    def __init__(self, data_dim):
        super(VariationalVar, self).__init__()
        # droprate = 0.35
        # self.add_module('linear1', torch.nn.Identity())
        self.add_module('linear1', torch.nn.Linear(data_dim, data_dim))
        self.add_module('relu1', torch.nn.Sigmoid())
        self.add_module('linear2', torch.nn.Linear(data_dim, 1))
        self.apply(init_custom)
        torch.nn.init.constant_(self[-1].bias,-2) # Initial bias to make inital variance reasonable
        # self.add_module('relu5', torch.nn.Sigmoid())

# Variational layer after DKL projection
class VAELayer(torch.nn.Module):
    def __init__(self, data_dim):
        super(VAELayer, self).__init__()
        
        self.mu = VariationalMean(data_dim)
        self.logvar = VariationalVar(data_dim)
    
    # Layer between the feature extraction and the GPR
    
    def encode(self, x):
        return self.mu(x), self.logvar(x)
        # mu = self.mu(x)
        # return mu, self.logvar_lazy*torch.ones_like(mu)
    
    # Samples the latent distribution to send to GP
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ApproximateGPLayer(gpytorch.models.ApproximateGP):
        def __init__(self, num_dim, grid_size=100):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=grid_size)
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, torch.randn((grid_size,num_dim)),
                variational_distribution=variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            if num_dim < 2:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=num_dim))
            
        
        def forward(self, x):
            # print(x.shape)
            mean = self.mean_module(x)
            # print(mean.shape)
            covar = self.covar_module(x)
            # print(covar.shape)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLVAEModel(gpytorch.Module):
    def __init__(self, in_dim, num_dim, grid_size=100):
        super(DKLVAEModel, self).__init__()
        # DKL layer
        self.feature_extractor = LargeFeatureExtractor(in_dim, num_dim)
        # variational layer
        self.vae_layer = VAELayer(num_dim)
        # GPR on the latent projection
        self.gp_layer = ApproximateGPLayer(num_dim=num_dim, grid_size=grid_size)
        self.num_dim = num_dim

    def forward(self, x):
        lat = self.feature_extractor(x)
        feat, mu, logvar = self.vae_layer(lat)
        res = self.gp_layer(feat)
        # res is the gpr prediction on the provided points
        # mu is the mean of the latent space projection
        # logvar is the log variance of the latent space distribution
        # feat is the stochastic sample of the latent space distributions
        return res, mu, logvar, feat

def prior_exp(x):
    out_dim = len(x[0,:])
    n_0 = gamma(out_dim/2)/(2*np.pi**(out_dim/2)*gamma(out_dim))
    return n_0*torch.exp(-torch.sqrt(torch.sum(x**2,1)))


# This is the used KL divergence function 
# Sometimes unstable on scruggs
def kl_vae_qp(x, mu, logvar):
    out_dim = len(x[0,:])
    var = torch.exp(logvar[:,0])
    # dim_out = len(x[0])
    diff_tensor = torch.zeros((out_dim,len(x[:,0]),len(mu[:,0])))
    
    # Squared difference tensor between the given sample points x and the center of the given distributions
    for i in range(out_dim):
        diff_tensor[i] = (x[:,i:(i+1)] - mu[:,i:(i+1)].T)**2
    
    # Applies the exponential to the distance tensor
    diff_tensor = torch.exp(-.5*torch.sum(diff_tensor,0)/var)/var**(out_dim/2)
    
    #Sums the exponentials over the batch and normalizes
    pdf_x = torch.sum(diff_tensor,1)/len(x[:,0])/(2*np.pi)**(out_dim/2)
    
    # pdf_prior = prior_pdf(x)
    pdf_prior = torch.exp(-.5*torch.sum(x**2,1))/(2*np.pi)**(out_dim/2)
    # pdf_prior = prior_exp(x)
    # print(pdf_x)
    # print(pdf_prior)
    # Returns the batch sum, which is the approximate KL divergence 
    return torch.sum(torch.log( pdf_x / pdf_prior ))

# Take in train/test data, return full vdkl fit and gmm clustering analysis
# dn is working directory, f_ext added to data files to distinguish
def vdkl_trial(train_x, train_y, test_x, test_y, params, dn, f_ext, train_mean):
    dir_out = 'CV_test'
    
    n_batch = int(params[0])
    out_dim = int(params[1])
    k_fold = int(params[2])
    lambda_kl = params[3]
    
    n_induc = 1000
    dim_d = len(train_x[0,:])
    n_train = len(train_x[:,0])
    
    n_train_sub = int(n_train*(k_fold-1)//k_fold)
    if k_fold == 1:
        n_train_sub = n_train
    
    
    model_list = [DKLVAEModel(dim_d,out_dim,grid_size=n_induc) for k in range(k_fold)]
    likelihood_list = [gpytorch.likelihoods.GaussianLikelihood() for k in range(k_fold)]
    mll_list = [gpytorch.mlls.PredictiveLogLikelihood(likelihood_list[k], model_list[k].gp_layer, num_data=n_train_sub) for k in range(k_fold)]
    
    hypers = {
        'covar_module.outputscale': torch.tensor(np.std(train_y.detach().cpu().numpy())),
    }
    for model in model_list:
        model.gp_layer.initialize(**hypers)
    hypers = {
        'noise': torch.tensor(np.std(train_y.detach().cpu().numpy())),
    }
    for likelihood in likelihood_list:
        likelihood.initialize(**hypers)
    
    val_data = np.zeros(k_fold)
    # print(model_list[0].gp_layer.covar_module.outputscale.detach().cpu().numpy())
    for k in range(k_fold):
        if k_fold > 1:
            inds_k = np.roll(range(n_train),k*n_train//k_fold)[:n_train_sub]
            inds_val = np.roll(range(n_train),k*n_train//k_fold)[n_train_sub:]
            train_x_k = train_x[inds_k,:]
            train_y_k = train_y[inds_k]
            train_x_val = train_x[inds_val,:]
            train_y_val = train_y[inds_val]
        else:
            train_x_k = train_x
            train_y_k = train_y
            train_x_val = train_x
            train_y_val = train_y
        
        # print(type(n_batch))
        if torch.cuda.is_available():
            model_list[k] = model_list[k].cuda()
            likelihood_list[k] = likelihood_list[k].cuda()
            mll_list[k] = mll_list[k].cuda()
        
        optimizer = torch.optim.Adam([
            {'params': model_list[k].feature_extractor.parameters()},
            {'params': model_list[k].gp_layer.parameters()},
            {'params': likelihood_list[k].parameters()},
            {'params': model_list[k].vae_layer.parameters()},
        ], lr=0.05, weight_decay=1e-3)
        
        # for param in model_list[k].vae_layer.parameters():
        #     print(param)
        
        model_list[k].train()
        likelihood_list[k].train()
        mll_list[k].train()
        
        l_test = torch.nn.MSELoss()
        
        # Train dkl and gpr in unison
        n_steps=150
        train_dataset = TensorDataset(train_x_k, train_y_k)
        for i in range(n_steps):
            minibatch_iter = DataLoader(train_dataset, batch_size=n_batch, shuffle=False)
            for batch_x, batch_y in minibatch_iter:
                optimizer.zero_grad()
                if i < 5:
                    with gpt_settings.cholesky_jitter(1e-1),gpt_settings.fast_computations(covar_root_decomposition=False):
                        output, mu, logvar, x_lat = model_list[k](batch_x)
                else:
                    output, mu, logvar, x_lat = model_list[k](batch_x)
                    
                    # print(model.vae_layer.logvar.bias)
                    
                    # x_lat, _, _ = model.vae_layer(model.feature_extractor(batch_x))
                    # kl_div = kl_vae_qp(x_lat,mu,logvar)
                try:
                    mll_loss = -mll_list[k](output, batch_y)
                except ValueError:
                    for param in model_list[k].vae_layer.parameters():
                        print(param)
                    print()
                    
                loss = mll_loss # + lambda_kl*kl_div
                    # print(loss)
                loss.backward(retain_graph=True)
                loss=loss.item()
                    # if ((i + 1) % 5 == 0):
                        # print(f"{dn} Iter {i + 1}/{n_steps}: {loss}")
                        
                optimizer.step()
                
                
        # Fix dkl projection, train just gpr
        # No need for regularization, if previously used
        # Training on full dataset, could be adjusted for very large datasets
        optimizer = torch.optim.Adam([
            {'params': model_list[k].gp_layer.parameters()},
            {'params': likelihood_list[k].parameters()},
        ], lr=0.05)
        
        n_steps=150
        for i in range(n_steps):
            optimizer.zero_grad()
            output, mu, logvar, x_lat = model_list[k](train_x_k)
            loss = -mll_list[k](output, train_y_k)
            # print(loss)
            loss.backward(retain_graph=True)
            loss=loss.item()
            # if ((i + 1) % 10 == 0):
            #     print(f"Iter {i + 1}/{n_steps}: {loss}")
            optimizer.step()
        
        model_list[k].eval()
        likelihood_list[k].eval()
        output_test, mu_test, logvar_test, x_lattest = model_list[k](train_x_val)
        val_loss = np.sqrt(l_test(likelihood_list[k](output_test).mean.flatten(),train_y_val).item())
        val_data[k] = val_loss
        
    # Select best iteration of cross-validation
    k_min = np.argmin(val_data)
    model = model_list[k_min]
    likelihood = likelihood_list[k_min]
    
    
    # Test
    
    
    # torch.save(model.feature_extractor.state_dict(),dn+'/CV/dict_opt_fe_'+f_ext+'.pth')
    # torch.save(model.vae_layer.state_dict(),dn+'/CV/dict_opt_var_'+f_ext+'.pth')
    
    # Evaluates the learned model on the test and train data
    with torch.no_grad(), gpt_settings.fast_pred_var(),gpt_settings.max_cg_iterations(2500):
        # vdkl projection of test data
        _, test_vae, _ = model.vae_layer(model.feature_extractor(test_x))
        # gpr on test data
        model_test = model.gp_layer(test_vae)
        # posterior probability (predictive distribution)
        pos_pred_2 = likelihood(model_test)
        pos_train_mean = pos_pred_2.mean.flatten()
        
        # basic prediction results on test data
        # observed value, pred mean, pred stdev
        data_out = np.zeros((len(test_y),3))
        data_out[:,0] = (test_y+train_mean).cpu()
        data_out[:,1] = (pos_train_mean+train_mean).cpu()
        data_out[:,2] = pos_pred_2.stddev.flatten().cpu()
        # return data_out
        # np.savetxt("data/gpy/ham_homo_out/approx_fit_"+str(ind_d-5)+".csv",data_out,delimiter=',')
        # np.savetxt(dn+"/CV_test/vdkl_fit_opt_"+f_ext+".csv",data_out,delimiter=',')
        # np.savetxt(dn+"/vdkl_fit_"+str(n_batch)+".csv",data_out,delimiter=',')
        
        _, tex_eff, tex_sig = model.vae_layer(model.feature_extractor(test_x))
        tex_sig = torch.exp(.5*tex_sig)
        feat_effte = np.zeros((len(pos_train_mean),out_dim+1+3+2))
        # latent space position of test data
        feat_effte[:,:out_dim] = tex_eff.detach().cpu().numpy()
        # width in latent space 
        feat_effte[:,out_dim] = tex_sig.flatten().detach().cpu().numpy()
        # observed test energy
        feat_effte[:,out_dim+1] = test_y.detach().cpu().numpy()
        # pred mean
        feat_effte[:,out_dim+2] = pos_train_mean.detach().cpu().numpy()
        # pred noise
        feat_effte[:,out_dim+3] = pos_pred_2.stddev.detach().cpu().numpy()
        
        # variance decomposition on the test data
        #inducing point locations
        u = model.gp_layer.variational_strategy.inducing_points.detach()
        # variational distribution of the inducing points
        s_var = model.gp_layer.variational_strategy.variational_distribution._covar.evaluate().detach()
        k_uu = model.gp_layer.covar_module(u,u)
        k_xu = model.gp_layer.covar_module(u,tex_eff.detach())
        k_xx = model.gp_layer.covar_module(tex_eff.detach(),tex_eff.detach())
        k_uu_L = model.gp_layer.variational_strategy._cholesky_factor(k_uu)
        interp_term = k_uu_L.inv_matmul(k_xu.evaluate().double()).to(k_xu.dtype)
        # k_xx - k_xu k_uu^-1 k_ux
        pred_cov_1 = torch.diag(k_xx.evaluate() - interp_term.T@interp_term)
        # k_xu k_uu^-1 S k_uu^-1 k_ux
        pred_cov_2 = torch.diag(interp_term.T@s_var@interp_term)
        # np.maximum just in case of negative values (feat_effte was previously 0)
        feat_effte[:,out_dim+4] = np.sqrt(np.maximum(feat_effte[:,out_dim+4],pred_cov_1.detach().cpu().numpy()))
        feat_effte[:,out_dim+5] = np.sqrt(np.maximum(feat_effte[:,out_dim+5],pred_cov_2.detach().cpu().numpy()))
        
        np.savetxt(dn+"/"+dir_out+"/vdkl_feat_opt_test_"+f_ext+".csv",feat_effte,delimiter=',')
        # np.savetxt(dn+"/vdkl_feat_test_"+str(n_batch)+".csv",feat_effte,delimiter=',')
        # tex_args = torch.argsort(tex_eff.flatten())
        # tex_sorted = tex_eff.flatten()[tex_args]
        
        # Same stuff as before, but on the training data
        _, trx_eff, trx_sig = model.vae_layer(model.feature_extractor(train_x))
        trx_sig = torch.exp(.5*trx_sig)
        pos_pred_tr = likelihood(model.gp_layer(trx_eff))
        feat_efftr = np.zeros((len(train_y),out_dim+1+3+2))
        feat_efftr[:,:out_dim] = trx_eff.detach().cpu().numpy()
        feat_efftr[:,out_dim] = trx_sig.flatten().detach().cpu().numpy()
        feat_efftr[:,out_dim+1] = train_y.detach().cpu().numpy()
        feat_efftr[:,out_dim+2] = pos_pred_tr.mean.flatten().detach().cpu().numpy()
        feat_efftr[:,out_dim+3] = pos_pred_tr.stddev.detach().cpu().numpy()
        
        k_xu = model.gp_layer.covar_module(u,trx_eff.detach())
        k_xx = model.gp_layer.covar_module(trx_eff.detach(),trx_eff.detach())
        interp_term = k_uu_L.inv_matmul(k_xu.evaluate().double()).to(k_xu.dtype)
        pred_cov_1 = torch.diag(k_xx.evaluate() - interp_term.T@interp_term)
        pred_cov_2 = torch.diag(interp_term.T@s_var@interp_term)
        feat_efftr[:,out_dim+4] = np.sqrt(np.maximum(feat_efftr[:,out_dim+4],pred_cov_1.detach().cpu().numpy()))
        feat_efftr[:,out_dim+5] = np.sqrt(np.maximum(feat_efftr[:,out_dim+5],pred_cov_2.detach().cpu().numpy()))
        
        np.savetxt(dn+"/"+dir_out+"/vdkl_feat_opt_train_"+f_ext+".csv",feat_efftr,delimiter=',')
        # np.savetxt(dn+"/vdkl_feat_train_"+str(n_batch)+".csv",feat_efftr,delimiter=',')
        # trx_args = torch.argsort(trx_eff.flatten())
        # trx_sorted = trx_eff.flatten()[trx_args]
        
        # Do gmm analysis at a variety of target numbers of clusters
        # return all for more data
        for i in range(5):
            if i == 0:
                data_gmm = lat_gmm(20,out_dim,feat_efftr, feat_effte)
            else:
                data_gmm = np.concatenate((data_gmm,lat_gmm(20*(i+1),out_dim,feat_efftr, feat_effte)))
        # np.savetxt(dn+"/CV_test/gmm_"+f_ext+".csv",data_gmm,delimiter=',')
        
        
    return val_data, data_out, feat_effte, data_gmm


def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def vdkl_experiment(n_train, dn, fn_list, params, f_ext):
    
    # Loads the distance matrix data and normalizes the input vectors
    
    fname_x = dn+"/"+fn_list[0]
    fname_y = dn+"/"+fn_list[1]
    data_x = torch.Tensor(np.loadtxt(fname_x,delimiter=','))
    
    # standardize training data
    train_xmean = torch.mean(data_x,0)
    train_std = torch.std(data_x,0)
    data_x -= train_xmean
    data_x /= train_std
    data_y = torch.Tensor(np.loadtxt(fname_y))
    train_mean = torch.mean(data_y)
    data_y -= train_mean
    # train_perm is the random permutation to divide training and test data
    train_perm = np.random.permutation(len(data_y))
    
    train_x = data_x[train_perm[:n_train],:]
    train_y = data_y[train_perm[:n_train]]
    
    fname_xt = dn+"/"+fn_list[2]
    fname_yt = dn+"/"+fn_list[3]
    test_x = torch.Tensor(np.loadtxt(fname_xt,delimiter=','))
    test_y = torch.Tensor(np.loadtxt(fname_yt))
    
    # Apply same standardization to test set
    test_x -= train_xmean
    test_x /= train_std
    test_y -= train_mean
    
    # test_perm = np.random.permutation(len(data_y))
    # test_x = test_x[test_perm[:n_train],:]
    # test_y = test_y[test_perm[:n_train]]
    
    
    val_trial, data_trial, feat_trial, gmm_trial = vdkl_trial(train_x, train_y, test_x, test_y, params, dn, f_ext, train_mean)
    
    data_out = np.zeros(30)
    
    data_out[0] = np.sqrt(np.mean((data_trial[:,0]-data_trial[:,1])**2)) # rmse
    data_out[1] = np.std(np.abs(data_trial[:,0]-data_trial[:,1])) # stdev of ae
    data_out[2] = np.sqrt(np.mean(((data_trial[:,0]-data_trial[:,1])/data_trial[:,2])**2)) # rrmse
    data_out[3] = np.std(np.abs((data_trial[:,0]-data_trial[:,1])/data_trial[:,2])) # stdev of re
    data_out[4] = np.sqrt(np.mean(data_trial[:,2]**2)) # mean pred noise
    data_out[5] = np.std(data_trial[:,2]) # stdev of pred noise
    
    n_gmm = len(gmm_trial[:,2]) # total number of gmm clusters
        
    data_out[6] = np.sqrt(np.mean(gmm_trial[:,2]**2)) # mean of pred stdev
    data_out[7] = np.std(gmm_trial[:,2])/np.sqrt(n_gmm) # std err of mean of pred stdev
    data_out[8] = np.mean(gmm_trial[:,3]) # mean of std err of pred stdev
    data_out[9] = np.std(gmm_trial[:,2]) # stdev of pred stdev
    data_out[10] = np.std(gmm_trial[:,2])/np.sqrt(2*n_gmm) # std err of prev calc
    
    n_samp = 1000
    hetsked_samp = gmm_trial[:,2] + gmm_trial[:,3]*np.random.randn(n_samp,n_gmm) # sample each estimate of the noise
    hetsked_samp = np.std(hetsked_samp,axis=1)
    data_out[11] = np.mean(hetsked_samp) # bootstrap estimate of heteroskedasticity
    data_out[12] = np.std(hetsked_samp) # bootstrap error on heteroskedasticity
    
    data_out[13] = np.sqrt(np.mean((gmm_trial[:,0]-gmm_trial[:,4])**2)) # rmse : mean
    data_out[14] = np.std(np.abs(gmm_trial[:,0]-gmm_trial[:,4])) # stdev of abs mean errs
    data_out[15] = np.sqrt(np.mean((gmm_trial[:,2]-gmm_trial[:,6])**2)) # rmse : noise
    data_out[16] = np.std(np.abs(gmm_trial[:,2]-gmm_trial[:,6])) # stdev of abs noise errs
    data_out[17] = np.sqrt(np.mean(((gmm_trial[:,0]-gmm_trial[:,4])/gmm_trial[:,2])**2)) # rrmse : mean
    data_out[18] = np.std(np.abs((gmm_trial[:,0]-gmm_trial[:,4])/gmm_trial[:,2])) # stdev : abs rel mean
    data_out[19] = np.sqrt(np.mean(((gmm_trial[:,2]-gmm_trial[:,6])/gmm_trial[:,2])**2)) # rrmse : noise
    data_out[20] = np.std(np.abs((gmm_trial[:,2]-gmm_trial[:,6])/gmm_trial[:,2])) # stdev : abs rel noise
    
    data_out[21] = np.sqrt(np.mean(feat_trial[:,-3]**2)) # full noise
    data_out[22] = np.std(feat_trial[:,-3])
    data_out[23] = np.sqrt(np.mean(feat_trial[:,-2]**2)) # fit unc
    data_out[24] = np.std(feat_trial[:,-2])
    data_out[25] = np.sqrt(np.mean(feat_trial[:,-1]**2)) # s unc
    data_out[26] = np.std(feat_trial[:,-1])
    
    data_out[27] = np.mean(gmm_trial[:,8]) # mean cluster pop
    data_out[28] = np.std(gmm_trial[:,8])
    
    data_out[29] = n_gmm # number of clusters
    
    print(data_out)
    return data_out


torch.set_default_tensor_type(torch.cuda.FloatTensor)
#torch.set_default_tensor_type(torch.FloatTensor)

big_dir = 'Example'
# these files should be contained in big_dir
fn_data = ['dist_train.csv','homo_train.txt','dist_test.csv','homo_test.txt']

# batch size, latent feature size, number of cross-validation steps, kl regularization term
# regularization currently turned off in vdkl_experiment
param_job = np.array([500,6,5,1e-4])
# vdkl_experiment saves various files to a further subdirectory big_dir/CV_test
data_full = vdkl_experiment(9900, big_dir, fn_data, param_job,'_ur')
np.savetxt(big_dir+'/CV_test/data.csv', data_full, delimiter=',')


input("Press Enter to continue...")
