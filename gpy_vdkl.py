import os
# os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import torch
import gpytorch
from gpytorch import settings as gpt_settings
from torch.utils.data import DataLoader, TensorDataset
from gpytorch.constraints import GreaterThan, Interval

# Generates data using a toy model function
def gen_data(n_train, dim_d, phi, sig_mult):
    data_x = torch.rand((n_train,dim_d))*2-1
    sig_d = 1/np.sqrt(2.0*dim_d)
    data_y_noise = torch.zeros((n_train,1))
    data_y_noise[:,0] = torch.prod(1 + sig_d*torch.sin(np.pi*data_x),axis=1)*sig_mult
    data_y = torch.randn((n_train,1))*data_y_noise
    
    data_y_test = torch.zeros((n_train,1))
    sig_f = np.sqrt(2/dim_d)
    # sig_f = 1
    data_y_test[:,0] = torch.sum(sig_f*torch.sin(np.pi*data_x + phi),axis=1)
    # print(torch.var(data_y_test))
    return data_x, (data_y+data_y_test).flatten(), data_y_test.flatten(), data_y_noise.flatten()


# First neural network layer
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
        # self.add_module('relu5', torch.nn.Sigmoid())

# Variational layer
class VAELayer(torch.nn.Module):
    def __init__(self, data_dim):
        super(VAELayer, self).__init__()
        
        self.mu = torch.nn.Linear(data_dim, data_dim)
        self.logvar = torch.nn.Linear(data_dim, 1)
        # self.register_parameter(name='logvar_lazy', param=torch.nn.Parameter(torch.ones(1)*-7.37373))
    
    # Layer between the feature extraction and the gpr
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
                self, torch.rand((grid_size,num_dim)),
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



class DKLModel(gpytorch.Module):
    def __init__(self, in_dim, num_dim, grid_size=100):
        super(DKLModel, self).__init__()
        self.feature_extractor = LargeFeatureExtractor(in_dim, num_dim)
        self.gp_layer = ApproximateGPLayer(num_dim=num_dim, grid_size=grid_size)
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)
        return res


class DKLVAEModel(gpytorch.Module):
    def __init__(self, in_dim, num_dim, grid_size=100):
        super(DKLVAEModel, self).__init__()
        self.feature_extractor = LargeFeatureExtractor(in_dim, num_dim)
        self.vae_layer = VAELayer(num_dim)
        self.gp_layer = ApproximateGPLayer(num_dim=num_dim, grid_size=grid_size)
        self.num_dim = num_dim

    def forward(self, x):
        lat = self.feature_extractor(x)
        feat, mu, logvar = self.vae_layer(lat)
        res = self.gp_layer(feat)
        return res, mu, logvar

# This is the used KL divergence function 
def kl_vae_qp(x, mu, logvar):
    var = torch.exp(logvar[:,0])
    dim_out = len(x[0])
    diff_tensor = torch.zeros((dim_out,len(x[:,0]),len(mu[:,0])))
    
    # Squared difference tensor between the given sample points x and the center of the given distributions
    for i in range(dim_out):
        diff_tensor[i] = (x[:,i:(i+1)] - mu[:,i:(i+1)].T)**2
    
    # Applies the exponential to the distance tensor
    diff_tensor = torch.exp(-.5*torch.sum(diff_tensor,0)/var)/var**(dim_out/2)
    
    #Sums the exponentials over the batch and normalizes
    pdf_x = torch.sum(diff_tensor,1)/len(x[:,0])
    # Returns the batch sum, which is the approximate KL divergence 
    return torch.sum(torch.log( pdf_x / torch.exp(-.5*torch.sum(x**2,1)) ))

#Switches distributions from the previous function
def kl_vae_pq(x, mu, logvar):
    var = torch.exp(logvar[:,0])
    dim_out = len(x[0])
    diff_tensor = torch.zeros((dim_out,len(x[:,0]),len(mu[:,0])))
    for i in range(dim_out):
        diff_tensor[i] = (x[:,i:(i+1)] - mu[:,i:(i+1)].T)**2
    
    
    diff_tensor = torch.exp(-.5*torch.sum(diff_tensor,0)/var)/var**(dim_out/2)
    
    # print(mu)
    pdf_x = torch.sum(diff_tensor,1)/len(x[:,0])
    return torch.sum(torch.log(torch.exp(-.5*torch.sum(x**2,1)) / pdf_x ))




def train_and_test_approximate_gp(n_train, dn, dih_flag):
    
    # Loads the distance matrix data and normalizes the input vectors
    
    fname_x = dn+"/dist.csv"
    fname_y = dn+"/homo.txt"
    data_x = torch.Tensor(np.loadtxt(fname_x,delimiter=','))
    train_xmean = torch.mean(data_x,0)
    train_std = torch.std(data_x,0)
    data_x -= train_xmean
    data_x /= train_std
    data_y = torch.Tensor(np.loadtxt(fname_y))
    train_mean = torch.mean(data_y)
    data_y -= train_mean
    # train_perm is the random permutation to divide training and test data
    # I've been using a fixed permutation to more simply plot data against other variables like dihedral angles
    train_perm = range(len(data_y))
    train_x = data_x[train_perm[:n_train],:]
    train_y = data_y[train_perm[:n_train]]
    
    dim_d = len(train_x[0,:])
    test_x = data_x[train_perm[n_train:],:]
    test_target = data_y[train_perm[n_train:]]
    # test_target /= train_std
    test_noise = torch.ones(len(test_target))*torch.std(test_target)
    
    
    # Sets up the fixed dihedral data, if desired
    if dih_flag == 1:
        test_x2 = torch.zeros((8,500,dim_d))
        test_y2 = torch.zeros((8,500))
        ang_list = ['0','45','90','135','180','225','270','315']
        for i in range(8):
            test_x2[i,:,:] = torch.Tensor(np.loadtxt(dn+"/fixdih_"+ang_list[i]+"/dist.csv",delimiter=','))
            test_x2[i,:,:] -= train_xmean
            test_x2[i,:,:] /= train_std
            test_y2[i,:] = torch.Tensor(np.loadtxt(dn+"/fixdih_"+ang_list[i]+"/homo.txt"))
            test_y2[i,:] -= train_mean
    
    # Dimension of the latent space and number of inducing points for the GP kernel
    out_dim = 2
    n_induc = 1000
    
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = DKLModel(dim_d,out_dim,grid_size=n_induc)
    model = DKLVAEModel(dim_d,out_dim,grid_size=n_induc)
    mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model.gp_layer, num_data=train_y.numel())
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        mll = mll.cuda()
    
    # model.gp_layer.covar_module.base_kernel.lengthscale = .2*torch.ones((1,out_dim))
    # model.gp_layer.covar_module.base_kernel.raw_lengthscale_constraint = Interval(0.1,2)
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.gp_layer.parameters()},
        {'params': likelihood.parameters()},
        {'params': model.vae_layer.parameters()},
    ], lr=0.005)
    
    # This starts the latent distributions at a much smaller value than the converged value
    with torch.no_grad():
        print(model.vae_layer.logvar.bias)
        model.vae_layer.logvar.bias.data = -5*torch.nn.Parameter(torch.ones_like(model.vae_layer.logvar.bias))
        
    
    model.train()
    likelihood.train()
    mll.train()
    
    n_steps=1000
    n_batch = 500
    train_dataset = TensorDataset(train_x, train_y)
    for i in range(n_steps):
        minibatch_iter = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        # print(model.vae_layer.logvar_lazy)
        for batch_x, batch_y in minibatch_iter:
            optimizer.zero_grad()
            output, mu, logvar = model(batch_x)
            # print(model.vae_layer.logvar.bias)
            x_lat, _, _ = model.vae_layer(model.feature_extractor(batch_x))
            kl_div = kl_vae_qp(x_lat,mu,logvar)
            # I've been playing around with the value of the kl hyperparameter
            # .001 results in pretty strong enforcement
            # .0001 shows clear deviations from the desired prior distribution
            loss = -mll(output, batch_y) + .0001*kl_div
            # print(loss)
            loss.backward(retain_graph=True)
            loss=loss.item()
            if ((i + 1) % 10 == 0):
                    print(f"Iter {i + 1}/{n_steps}: {loss}")
                    print(model.gp_layer.covar_module.base_kernel.lengthscale.detach())
                    print(np.sqrt(likelihood.noise_covar.noise.item()))
                    print(torch.mean(torch.exp(0.5*logvar)))
            optimizer.step()
    
    
    # n_steps=1000
    # n_batch = 100
    # train_dataset = TensorDataset(train_x, train_y)
    # for i in range(n_steps):
    #     minibatch_iter = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
    #     # print(model.vae_layer.logvar_lazy)
    #     for batch_x, batch_y in minibatch_iter:
    #         optimizer.zero_grad()
    #         output, mu, logvar = model(batch_x)
    #         # print(model.vae_layer.logvar.bias)
    #         x_samp = torch.randn((n_batch,out_dim))
    #         kl_div = kl_vae_pq(x_samp,mu,logvar)
    #         # print(torch.mean(mu,0))
    #         # print(torch.std(mu,0))
    #         loss = -mll(output, batch_y) + .001*kl_div
    #         # print(loss)
    #         loss.backward(retain_graph=True)
    #         loss=loss.item()
    #         if ((i + 1) % 10 == 0):
    #                 print(f"Iter {i + 1}/{n_steps}: {loss}")
    #                 print(model.gp_layer.covar_module.base_kernel.lengthscale.detach())
    #                 print(model.gp_layer.covar_module.outputscale.detach())
    #                 print(np.sqrt(likelihood.noise_covar.noise.item()))
    #                 print(torch.mean(torch.exp(0.5*logvar)))
    #         optimizer.step()
    
    
    
    
    
    # Test
    model.eval()
    likelihood.eval()
    
    torch.save(model.feature_extractor.state_dict(),dn+'/vdkl_fe_dict.pth')
    
    # Evaluates the learned model on the test and train data, as well as fixed dihedral data
    with torch.no_grad(), gpt_settings.fast_pred_var(),gpt_settings.max_cg_iterations(2500):
        _, test_vae, _ = model.vae_layer(model.feature_extractor(test_x))
        model_test = model.gp_layer(test_vae)
        pos_pred_2 = likelihood(model_test)
        test_noise_err = torch.abs(test_noise-pos_pred_2.stddev).detach()
        pos_train_mean = pos_pred_2.mean.flatten()
        test_err = torch.abs(test_target-pos_train_mean).detach()
    
        kl_div = torch.log((pos_pred_2.stddev/test_noise).detach())+ ((test_noise**2+test_err**2)/(2*pos_pred_2.stddev**2)).detach()-.5
        
        data_out = np.zeros((len(test_target),4))
        data_out[:,0] = (test_target+train_mean).cpu()
        data_out[:,1] = test_noise.cpu()
        data_out[:,2] = (pos_train_mean+train_mean).cpu()
        data_out[:,3] = pos_pred_2.stddev.flatten().cpu()
        # return data_out
        # np.savetxt("data/gpy/ham_homo_out/approx_fit_"+str(ind_d-5)+".csv",data_out,delimiter=',')
        np.savetxt(dn+"/vdkl_fit.csv",data_out,delimiter=',')
        # np.savetxt(dn+"/vdkl_fit_"+str(n_batch)+".csv",data_out,delimiter=',')
        
        _, tex_eff, tex_sig = model.vae_layer(model.feature_extractor(test_x))
        tex_sig = torch.exp(.5*tex_sig)
        feat_effte = np.zeros((len(pos_train_mean),out_dim+1+3))
        feat_effte[:,:out_dim] = tex_eff.detach().cpu().numpy()
        feat_effte[:,out_dim] = tex_sig.flatten().detach().cpu().numpy()
        feat_effte[:,out_dim+1] = test_target.detach().cpu().numpy()
        feat_effte[:,out_dim+2] = pos_train_mean.detach().cpu().numpy()
        feat_effte[:,out_dim+3] = pos_pred_2.stddev.detach().cpu().numpy()
        np.savetxt(dn+"/vdkl_feat_test.csv",feat_effte,delimiter=',')
        # np.savetxt(dn+"/vdkl_feat_test_"+str(n_batch)+".csv",feat_effte,delimiter=',')
        # tex_args = torch.argsort(tex_eff.flatten())
        # tex_sorted = tex_eff.flatten()[tex_args]
        
        _, trx_eff, trx_sig = model.vae_layer(model.feature_extractor(train_x))
        trx_sig = torch.exp(.5*trx_sig)
        pos_pred_tr = likelihood(model.gp_layer(trx_eff))
        feat_efftr = np.zeros((len(train_y),out_dim+1+3))
        feat_efftr[:,:out_dim] = trx_eff.detach().cpu().numpy()
        feat_efftr[:,out_dim] = trx_sig.flatten().detach().cpu().numpy()
        feat_efftr[:,out_dim+1] = train_y.detach().cpu().numpy()
        feat_efftr[:,out_dim+2] = pos_pred_tr.mean.flatten().detach().cpu().numpy()
        feat_efftr[:,out_dim+3] = pos_pred_tr.stddev.detach().cpu().numpy()
        np.savetxt(dn+"/vdkl_feat_train.csv",feat_efftr,delimiter=',')
        # np.savetxt(dn+"/vdkl_feat_train_"+str(n_batch)+".csv",feat_efftr,delimiter=',')
        # trx_args = torch.argsort(trx_eff.flatten())
        # trx_sorted = trx_eff.flatten()[trx_args]
        
        if dih_flag == 1:
            for i in range(8):
                _, tex_eff2, tex_sig2 = model.vae_layer(model.feature_extractor(test_x2[i]))
                tex_sig2 = torch.exp(.5*tex_sig2)
                pos_pred_te2 = likelihood(model.gp_layer(tex_eff2))
                feat_effte2 = np.zeros((len(test_y2[i]),out_dim+1+3))
                feat_effte2[:,:out_dim] = tex_eff2.detach().cpu().numpy()
                feat_effte2[:,out_dim] = tex_sig2.flatten().detach().cpu().numpy()
                feat_effte2[:,out_dim+1] = test_y2[i].detach().cpu().numpy()
                feat_effte2[:,out_dim+2] = pos_pred_te2.mean.flatten().detach().cpu().numpy()
                feat_effte2[:,out_dim+3] = pos_pred_te2.stddev.flatten().detach().cpu().numpy()
                np.savetxt(dn+"/fixdih_"+ang_list[i]+"/vdkl_feat_test.csv",feat_effte2,delimiter=',')
        
    return data_out, np.array([torch.mean(pos_pred_2.mean).item(),torch.std(pos_pred_2.mean).item(), 
                      torch.mean(test_err).item(),torch.std(test_err).item(),
                      torch.mean(pos_pred_2.stddev).item(),torch.std(pos_pred_2.stddev).item(),
                      torch.mean(test_noise_err).item(),torch.std(test_noise_err).item(),
                      torch.mean(kl_div).item(),torch.std(kl_div).item()])



    
    
# train_and_test_approximate_gp(gpytorch.mlls.VariationalELBO)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# dtemp, ftemp = train_and_test_nn_regress(8000)
# print(ftemp)

dtemp, ftemp = train_and_test_approximate_gp(8000,'16_bead',1)
print(ftemp)






input("Press Enter to continue...")
