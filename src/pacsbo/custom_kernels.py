import torch
import gpytorch
from gpytorch.kernels import Kernel
import warnings
import numpy as np
from linear_operator.operators import DenseLinearOperator




class BrowianMotionKernel(gpytorch.kernels.Kernel):
    def __init__(self, active_dims, a_parameter, **kwargs):
        super().__init__(active_dims=active_dims, **kwargs)
        self.a_parameter = a_parameter

    def forward(self, t1, t2, **params):
        K = torch.minimum(t1, t2.T)/self.a_parameter
        # return gpytorch.lazy.NonLazyTensor(K)
        return DenseLinearOperator(K)

class ReverseBrownianMotionKernel(gpytorch.kernels.Kernel):
    def __init__(self, active_dims, a_parameter, **kwargs):
        super().__init__(active_dims=active_dims, **kwargs)
        self.a_parameter = a_parameter

    def forward(self, t1, t2, **params):
        K = torch.minimum(self.a_parameter-t1, self.a_parameter-t2.T)/self.a_parameter
        # return gpytorch.lazy.NonLazyTensor(K)
        return DenseLinearOperator(K)


class Matern12_RBF_WeightedSumKernel(gpytorch.kernels.Kernel):
    is_stationary = False  # Since 'a' is input-dependent, the kernel is non-stationary

    def __init__(self, active_dims, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, output_variance_Ma12, a_parameter, **kwargs):
        super().__init__(**kwargs)
        self.a_parameter = a_parameter
        self.rbf = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=active_dims)#,
            # outputscale=output_variance_RBF
        )
        self.rbf.base_kernel.lengthscale = lengthscale_temporal_RBF
        self.rbf.outputscale = output_variance_RBF  # Ensure this is correctly set
        self.rbf.base_kernel.raw_lengthscale.requires_grad = False
        self.rbf.raw_outputscale.requires_grad = False

        self.matern12 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1/2, active_dims=active_dims),
        )
        self.matern12.base_kernel.lengthscale = lengthscale_temporal_Ma12
        self.matern12.outputscale = output_variance_Ma12
        self.matern12.base_kernel.raw_lengthscale.requires_grad = False
        self.matern12.raw_outputscale.requires_grad = False

        # self.a_kernel = A_Kernel(a_parameter)
        # self.a_neg_kernel = (a_parameter)
        # self.periodic_kernel = MyPeriodicKernel(period=30, lengthscale=0.5)
        # self.rq_kernel = MyRQKernel(lengthscale=25.0, alpha=2.0)
        self.bm_kernel = BrowianMotionKernel(active_dims=active_dims, a_parameter=self.a_parameter)
        self.rbm_kernel = ReverseBrownianMotionKernel(active_dims=active_dims, a_parameter=self.a_parameter)

        self.bm_rbm_product = gpytorch.kernels.ProductKernel(self.rbm_kernel, self.bm_kernel)
        self.RBF_part = self.rbf
        # self.Ma12_part = gpytorch.kernels.ProductKernel(self.matern12, self.bm_kernel)
        self.Ma12_part = gpytorch.kernels.ProductKernel(self.matern12, self.bm_rbm_product)
        # self.temporal_kernel = self.RBF_part + self.Ma12_part
        # self.temporal_kernel = self.RBF_part + self.Ma12_part
        temporal_kernel = self.RBF_part + self.Ma12_part
        self.temporal_kernel = temporal_kernel

    def forward(self, x1, x2, **params):
        return self.temporal_kernel(x1, x2)  # self.temporal_kernel(x1, x2)


if __name__ == '__main__':
    # This is to plot covariances
    iterations = 50
    lengthscale = 10**-10
    T, T_prime = np.meshgrid(np.linspace(1, iterations, 1000), np.linspace(1, iterations, 1000))  # we need linspace for plotting and not range because of interpolation
    ell_matrix = np.ones_like(T)*lengthscale  # like this is constant
    for i in range(ell_matrix.shape[0]):
        for j in range(ell_matrix.shape[1]):
            ell_matrix[i, j] = (0.5*max(T[i,j], T_prime[i,j]))  # 10**(-6+int(max(T[i,j], T_prime[i,j])))
    K = custom_kernel(T, T_prime, ell=ell_matrix)
    plt.figure()
    plt.contourf(T, T_prime, K, levels=50, cmap='viridis')
    plt.colorbar(label=r'$k(t, t^\prime)$')
    plt.xlabel('$t$')
    plt.ylabel(r'$t^\prime$')
    plt.title(rf'Kernel $k(t, t^\prime)$ with time-varying $\ell$')
    plt.show()
    plt.savefig(f'custom_kernel_ell_timevarying_iterations_{iterations}.png')


    plt.figure()
    plt.plot(range(len(ell_matrix[:,0])), ell_matrix[:,0])
    plt.yscale('log')
    plt.xlabel('Iteration $t$')
    plt.ylabel(r'$\ell(t)$')
    plt.savefig('time_varying_lengthscale.png')
