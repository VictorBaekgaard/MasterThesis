import generate_data, visualize, loss_functions, algorithms
import torch


d               = 2
n_samples       = 1000
inner_loop_size = 30
n_iter          = 300
beta1, beta2    = 0.9, 0.99 


X, y, sc_theta_true = generate_data.generate_strongly_convex_data(n_samples=n_samples, n_features=d, noise_std=0.1, condition_number=10, random_state=302)
sc_data = list(zip(X, y))

func_args = {'lambda_reg': 0.01}
theta_init = torch.zeros_like(sc_theta_true)

lr_svrg  = 0.02
lr_ada   = 0.1
lr_adam  = 0.02

svrg_sc_res = algorithms.SVRG(
            func=loss_functions.ridge_regression_loss,
            lr=lr_svrg,
            n_epochs=10,
            inner_loop_size=inner_loop_size,
            data=sc_data,
            theta_true=sc_theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args
        )

ada_sc_res = algorithms.AdaGrad(
            func=loss_functions.ridge_regression_loss,
            lr=lr_ada,
            n_iter=n_iter,
            data=sc_data,
            theta_true=sc_theta_true,
            theta_init=theta_init.clone(),
            func_args=func_args
        )

adam_sc_res =  algorithms.Adam(
            func=loss_functions.ridge_regression_loss,
            lr=lr_adam,
            beta_1=beta1,
            beta_2=beta2,
            n_iter=n_iter,
            data=sc_data,
            theta_true=sc_theta_true,
            theta_init=theta_init.clone(),
            return_moments=False,  # Make sure your Adam implementation returns moment estimates
            func_args=func_args
        )

visualize.visualize_optimization(sc_data, svrg_sc_res[2], sc_theta_true, loss_functions.absolute_loss, "SVRG Strongly Convex")
visualize.visualize_optimization(sc_data, ada_sc_res[2], sc_theta_true, loss_functions.absolute_loss, "AdaGrad Strongly Convex")
visualize.visualize_optimization(sc_data, adam_sc_res[2], sc_theta_true, loss_functions.absolute_loss, "Adam Strongly Convex")