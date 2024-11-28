import generate_data, visualize, loss_functions, algorithms
import torch


d               = 2
n_samples       = 1000
inner_loop_size = 30
n_iter          = 300
beta1, beta2    = 0.9, 0.99 

convex_data, convex_theta_true = generate_data.generate_convex_data(N=n_samples, d=d, random_state=302)
X, y, sc_theta_true = generate_data.generate_strongly_convex_data(n_samples=n_samples, n_features=d, noise_std=0.1, condition_number=10, random_state=302)
sc_data = list(zip(X, y))


theta_init = torch.zeros_like(convex_theta_true)

lr_svrg  = 0.01
lr_ada   = 0.1
lr_adam  = 0.01

svrg_convex_res = algorithms.SVRG(
            func=loss_functions.absolute_loss,
            lr=lr_svrg,
            n_epochs=10,
            inner_loop_size=inner_loop_size,
            data=convex_data,
            theta_true=convex_theta_true,
            theta_init=theta_init.clone(),
        )

ada_convex_res = algorithms.AdaGrad(
            func=loss_functions.absolute_loss,
            lr=lr_ada,
            n_iter=n_iter,
            data=convex_data,
            theta_true=convex_theta_true,
            theta_init=theta_init.clone()
        )

adam_convex_res =  algorithms.Adam(
            func=loss_functions.absolute_loss,
            lr=lr_adam,
            beta_1=beta1,
            beta_2=beta2,
            n_iter=n_iter,
            data=convex_data,
            theta_true=convex_theta_true,
            theta_init=theta_init.clone(),
            return_moments=False 
        )

visualize.visualize_optimization(convex_data, svrg_convex_res[2], convex_theta_true, loss_functions.absolute_loss, "SVRG Convex")
visualize.visualize_optimization(convex_data, ada_convex_res[2], convex_theta_true, loss_functions.absolute_loss, "AdaGrad Convex")
visualize.visualize_optimization(convex_data, adam_convex_res[2], convex_theta_true, loss_functions.absolute_loss, "Adam Convex")