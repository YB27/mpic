import numpy as np
import scipy as scp
import scipy.special
import scipy.stats


def expectationPsi_closeForm(params):
    psi_mean = params["psi"]["mean"]
    psi_var = params["psi"]["std"]**2

    exp_half_var_minus = np.exp(-0.5*psi_var)
    exp_double_var_minus = np.exp(-2.*psi_var)
    cos_2_psi = np.cos(2.*psi_mean)

    e_cos = np.cos(psi_mean)*exp_half_var_minus
    e_sin = np.sin(psi_mean)*exp_half_var_minus
    e_csqr = 0.5*(1. + exp_double_var_minus*cos_2_psi)
    e_ssqr = 0.5*(1. - exp_double_var_minus*cos_2_psi)
    e_cs = 0.5*exp_double_var_minus*np.sin(2.*psi_mean)

    return e_cos, e_sin, e_cs, e_csqr, e_ssqr

''' Compute expectation relative to theta in closed-form'''
def expectationTheta_closeForm(params, n_approx):
    alpha = params['theta']['alpha']
    beta = params['theta']['beta']
    b = params['theta']['b']
    half_b = 0.5 * b

    ''' Compute the hypergeometric function terms F_12(-2n,alpha, alpha+beta;2) '''
    F_12_cos = scp.special.hyp2f1(-2. * np.arange(0, n_approx, step=1), alpha, alpha + beta, 2)

    ''' Compute the hypergeometric function terms F_12(-2n-1,alpha, alpha+beta;2) '''
    F_12_sin = scp.special.hyp2f1(-2. * np.arange(0, n_approx, step=1) - np.ones((n_approx,)), alpha, alpha + beta, 2)

    ''' Compute expectation relative to theta'''
    expectation_cos_theta = 0
    expectation_sin_theta = 0
    expectation_cos_sin_theta = 0
    expectation_cos_sqr_theta = 1.
    for i in range(0, n_approx):
        if (i % 2 == 0):
            sign = 1
        else:
            sign = -1
        a_pair = sign / np.math.factorial(2. * i)
        a_impair = sign / np.math.factorial(2. * i + 1.)
        c = half_b ** (2. * i)
        d = b ** (2 * i)
        expectation_cos_theta += a_pair * c * F_12_cos[i]
        expectation_sin_theta += a_impair * c * F_12_sin[i]
        expectation_cos_sin_theta += a_impair * d * F_12_sin[i]
        if(i > 0):
            expectation_cos_sqr_theta += 0.5 * a_pair * d * F_12_cos[i]

    expectation_sin_theta *= -half_b
    expectation_cos_sin_theta *= -half_b
    expectation_sin_sqr_theta = 1. - expectation_cos_sqr_theta

    return expectation_cos_theta, expectation_sin_theta, \
           expectation_cos_sin_theta, expectation_cos_sqr_theta,\
           expectation_sin_sqr_theta

''' Approximate the local cartesian coords by a Gaussian using closed-form expressions '''
def approximateToGaussian_closedForm(params):
    n_approx = 5 # Number of terms in the series to take
    e_c_t, e_s_t, e_cs_t, e_csqr_t, e_ssqr_t = expectationTheta_closeForm(params, n_approx)
    e_c_p, e_s_p, e_cs_p, e_csqr_p, e_ssqr_p = expectationPsi_closeForm(params)

    rho_mean = params["rho"]["mean"]
    rho_var = params["rho"]["std"]**2
    mean = rho_mean*np.array([e_c_t * e_c_p,
                              e_c_t * e_s_p,
                              e_s_t])

    rho_mean_sqr = rho_mean**2
    e_rho_sqr = rho_mean_sqr + rho_var
    e_c_t_sqr = e_c_t**2
    sigma_xx = e_rho_sqr*e_csqr_p*e_csqr_t - rho_mean_sqr*(e_c_p**2)*e_c_t_sqr
    sigma_yy = e_rho_sqr*e_ssqr_p*e_csqr_t - rho_mean_sqr*(e_s_p**2)*e_c_t_sqr
    sigma_zz = e_rho_sqr*e_ssqr_t - rho_mean_sqr*(e_s_t**2)
    sigma_xy = e_rho_sqr*e_cs_p*e_csqr_t - rho_mean_sqr*e_c_p*e_s_p*e_c_t_sqr
    sigma_xz = e_rho_sqr*e_c_p*e_cs_t - rho_mean_sqr*e_c_p*e_c_t*e_s_t
    sigma_yz = e_rho_sqr*e_s_p*e_cs_t - rho_mean_sqr*e_s_p*e_c_t*e_s_t
    cov = np.array([[sigma_xx, sigma_xy, sigma_xz],
                    [sigma_xy, sigma_yy, sigma_yz],
                    [sigma_xz, sigma_yz, sigma_zz]
                   ])

    return {'mean': mean, 'cov': cov}
