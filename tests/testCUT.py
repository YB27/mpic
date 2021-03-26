''' Test the Conjugate Unscented Transform (CUT) method detailed in
"Conjugate unscented transformation: Applications to estimation and control, Adurthi, Nagavenkat and Singla, Puneet and Singh, Tarunraj, 2018"
in the context of ICP for non-gaussian uncertainty
'''

import time
import numpy as np
import scipy.linalg.matfuncs
import scipy.spatial.transform
import scipy.stats
import scipy.integrate
import numba as nb
import ctypes
from itertools import combinations, product
import matplotlib as plot


''' Values '''
rho_mean     = 5
rho_variance = 0.2
b = np.deg2rad(35.)
theta_mean = np.deg2rad(13)
theta_var = 0.001
theta_k = (b**2 - 4.*(theta_mean**2 + theta_var))/(8.*b*theta_var)
theta_alpha = (b + 2.*theta_mean)*theta_k
theta_beta  = (b - 2.*theta_mean)*theta_k
print("theta_alpha,beta : " + str(theta_alpha) + "," + str(theta_beta))
psi_mean = np.deg2rad(25)
psi_variance = 0.01**2

def beta_mode(alpha,beta,b):
    if(alpha > 1 and beta > 1):
        res = 0.5*b*(alpha - beta)/(alpha + beta -2)
    elif(alpha <= 1):
        res = -0.5*b
    elif(beta <= 1):
        res = 0.5*b
    elif(alpha < 1 and beta < 1):
        print("Error : theta distribution is bimodal and should not happen !")
    return res

theta_mode = beta_mode(theta_alpha, theta_beta, b)

''' 
List of ranges and weights for n=6 : [w_1, w_2, ... ]
Directly taken from the paper tables 

Note : w_0 value are not explicitly given for the cut-6 and cut-8 cases
For cut-4 :
    It is 0 (for n > 2, here n = 6)
For cut-6 :
    Just compute it using the equation (42)
For cut-8 :
    Generate the same formula from E(x^2_i) = 1 which gives
    w_o = 1 - 2n*w_1 - 2^n*w_2 - 2n(n-1)*w_3 - 2^n*w_4 - n_1*w_5 - n*2^n*w_6
    avec n_1 = 4n(n-1)(n-2)/3

'''
cut4_6_w = [0., 0.0625      , 0.00390625]
cut4_6_r = [2.           , np.sqrt(2)]
cut6_6_w = [0.06746372199999998, 0.0365072564, 0.0069487173, 0.0008288549]
cut6_6_r = [1.9488352799, 1.1445968942, 2.9068006056]
cut8_6_w = [0.08827161775999999, 0.0061728395, 0.0069134430, 0.0041152263, 0.0002183265,  0.00065104166, 0.00007849171]
cut8_6_r = [2.4494897427,  0.8938246941221211,  1.7320508075,  1.531963037906212, 2, 1.0954451150]

''' Generate an array of array corresponding to the conjugate axes c^6 for n=6. Used for all CUT '''
def generate_c_6(index,current_c,list_c):
    if (index > 5):
        list_c.append(current_c.copy())
    else:
        new_c_p = current_c.copy()
        new_c_p[index] = 1
        generate_c_6(index + 1, new_c_p, list_c)
        new_c_m = current_c.copy()
        new_c_m[index] = -1
        generate_c_6(index + 1, new_c_m, list_c)

''' Generate an array of array corresponding to the conjugate axes c^2 for n=6. Used for CUT-6 and CUT-8 '''
def generate_c_n(n):
    list_c = []
    non_zero_indexes = list(combinations([0,1,2,3,4,5],n))
    values = list(product([1,-1],repeat=n))
    for indexes in non_zero_indexes:
        current_c = np.array([0., 0., 0., 0., 0., 0.])
        for val in values:
            j = 0
            for i in indexes:
                current_c[i] = val[j]
                j = j + 1
            list_c.append(current_c.copy())

    return list_c

list_c_6 = []
generate_c_6(0,np.array([0.,0.,0.,0.,0.,0.]), list_c_6)
list_c_2 = generate_c_n(2)
list_c_3 = generate_c_n(3)

''' Generate an array of array corresponding to the scaled conjugate axes s^6 for n=6. Used for CUT-8 '''
def generate_s_6():
    global list_c_6
    h = 3
    list_s = []
    for c in list_c_6:
        for i in range(0,6):
            s = c.copy()
            s[i] = h*s[i]
            list_s.append(s)

    return list_s

list_s_6 = generate_s_6()

''' List of samples x = [x_sigma, x_c] for cut4 and n=6 '''
def generate_cut4_6_x(cut_6_r=cut4_6_r):
    global list_c_6
    x = []

    ''' Center '''
    x.append(np.array([0., 0., 0., 0., 0., 0.]))

    ''' For x_sigma (principale axes)'''
    for i in range(0,6):
        x_sample = np.array([0.,0.,0.,0.,0.,0.])
        x_sample[i] = cut_6_r[0]
        print(x_sample)
        x.append(x_sample.copy())
        x_sample[i] = -cut_6_r[0]
        x.append(x_sample.copy())

    ''' For x_c (conjugated axes) '''
    for c in list_c_6:
        x.append(cut_6_r[1]*np.array(c))

    return x

''' List of samples x = [x_sigma, x_c] for cut6 and n=6 '''
def generate_cut6_6_x(cut_6_r=cut6_6_r):
    global list_c_2

    x = []

    ''' The first 2n + 2**n elements are the same (with different r values) '''
    x.extend(generate_cut4_6_x(cut_6_r))

    for c in list_c_2:
        x.append(cut_6_r[2]*c)

    return x

''' List of samples x = [x_sigma, x_c, x_s] for cut8 and n=6 '''
def generate_cut8_6_x(cut_6_r= cut8_6_r):
    global list_c_3, list_c_6, list_s_6

    ''' The first 2n + 2**n elements are the same (with different r values) '''
    x = generate_cut6_6_x(cut_6_r)

    for c in list_c_6:
        x.append(cut_6_r[3]*c)

    for c in list_c_3:
        x.append(cut_6_r[4]*c)

    for s in list_s_6:
        x.append(cut_6_r[5]*s)

    return x

''' q + x '''
def compose(q,x):
    rot = scipy.spatial.transform.Rotation.from_euler('ZYX',q[0:3])
    x_composed = rot.apply(x) + q[3:6]
    return x_composed

''' x - q'''
def inverse_compose(q,x):
    rot_T = scipy.spatial.transform.Rotation.from_euler('ZYX',q[0:3]).inv()
    x_composed = rot_T.apply(x - q[3:6])
    return x_composed

''' g '''
def fromSphericalToCart(x):
    cos_theta = np.cos(x[1])
    sin_theta = np.sin(x[1])
    cos_phi = np.cos(x[2])
    sin_phi = np.sin(x[2])

    return [x[0]*cos_theta*cos_phi, x[0]*cos_theta*sin_phi, x[0]*sin_theta]

''' g^-1 '''
def fromCartToSpherical(x):
    rho = np.linalg.norm(x)
    return [rho, np.arcsin(x[2]/rho), np.arctan2(x[1],x[0])]

''' Pdf of gamma law (can be approximated by a gaussian) '''
def f_rho(rho):
    return scipy.stats.norm.pdf(rho, loc=rho_mean, scale=rho_variance)

''' Pdf of a gaussian'''
def f_psi(psi):
    return scipy.stats.norm.pdf(psi, loc=psi_mean, scale = psi_variance)

''' Pdf of scaled beta '''
def f_theta(theta):
    global b, theta_alpha, theta_beta
    return scipy.stats.beta.pdf(theta, a=theta_alpha, b= theta_beta, loc=-b/2., scale = b)

''' f_spherical = f_rho*f_theta*f_psi'''
def f_spherical(x_spherical):
    return f_rho(x_spherical[0])*f_theta(x_spherical[1])*f_psi(x_spherical[2])

def jacobianDeterminant(x):
    return -1./(np.linalg.norm(x)*np.sqrt(x[0]**2 + x[1]**2))

''' h(x,q) = f_(rho,theta,phi)(x_minus_q) |J|(x,q) '''
def h(x,q):
    x_minus_q = inverse_compose(q,x)

    ''' f_(rho,theta,psi)(g^-1(x_minus_q)) '''
    locSphericalCoords = fromCartToSpherical(x_minus_q)
    f_s = f_spherical(locSphericalCoords)

    ''' Determinant of the jacobian '''
    J_det = jacobianDeterminant(x_minus_q)

    return f_s*J_det

''' pdf of X : f_X(x) = E_q(h(q,x)) '''
def f_X(x,mean_q, covariance_q, n):
    return CUT(lambda q : h(x,q),mean_q,covariance_q,n)

''' Define H(q,X,X') = f_X(q + mean(X')) * f_X'(mean(X) - q) '''
def H(q_opt,modeX,modeXprime,mean_q, covariance_q, mean_qprime, covariance_qprime, n):
    return f_X(compose(q_opt,modeXprime), mean_q, covariance_q,n)*\
           f_X(inverse_compose(q_opt,modeX), mean_qprime, covariance_qprime)

''' Optimize the numerical integration using scipy.LowLevelCallable and numba (compilation to c code) 
    See https://stackoverflow.com/questions/58421549/passing-numpy-arrays-as-arguments-to-numba-cfunc
'''
def func_for_gt(modeX, q, mean_q, covariance_q):
    return h(modeX,q)*scipy.stats.multivariate_normal.pdf(q, mean=mean_q, cov=covariance_q)

def jitted_func_for_gt(func, args, args_dtype):
    jitted_func = nb.jit(func)

    @nb.cfunc(nb.types.float64(nb.types.CPointer(args_dtype)))
    def wrapped(user_data_p):
        user_data = nb.carray(user_data_p,1)
        modeX        = user_data[0].modeX
        mean_q       = user_data[0].mean_q
        covariance_q = user_data[0].covariance_q

        return jitted_func(modeX, mean_q, covariance_q)
    return wrapped

''' Expectation ground truth. As we don't have analytic expression, rely on numerical method '''
def groundTruthEstimation(modeX,mean_q, covariance_q):
    #func = lambda x0,x1,x2,x3,x4,x5 : h(modeX,[x0,x1,x2,x3,x4,x5])*scipy.stats.multivariate_normal.pdf([x0,x1,x2,x3,x4,x5], mean=mean_q, cov=covariance_q)
    #return scipy.integrate.nquad(func, [[-np.pi,np.pi], [-0.5*np.pi + 0.01, 0.5*np.pi - 0.01], [-np.pi,np.pi],[-100,100],[-100,100],[-100,100]])
    args_dtype = nb.types.Record.make_c_struct([('modeX'        , nb.types.NestedArray(dtype=nb.types.float64, shape=(6,1))),
                                                ('mean_q'       , nb.types.NestedArray(dtype=nb.types.float64, shape=(6,1))),
                                                ('covariance_q' , nb.types.NestedArray(dtype=nb.types.float64, shape=(6,6)))])
    args = np.array((modeX, mean_q, covariance_q), dtype=args_dtype)
    integrand_func = scipy.LowLevelCallable(jitted_func_for_gt(func_for_gt,args,args_dtype), user_data=args.ctypes.data_as(ctypes.c_void_p))
    return scipy.integrate.nquad(integrand_func, [[-np.pi, np.pi], [-0.5 * np.pi + 0.01, 0.5 * np.pi - 0.01], [-np.pi, np.pi],
                                        [-100, 100], [-100, 100], [-100, 100]], full_output=True)

''' Convert the samples computed for the reduced centered normal to the normal of pdf(mean,covariance) '''
def convertSamplesToPDF(normalized_samples,mean,covariance):
    samples = []

    cov_sqrt = scipy.linalg.sqrtm(covariance)
    for x in normalized_samples:
        sample = np.matmul(cov_sqrt,x) + mean
        samples.append(sample)

    return samples

''' Expectation of func computed using CUT  with a normal pdf (mean, covariance) '''
def CUT(func, mean, covariance, n):
    ''' Get the samples and weights for the reduced centered normal '''
    if (n==4) :
        x_samples_ = generate_cut4_6_x()
        weights = cut4_6_w
    elif (n==6):
        x_samples_ = generate_cut6_6_x()
        weights = cut6_6_w
    elif (n==8):
        x_samples_ = generate_cut8_6_x()
        weights = cut8_6_w
    else:
        print("Error in CUT. Can only be set with n in {4,6,8}")

    start = time.time()
    ''' Convert the samples to provided normal '''
    x_samples = convertSamplesToPDF(x_samples_, mean, covariance)

    expectation = weights[0]*func(x_samples[0])
    for i in range(1,13):
        f_x = func(x_samples[i])
        expectation = expectation + weights[1]*f_x
    for i in range(13, 77):
        f_x = func(x_samples[i])
        expectation = expectation + weights[2]*f_x
    if(n > 4):
        for i in range(77,137):
            f_x = func(x_samples[i])
            expectation = expectation + weights[3] * f_x
        if(n > 6):
            for i in range(137,201):
                f_x = func(x_samples[i])
                expectation = expectation + weights[4] * f_x
            for i in range(201,361):
                f_x = func(x_samples[i])
                expectation = expectation + weights[5] * f_x
            for i in range(361,745):
                f_x = func(x_samples[i])
                expectation = expectation + weights[6] * f_x

    comp_time = time.time() - start
    return expectation, comp_time


''' Test with the example 5.1.2 given in the paper '''
def testWithCos():
    gt_value = -0.543583844
    cut4_paper_value = -0.5492
    cut6_paper_value = -0.5419
    cut8_paper_value = -0.5430

    mean = np.array([0.,0.,0.,0.,0.,0.])
    covariance = np.array([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,1.]])
    cut4_value, cut4_time = CUT(lambda x: np.cos(np.linalg.norm(x)), mean, covariance,4)
    cut6_value, cut6_time = CUT(lambda x: np.cos(np.linalg.norm(x)), mean, covariance,6)
    cut8_value, cut8_time = CUT(lambda x: np.cos(np.linalg.norm(x)), mean, covariance,8)
    print("GT value : " + str(gt_value))
    print(" ---- cut4 ---- ")
    print(" -> paper value : " + str(cut4_paper_value) + ", computed value : " + str(cut4_value), " , comp_time : " + str(cut4_time))
    print(" ---- cut6 ---- ")
    print(" -> paper value : " + str(cut6_paper_value) + ", computed value : " + str(cut6_value), " , comp_time : " + str(cut6_time))
    print(" ---- cut8 ---- ")
    print(" -> paper value : " + str(cut8_paper_value) + ", computed value : " + str(cut8_value), " , comp_time : " + str(cut8_time))

def testWithGT():
    global theta_mode, rho_mean, psi_mean
    ''' Compute f_X with nquad. Used as ground truth'''
    mean_q = [0., 0., 0., 1., 2., 3.]
    covariance_q = [[0.01, 0., 0., 0., 0., 0.],
                    [0., 0.01, 0., 0., 0., 0.],
                    [0., 0., 0.01, 0., 0., 0.],
                    [0., 0., 0., 0.1, 0., 0.],
                    [0., 0., 0., 0., 0.1, 0.],
                    [0., 0., 0., 0., 0., 0.1]
                    ]
    print("Theta_mode : " + str(np.rad2deg(theta_mode)))
    modeX = compose(mean_q, fromSphericalToCart([rho_mean, theta_mode, psi_mean]))
    print("ModeX")
    print(modeX)
    res, abserr = groundTruthEstimation(modeX, mean_q, covariance_q)
    print("Ground truth f_X computed with nquad : "  + str(res) + " with abserr : " + str(abserr))

    ''' Compute f_X with CUT '''
    res,comp_time = f_X(modeX, mean_q, covariance_q)
    print("f_X computed with CUT8 : " + str(res) + ", comp_time : " + str(comp_time))

#------------- MAIN ---------------------
if __name__ =="__main__":
    testWithGT()
