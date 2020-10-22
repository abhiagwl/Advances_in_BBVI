import autograd.numpy as np
import autograd.numpy.random as npr
# from autograd.scipy.special import logsumexp
from autograd.scipy.special import logsumexp
class GaussianVI():
    def logpdf(samples, mu, sig):
        """
        Evaluates the log pdf of the multivariate gaussian distribution.
        The mean and scale are supposed to be of (D,) and (D,D) shapes while
        the samples can be of arbirtrary batch size with last dimension being D
        
        Returns a the log pdf of shape Z.shape[:-1] = (*B)
        """
        D = samples.shape[-1]
        assert mu.shape == (D,)
        assert sig.shape == (D,D)
    #     assert all(np.diag(sig)>0)
     
        cov = np.matmul(sig,sig.T)
        M = (samples-mu)
        a = D*np.log(2*np.pi)
        b = 2*np.sum(np.log((np.diag(sig))))
        c = np.sum(np.matmul(M,np.linalg.inv(cov)) * M, -1)
        assert (c.shape == samples.shape[:-1])
        return -0.5*(a+b+c) 

    def sample(mu, sig, sample_size=()):
        """
        Takes in a mean and sig of a multivariate guassian distribution
        mu is the mean and sig is the cholesky factor of the covariance matrix such that
        cov = sig*sig.T
        
        Can sample any arbirtrary sample size using the reparameterization trick
        """
        
        D = mu.shape[-1]
        assert(mu.shape == (D,))
        assert(sig.shape == (D,D))
    #     sig = np.tril(sig)
        samples = mu + np.dot(npr.randn(*sample_size,D),sig.T)
        assert(samples.shape == (*sample_size,D))
        return samples

def gaussian_logpdf(samples, mu, sig):
    """
    Evaluates the log pdf of the multivariate gaussian distribution.
    The mean and scale are supposed to be of (D,) and (D,D) shapes while
    the samples can be of arbirtrary batch size with last dimension being D
    
    Returns a the log pdf of shape Z.shape[:-1] = (*B)
    """
    D = samples.shape[-1]
    assert mu.shape == (D,)
    assert sig.shape == (D,D)
#     assert all(np.diag(sig)>0)
 
    cov = np.matmul(sig,sig.T)
    M = (samples-mu)
    a = D*np.log(2*np.pi)
    b = 2*np.sum(np.log((np.diag(sig))))
    c = np.sum(np.matmul(M,np.linalg.inv(cov)) * M, -1)
    assert (c.shape == samples.shape[:-1])
    return -0.5*(a+b+c) 

def gaussian_sample(mu, sig, sample_size=()):
    """
    Takes in a mean and sig of a multivariate guassian distribution
    mu is the mean and sig is the cholesky factor of the covariance matrix such that
    cov = sig*sig.T
    
    Can sample any arbirtrary sample size using the reparameterization trick
    """
    
    D = mu.shape[-1]
    assert(mu.shape == (D,))
    assert(sig.shape == (D,D))
#     sig = np.tril(sig)
    samples = mu + np.dot(npr.randn(*sample_size,D),sig.T)
    assert(samples.shape == (*sample_size,D))
    return samples

def gaussian_entropy(mu, sig):
    """
    Takes in a mean and sig of a multivariate guassian distribution
    mu is the mean and sig is the cholesky factor of the covariance matrix such that
    cov = sig*sig.T
    
    gives back the closed for entropy for that normal distribution 
    """
    
    D = mu.shape[-1]
    assert(mu.shape == (D,))
    assert(sig.shape == (D,D))
    a = D*np.log(2*np.pi) + D
    b = 2*np.sum(np.log((np.diag(sig))))
    return 0.5*(a+b)


def generate_mask(hyper_params):
    if hyper_params['data_dim']<2:
        print("Add more data_dim. rNVP won't work for less than two")
        exit()
    D = hyper_params['data_dim']
    d_1 = np.int(D/2)
    d_2 = np.int(D - d_1)
    assert(d_1 + d_2 ==  D)
    """
    The correct way to extend this code would be to always have d1 ones and d2 zeros. 
    We can change the positions of the ones and zeros and the rest of the code should still work.
    But the later part does indeed assume this:
        - for a coupling layer, d1 ones and d2 zeros in a mask and then this mask in inverted
        - so first d1 elements go unchanged and then the transformed d2 elements are unchanged
         while the previous d1 are transformed
    """
    m = np.concatenate([np.ones(d_1), np.zeros(d_2)])#change here the permutation of ones and zeros to get random masks
    mask = [m]*np.int(hyper_params['rnvp_num_transformations'])
    return mask

def mix_of_gaussian_sample(normals, a, num_samples):
    assert(len(a.shape)==1)
    assert(len(normals)== a.shape[0])
    K = len(a)

    mix_samples = []
    for t in range(num_samples):
        i = np.argmax(npr.multinomial(1, a))
        mu,sig = normals[i]
        mix_samples.append(gaussian_sample(mu,sig,(1,)))
    mix_samples = np.array(mix_samples)
    return np.squeeze(mix_samples)

def mix_of_gaussian_pdf(samples, normals, a):
    assert(len(a.shape)==1)
    assert(len(normals)== a.shape[0])
    K = len(a)
    
    p = np.zeros(samples.shape[:-1])
    for i in range(K):
        mu,sig = normals[i]
        p += a[i]*np.exp(gaussian_logpdf(samples,mu,sig))
        # log_pdf.append(+np.log(a[i]))
    # log_pdf = np.array(log_pdf)
    # return logsumexp(log_pdf, axis = 0)
    return p 
    
def mix_of_gaussian_logpdf(samples,normals,a):
    assert(len(a.shape)==1)
    assert(len(normals)== a.shape[0])
    K = len(a)
    
    log_pdf = []
    for i in range(K):
        mu,sig = normals[i]
        log_pdf.append(gaussian_logpdf(samples,mu,sig)+np.log(a[i]))
    log_pdf = np.array(log_pdf)
    return logsumexp(log_pdf, axis = 0)

def relu(x):       return np.maximum(0, x)
def leakyrelu(x, slope = 0.01):       return np.maximum(0, x) + slope*np.minimum(0,x)
def sigmoid(x):    return 0.5 * (np.tanh(0.5*x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)
def tanh(x): return np.tanh(x)
def softmax(x): return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
def softmax_matrix(x): 
    y = x - np.repeat(np.max(x,-1), x.shape[-1]).reshape(x.shape)
    z = np.exp(y)/ np.repeat(np.sum(np.exp(y), -1), y.shape[-1]).reshape(y.shape)
    assert(z.shape == x.shape)
    del y,x
    return z

def batch_normalize(activations):
    axis_for_calc = tuple(range(len(activations.shape)-1))
    mbmean = np.mean(activations, axis=axis_for_calc , keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=axis_for_calc, keepdims=True) + 1)

def net_forward_s(params, inputs):
    inpW, inpb = params[0]
    inputs = leakyrelu(np.dot(inputs, inpW) + inpb)
    for W, b in params[1:-1]:
        # outputs = batch_normalize(np.dot(inputs, W) + b)
        outputs = np.dot(inputs, W) + b
        inputs = leakyrelu(outputs)
    outW, outb = params[-1]
    outputs = tanh(np.dot(inputs, outW) + outb)
    return outputs
def net_forward_t(params, inputs):
    inpW, inpb = params[0]
    inputs = leakyrelu(np.dot(inputs, inpW) + inpb)
    for W, b in params[1:-1]:
        # outputs = batch_normalize(np.dot(inputs, W) + b)
        outputs = np.dot(inputs, W) + b
        inputs = leakyrelu(outputs)
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def net_forward_st(params, inputs):
    inpW, inpb = params[0]
    inputs = leakyrelu(np.dot(inputs, inpW) + inpb)
    for W, b in params[1:-1]:
        # outputs = batch_normalize(np.dot(inputs, W) + b)
        outputs = np.dot(inputs, W) + b
        inputs = leakyrelu(outputs)
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    assert(outputs.shape[:-1] == inputs.shape[:-1])
    assert(outputs.shape[-1]%2 == 0)
    s,t = np.array_split(outputs, 2 , -1)
    s = tanh(s)
    assert(s.shape == t.shape)
    # assert(t.shape == inputs.shape)
    return s,t

    


def forward_transform(z, params, hyper_params):
    neg_log_det_J = np.zeros(z.shape[:-1])
    st_list = params

    for i in range(hyper_params['rnvp_num_transformations']):
        #Break Z in z1 and z2
        z_1, z_2 = np.array_split(z, 2 , -1)
        s, t = net_forward_st(params = st_list[i][0], inputs = z_1)#output should be of the shape of z_2
        assert(z_2.shape == s.shape)
        neg_log_det_J -= np.sum(s,axis=-1)  #Immediately update neg log jacobian with s outputs
        y_2 = z_2*np.exp(s) + t     #y2 is the transformed part
        del z_2, s, t 
        y_1 = z_1   #y1 is the part that propogates unchanged
        del z_1
        #at this point only y1 and y2 exist
        s, t = net_forward_st(params = st_list[i][1], inputs = y_2)#output should be of the shape of y_1
        assert(y_1.shape == s.shape)
        neg_log_det_J -= np.sum(s,axis=-1)  #Immediately update neg log jacobian with s outputs
        
        x_1 = y_1*np.exp(s) + t     #transforming the part z_1 = y_1 for the first time
        del y_1, s, t 
        x_2 = y_2   #the already transformed z_2 which was y_2, remains unchanged in x_2
        del y_2

        
        z = np.concatenate([x_1,x_2], -1)
        del x_1, x_2

    assert(z.shape[:-1] == neg_log_det_J.shape)
    return z, neg_log_det_J

def inverse_transform(x, params, hyper_params):
    neg_log_det_J = np.zeros(x.shape[:-1])
    st_list = params
    for i in reversed(range(hyper_params['rnvp_num_transformations'])):
        #split the given x in two components
        x_1, x_2 = np.array_split(x, 2 , -1)
        #in this x_2 should have been unchanged so that gives y-2 directly

        s, t = net_forward_st(params = st_list[i][1], inputs = x_2)#output should be of the shape of x_1
        assert(x_1.shape == s.shape)
        neg_log_det_J -= np.sum(s,axis=-1)
        # x_1 would have been obtained by using y_2 = x_2 as the input and passing it through relevant s,t 
        y_1  =  (x_1 - t)*np.exp(-s)
        del x_1, t, s
        y_2  = x_2 
        del x_2
        #Now only y-1 and y-2 exist
        #y_1 will go back unchanged to give z-1
        
        z_1 = y_1 
        del y_1
        
        s, t = net_forward_st(params = st_list[i][0], inputs = z_1)#output should be of the shape of x_2
        assert(y_2.shape == s.shape)
        neg_log_det_J -= np.sum(s,axis=-1)

        # y-2 would have been obtained by using y_1 = z_1 as the input and passing it through relevant s,t 
        z_2 = (y_2- t)*np.exp(-s)
        del y_2, t, s
        
        x = np.concatenate([z_1,z_2], -1)
        del z_1, z_2


    assert(x.shape[:-1] == neg_log_det_J.shape)
    return x, neg_log_det_J

def rnvp_sample_logpdf(params, hyper_params, sample_size = None, fixed_samples = None):

    assert(fixed_samples is None)
    z_o = gaussian_sample(mu = np.zeros(hyper_params['data_dim']), sig = np.eye(hyper_params['data_dim']), 
            sample_size =  sample_size)
    if hyper_params['laplaces_method_use']==1:
        raise ValueError
    else :
        samples, neg_log_det_J = forward_transform(z = z_o, params = params, hyper_params = hyper_params)
        lq = gaussian_logpdf(samples = z_o, mu = np.zeros(hyper_params['data_dim']), sig = np.eye(hyper_params['data_dim'])) + neg_log_det_J

    # assert(samples.shape == (*sample_size, hyper_params['data_dim']))
    assert(z_o.shape == samples.shape)
    assert(lq.shape == samples.shape[:-1])
    assert(neg_log_det_J.shape == lq.shape)
    del z_o, neg_log_det_J
    return samples, lq


def rnvp_inverse_sample_logpdf(params, hyper_params, samples):

    
    sample_size = samples.shape[:-1]
    
    z_o, neg_log_det_J = inverse_transform(x = samples, params = params, hyper_params = hyper_params)
    if hyper_params['laplaces_method_use']==1:
        raise ValueError
    else:
        lq = gaussian_logpdf(samples = z_o, mu = np.zeros(hyper_params['data_dim']), sig = np.eye(hyper_params['data_dim'])) + neg_log_det_J

    assert(z_o.shape == (*sample_size, hyper_params['data_dim']))
    assert(lq.shape == sample_size)
    assert(neg_log_det_J.shape == sample_size)
    return z_o, lq
