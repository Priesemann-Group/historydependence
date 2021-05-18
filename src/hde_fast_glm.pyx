from __future__ import division
import numpy as np
import random
from scipy.special import factorial
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.uint8
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.uint8_t DTYPE_t

# Counts of spieks in the past bin of current spiking

def make_cts_C(np.ndarray[np.double_t, ndim=1] data): #Gives vector of counts for any sample consisting of numbers
	cdef np.ndarray[np.double_t, ndim=1] data_sorted=np.sort(data)
	cdef np.ndarray[np.double_t, ndim=1] uniq=np.unique(data_sorted)
	cdef int jmax=len(uniq)
	cdef np.ndarray[np.int_t, ndim=1] cts=np.zeros(jmax,dtype=np.int)
	cdef double dat_old,dat
	cdef int N, count, index
	dat_old=data_sorted[0]
	count=0
	j=0
	N=len(data_sorted)
	for i in range(N):
		dat=data_sorted[i]
		if dat==dat_old:
			count+=1
		else:
			dat_old=dat
			cts[j]=count
			j+=1
			count=1
	cts[jmax-1]=count
	return cts

def counts_C(np.ndarray[np.double_t, ndim=1] spiketimes, double t_bin, double T_0, double T_f, str embedding_mode):
	cdef int N = int((T_f - T_0) / t_bin)
	cdef np.ndarray[np.double_t, ndim = 1] sptimes = np.sort(np.append(spiketimes, [(N + 1) * t_bin]))
	cdef np.ndarray[np.int_t, ndim = 1] counts = np.zeros(N, dtype=int)
	cdef double t_up, t_low
	cdef int j_low, j_up
	t_up = t_bin
	j_low = 0
	j_up = 0
	t_low = 0.
	for i in range(N):
		while sptimes[j_low] < t_low:
			j_low += 1
		while sptimes[j_up] < t_up:
			j_up += 1
		t_up += t_bin
		t_low += t_bin
		counts[i] += j_up - j_low
	if embedding_mode == 'binary':
		counts[np.nonzero(counts)] = 1
	return counts

def past_activity_array(np.ndarray[np.double_t, ndim=1] spiketimes,np.ndarray[np.int_t, ndim=1] counts,int d_past,double kappa,double tau,double t_bin, double T_0, double T_f,str mode):
	assert counts.dtype == np.int
	cdef int N_bins=int((T_f-T_0)/t_bin)
	cdef int N_spikes=len(spiketimes)
	cdef np.ndarray[np.double_t, ndim=1] sptimes=np.sort(np.append(spiketimes,[N_bins*t_bin]))
	cdef np.ndarray[DTYPE_t, ndim=2] past=np.zeros(N_bins*d_past,dtype=DTYPE).reshape(d_past,N_bins)
	cdef np.ndarray[DTYPE_t, ndim=1] past_temp=np.zeros(N_bins,dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim=1] medians=np.zeros(d_past,dtype=DTYPE)
	cdef double t_up,t_low,t_low_mem
	cdef int j_low,j_up,m
	cdef double T=T_f-T_0
	if mode=='medians':
		t_up=0.
		for k in range(d_past):
			j_low=0
			j_up=0
			t_low=t_up-tau*np.power(10,k*kappa)
			t_low_mem=t_low
			for i in range(N_bins):
				while sptimes[j_low]<t_low:
					j_low+=1
				while sptimes[j_up]<t_up:
					j_up+=1
				t_up+=t_bin
				t_low+=t_bin
				past_temp[i]=j_up-j_low
			t_up=t_low_mem
			medians[k]=np.sort(past_temp)[int(N_bins/2.)]+1
		print medians
		t_up=0.
		for k in range(d_past):
			m=medians[k]
			j_low=0
			j_up=0
			past_sp=0
			past_nosp=0
			t_low=t_up-tau*np.power(10,k*kappa)
			t_low_mem=t_low
			for i in range(N_bins):
				while sptimes[j_low]<t_low:
					j_low+=1
				while sptimes[j_up]<t_up:
					j_up+=1
				t_up+=t_bin
				t_low+=t_bin
				if j_up-j_low>=m:
					past[k][i]=1
			t_up=t_low_mem
	return past

def compress_past(np.ndarray[DTYPE_t, ndim=2] past,int d_past):
	cdef np.ndarray[double, ndim=1] signature=np.array([random.random() for i in range(d_past)])
	cdef np.ndarray[double, ndim=1] past_compressed=np.dot(signature,past)
	return past_compressed

def shuffle_past(np.ndarray[DTYPE_t, ndim=2] past, np.ndarray[np.int_t, ndim=1] counts, int d_past):
	cdef int N_bins = len(counts)
	cdef np.ndarray[double, ndim=1] signature=np.array([random.random() for i in range(d_past)])
	cdef np.ndarray[np.double_t, ndim=1] past_compressed=np.zeros(N_bins,dtype=np.double)
	cdef np.ndarray[np.intp_t, ndim=1] indices_sp = np.nonzero(counts)[0]
	cdef np.ndarray[np.intp_t, ndim=1] indices_nosp = np.nonzero(counts-1)[0]
	cdef np.ndarray[DTYPE_t, ndim=1] past_sp
	cdef np.ndarray[DTYPE_t, ndim=1] past_nosp
	cdef np.ndarray[np.double_t, ndim=1] p_sp=np.zeros(d_past,dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] p_nosp=np.zeros(d_past,dtype=np.double)
	cdef double s
	for k in range(d_past):
		s=signature[k]
		past_sp=past[k][indices_sp]
		past_nosp=past[k][indices_nosp]
		p_sp[k]=np.sum(past_sp)/len(past_sp)
		p_nosp[k]=np.sum(past_nosp)/len(past_nosp)
		np.random.shuffle(past_sp)
		np.random.shuffle(past_nosp)
		past_compressed[[indices_sp]]+=past_sp*s
		past_compressed[[indices_nosp]]+=past_nosp*s
	return past_compressed, p_sp, p_nosp

# Embedding of past activity for the product with a discrete past kernel to compute the firing rate of the GLM

def past_activity(np.ndarray[np.double_t, ndim=1] spiketimes, int d, double kappa, double tau, double t_bin, int N, str embedding_mode):
	cdef np.ndarray[np.double_t, ndim= 1] sptimes = np.sort(np.append(spiketimes, [N * t_bin]))
	cdef np.ndarray[DTYPE_t, ndim= 1] past = np.zeros([N * d], dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim= 1] medians = np.zeros(d, dtype=DTYPE)
	cdef np.ndarray[DTYPE_t, ndim= 1] past_temp = np.zeros(N, dtype=DTYPE)
	cdef double t_up, t_low, t_low_mem, s
	cdef int j, j_low, j_up, m
	t_up = 0.
	if embedding_mode == 'medians':
		t_up = 0.
		for k in range(d):
			j_low = 0
			j_up = 0
			t_low = t_up - tau * np.power(10, k * kappa)
			t_low_mem = t_low
			for i in range(N):
				while sptimes[j_low] < t_low:
					j_low += 1
				while sptimes[j_up] < t_up:
					j_up += 1
				t_up += t_bin
				t_low += t_bin
				past_temp[i] = j_up - j_low
			t_up = t_low_mem
			medians[k] = np.sort(past_temp)[int(N / 2.)] + 1
		t_up = 0.
		for k in range(d):
			m = medians[k]
			j_low = 0
			j_up = 0
			t_low = t_up - tau * np.power(10, k * kappa)
			t_low_mem = t_low
			for i in range(N):
				while sptimes[j_low] < t_low:
					j_low += 1
				while sptimes[j_up] < t_up:
					j_up += 1
				t_up += t_bin
				t_low += t_bin
				if j_up - j_low >= m:
					past[i + k * N] = 1
			t_up = t_low_mem
	if embedding_mode == 'counts':
		for k in range(d):
			j_low = 0
			j_up = 0
			t_low = t_up - tau * np.power(10, k * kappa)
			t_low_mem = t_low
			for i in range(N):
				while sptimes[j_low] < t_low:
					j_low += 1
				while sptimes[j_up] < t_up:
					j_up += 1
				t_up += t_bin
				t_low += t_bin
				if j_up - j_low > 0:
					past[i + k * N] = j_up - j_low
			t_up = t_low_mem
	return past
#
# def past_activity_delay(np.ndarray[np.double_t, ndim=1] spiketimes, int d, double kappa, double tau, double t_bin, int N, str embedding_mode):
#     cdef np.ndarray[np.double_t, ndim= 1] sptimes = np.sort(np.append(spiketimes, [N * t_bin]))
#     cdef np.ndarray[DTYPE_t, ndim= 1] past = np.zeros([N * d], dtype=DTYPE)
#     cdef np.ndarray[DTYPE_t, ndim= 1] medians = np.zeros(d, dtype=DTYPE)
#     cdef np.ndarray[DTYPE_t, ndim= 1] past_temp = np.zeros(N, dtype=DTYPE)
#     cdef double t_up, t_low, t_low_mem, s
#     cdef int j, j_low, j_up, m
#     t_up = 0.
#     if embedding_mode == 'medians':
#         t_up = 0.
#         for k in range(d):
#             j_low = 0
#             j_up = 0
#             t_low = t_up - tau * np.power(10, k * kappa)
#             t_low_mem = t_low
#             for i in range(N):
#                 while sptimes[j_low] < t_low:
#                     j_low += 1
#                 while sptimes[j_up] < t_up:
#                     j_up += 1
#                 t_up += t_bin
#                 t_low += t_bin
#                 past_temp[i] = j_up - j_low
#             t_up = t_low_mem
#             medians[k] = np.sort(past_temp)[int(N / 2.)] + 1
#         t_up = 0.
#         for k in range(d):
#             m = medians[k]
#             j_low = 0
#             j_up = 0
#             t_low = t_up - tau * np.power(10, k * kappa)
#             t_low_mem = t_low
#             for i in range(N):
#                 while sptimes[j_low] < t_low:
#                     j_low += 1
#                 while sptimes[j_up] < t_up:
#                     j_up += 1
#                 t_up += t_bin
#                 t_low += t_bin
#                 if j_up - j_low >= m:
#                     past[i + k * N] = 1
#             t_up = t_low_mem
#     if embedding_mode == 'counts':
#         for k in range(d):
#             j_low = 0
#             j_up = 0
#             t_low = t_up - tau * np.power(10, k * kappa)
#             t_low_mem = t_low
#             for i in range(N):
#                 while sptimes[j_low] < t_low:
#                     j_low += 1
#                 while sptimes[j_up] < t_up:
#                     j_up += 1
#                 t_up += t_bin
#                 t_low += t_bin
#                 if j_up - j_low > 0:
#                     past[i + k * N] = j_up - j_low
#             t_up = t_low_mem
#     return past

def downsample_past_activity(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.int_t, ndim=1] indices, int N, int d):
	cdef int N_downsampled = len(indices)
	cdef np.ndarray[DTYPE_t, ndim= 1] past_downsampled = np.zeros([N_downsampled * d], dtype=DTYPE)
	for k in range(d):
		for i, index in enumerate(indices):
			past_downsampled[i + k * N_downsampled] = past[index + k * N]
	return past_downsampled


def lograte_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.double_t, ndim=1] kernel, int d, int N):
	cdef np.ndarray[np.double_t, ndim= 1] lograte = np.zeros(N, dtype=np.double)
	for k in range(d):
		lograte += kernel[k] * past[k * N:(k + 1) * N]
	return lograte


def jac_sum(np.ndarray[DTYPE_t, ndim=1] past, np.ndarray[np.int_t, ndim=1] counts, np.ndarray[np.double_t, ndim=1] reciproke_rate, int d, int N):
	cdef np.ndarray[np.double_t, ndim= 1] jac = np.zeros(d, dtype=np.double)
	for k in range(d):
		jac[k] = np.dot(past[k * N:(k + 1) * N], counts) - np.dot(past[k * N:(k + 1) * N], reciproke_rate)
	return jac

# Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
	cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
	cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
	cdef double L = np.dot(counts, log_rate) - np.sum(np.log(1 + rate))
	return L

# Jacobian of the Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def jac_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
	cdef int n_sp = np.sum(counts)
	cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
	cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
	cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -1), rate)
	cdef np.ndarray[np.double_t, ndim = 1] jac_kernel = jac_sum(past, counts, reciproke_rate, d, N)
	cdef double dmu = n_sp - np.sum(reciproke_rate)
	return np.append([dmu], jac_kernel)

# Hessian of the Bernoulli likelihood of the spiketrain for given past and GLM parameters h and mu


def hess_L_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, np.ndarray[np.double_t, ndim=1] kernel, double mu):
	cdef int dtot = d + 1
	cdef np.ndarray[np.double_t, ndim = 1] log_rate = lograte_sum(past, kernel, d, N) + mu
	cdef np.ndarray[np.double_t, ndim = 1] rate = np.exp(log_rate)
	cdef np.ndarray[np.double_t, ndim = 1] reciproke_rate = np.multiply(np.power(1 + rate, -2), rate)
	cdef np.ndarray[np.double_t, ndim = 2] hess = np.diag(np.zeros(dtot))
	# Compute elements involving mu
	hess[0][0] = -np.sum(reciproke_rate)
	for l in np.arange(1, dtot):
		hess[0][l] = hess[l][0] = - np.dot(past[(l - 1) * N:l * N], reciproke_rate)
    # Compute all other elements
	for j in np.arange(1, dtot):
		for l in np.arange(j, dtot):
			hess[j][l] = hess[l][j] = -np.dot(past[(l - 1) * N:l * N], np.multiply(past[(j - 1) * N:j * N], reciproke_rate))
	return hess

# Estimate of the conditional entropy based on an average of the likelihood over the data set


def H_cond_B_past(np.ndarray[np.int_t, ndim=1] counts, np.ndarray[DTYPE_t, ndim=1] past, int d, int N, double mu, np.ndarray[np.double_t, ndim=1] kernel):
	return -L_B_past(counts, past, d, N, kernel, mu) / N
