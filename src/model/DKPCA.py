from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import skew, kurtosis, shapiro
import numpy as np

def DKPCA(features_scaled, n_components=None, kernel=None):
    
    # Perform KPCA
    kpca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True)
    kpca.fit(features_scaled)
    
    # Transform the features
    projections = kpca.transform(features_scaled)
    
    # Determine thresholds
    thresholds = {}
    for j in range(projections.shape[1]):
        stat, p_value = shapiro(projections[:, j])
        if p_value > 0.05:
            mean = np.mean(projections[:, j])
            std = np.std(projections[:, j])
            thresholds[j] = mean + 2 * std  # 95% confidence interval
        else:
            thresholds[j] = np.percentile(projections[:, j], 95)  # 95th percentile
    
    # Select subset indices
    subset_indices = []
    for j in range(projections.shape[1]):
        candidate_indices = np.where(projections[:, j] < thresholds[j])[0]
        if candidate_indices.size > 0:
            subset_index = candidate_indices[np.argmax(projections[candidate_indices, j])]
            subset_indices.append(subset_index)
    
    subset_indices = list(set(subset_indices))
    
    # Compute the new kernel matrix using the same kernel function
    K_new = pairwise_kernels(features_scaled[subset_indices, :], features_scaled, metric=kernel)
    
    # Compute the DKPCA features
    eigenvectors_subset = kpca.eigenvectors_[subset_indices, :]
    features_dkpca = np.dot(K_new.T, eigenvectors_subset)
    
    return features_dkpca