import torch
import numpy as np

def pdist(A, B, dist_type='L2'):
  print(A.shape)
  print(B.shape)
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')

def find_knn_gpu(F0, F1, nn_max_n=-1, knn=1, return_distance=False):

  def knn_dist(f0, f1, knn=1, dist_type='L2'):
    knn_dists, knn_inds = [], []
    with torch.no_grad():
      dist = pdist(f0, f1, dist_type=dist_type)
      min_dist, ind = dist.min(dim=1, keepdim=True)

      knn_dists.append(min_dist)
      knn_inds.append(ind)

      if knn > 1:
        for k in range(knn - 1):
          NR, NC = dist.shape
          flat_ind = (torch.arange(NR) * NC).type_as(ind) + ind.squeeze()
          dist.view(-1)[flat_ind] = np.inf
          min_dist, ind = dist.min(dim=1, keepdim=True)

          knn_dists.append(min_dist)
          knn_inds.append(ind)

    min_dist = torch.cat(knn_dists, 1)
    ind = torch.cat(knn_inds, 1)

    return min_dist, ind

  F0 = np.squeeze(F0, axis=0)
  F1 = np.squeeze(F1, axis=0)
  print(F0.shape)
  print(F1.shape)

  if nn_max_n > 1:
    N = len(F0)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    dists, inds = [], []

    for i in range(C):
      with torch.no_grad():
        dist, ind = knn_dist(F0[i * stride:(i + 1) * stride], F1, knn=knn, dist_type='L2')
      dists.append(dist)
      inds.append(ind)

    dists = torch.cat(dists)
    inds = torch.cat(inds)
    assert len(inds) == N

  else:
    dist = pdist(F0, F1, dist_type='SquareL2')
    min_dist, inds = dist.min(dim=1)
    dists = min_dist.detach().unsqueeze(1)
  if return_distance:
    return inds, dists
  else:
    return inds