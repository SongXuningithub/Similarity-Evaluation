import torch

def get_vec_dist(vec1,vec2):
    return torch.sum(torch.pow(vec1-vec2,2))/len(vec1)

def get_dist_of_cols(row1,row2):
    assert len(row1) == len(row2)
    dist = 0
    for i in range(len(row1)):
        dist = dist + torch.sum(torch.pow(row1[i]-row2[i],2))/len(row1)
    return dist

def DTW_2D(mat1, mat2):
    mat1 = mat1.squeeze(0)  # [1,48,h,w]  -->  [48,h,w]
    mat2 = mat2.squeeze(0)
    mat1 = mat1.transpose(0, 1).transpose(1, 2)  # [48,h,w]  -->  [h,w,48]
    mat2 = mat2.transpose(0, 1).transpose(1, 2)
    h = mat1.shape[1]
    w = mat2.shape[1]
    dist_mat = torch.zeros(h, w)
    for i in range(h):
        for j in range(w):
            vec1 = mat1[:, i, :]
            vec2 = mat2[:, j, :]
            dist_mat[i, j] = get_dist_of_cols(vec1, vec2)
    min_dist_mat = torch.zeros(h, w)
    min_dist_mat[0, 0] = dist_mat[0, 0]
    for j in range(1, mat2.shape[1]):
        min_dist_mat[0, j] = min_dist_mat[0, j - 1] + dist_mat[0, j]
    for i in range(1, mat1.shape[1]):
        min_dist_mat[i, 0] = min_dist_mat[i - 1, 0] + dist_mat[i, 0]
    for i in range(1, h):
        for j in range(1, w):
            min_dist_mat[i, j] = min(min_dist_mat[i - 1, j - 1] + 2 * dist_mat[i, j],
                                     min_dist_mat[i - 1, j] + dist_mat[i, j], min_dist_mat[i, j - 1] +
                                     dist_mat[i, j])
    return min_dist_mat[h - 1, w  - 1]