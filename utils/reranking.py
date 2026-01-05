#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri, 25 May 2018 20:29:09


"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set (model tensor)
probFea: all feature vectors of the gallery set (model tensor)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""

import numpy as np
import torch


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    print("=" * 80)
    print("开始Re-ranking过程")
    print(f"参数设置: k1={k1}, k2={k2}, lambda={lambda_value}")
    print(f"查询集特征维度: {probFea.size()}, 图库集特征维度: {galFea.size()}")

    # if feature vector is numpy, you should use 'model.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    print(f"\n查询样本数: {query_num}, 总样本数: {all_num}")

    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        print(f"\n合并特征维度：{feat.size()}")
        # print('using GPU to compute original distance')
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1, -2, feat, feat.t())
        original_dist = distmat.cpu().numpy()
        print(f"原始距离矩阵维度: {original_dist.shape}")
        print(f"原始距离矩阵范围: [{original_dist.min():.4f}, {original_dist.max():.4f}]")

        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    print(f"归一化后距离矩阵维度: {original_dist.shape}")
    print(f"归一化后距离范围: [{original_dist.min():.4f}, {original_dist.max():.4f}]")

    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    print(f"\n初始排序矩阵维度: {initial_rank.shape}")
    print(f"查询0的前{k1}个最近邻索引: {initial_rank[0, :k1 + 1]}")

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    print(final_dist)
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist

