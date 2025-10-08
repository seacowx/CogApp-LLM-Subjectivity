import torch
import numpy as np
from typing import List


class LabelSmoother:

    def __init__(self):
        ...

    
    @staticmethod
    def __gaussian_pdf(x, mu: int, sigma: float, coefficient: float = 1.):
        pdf = lambda x, mu, sigma: \
            coefficient * 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x-mu)**2 / (2 * sigma**2))
        return pdf(x, mu, sigma)


    def __unimodal_gaussian_pdf(self, x, mu: int, sigma: float):

        out_vec = []
        for i in range(5):
            out_vec.append(self.__gaussian_pdf(i, mu, sigma))

        residual = 1 - np.sum(out_vec)
        out_vec[mu] += residual

        return out_vec

    
    def __bimodal_gaussian_pdf(self, x, mu: List[int], sigma: List[float]):

        out_vec_1 = []
        for i in range(5):
            out_vec_1.append(self.__gaussian_pdf(i, mu[0], sigma[0], coefficient=0.5))

        out_vec_2 = []
        for i in range(5):
            out_vec_2.append(self.__gaussian_pdf(i, mu[1], sigma[1], coefficient=0.5))

        out_vec = [ele1 + ele2 for ele1, ele2 in zip(out_vec_1, out_vec_2)]

        residual = 1 - np.sum(out_vec)
        out_vec[mu[0]] += residual / 2
        out_vec[mu[1]] += residual / 2

        return out_vec


    def smooth(self, label_lst: List[int] | List[List[int]], modal: int):

        if modal == 1:
            smooth_func = self.__unimodal_gaussian_pdf
        else:
            smooth_func = self.__bimodal_gaussian_pdf

        if modal == 1:
            sigma = 1.
        else:
            sigma = [0.5, 0.5]

        # smoothe the label distribution using gaussian pdf
        smoothed_label_lst = [
            smooth_func(label_lst[i], label_lst[i], sigma)
            for i in range(len(label_lst))
        ]

        return smoothed_label_lst

