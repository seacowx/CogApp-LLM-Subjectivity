"""
Code to post-process the results of pair-rank prompting is modified from the following code snippet:
https://github.com/MiaoXiong2320/llm-uncertainty/blob/main/vis_aggregated_conf_top_k.py#L334
"""

import torch
import numpy as np
from aggregators.base_aggregator import BaseAggregator


class PairRankAggregator(BaseAggregator):

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set seed for reproducibility
        torch.manual_seed(96)

    @staticmethod
    # Define the loss function: Frobenius norm of W
    def nll_loss_func(w_cat, rank_matrix):
        p_cat = torch.nn.functional.softmax(w_cat, dim=0)
        loss = 0
        # Compute the denominator for all combinations of p_cat[row] and p_cat[col]
        # denominator[i,j] = p_cat[i] + p_cat[j]
        denominator = p_cat.view(-1, 1) + p_cat.view(1, -1)
        # Avoid division by zero by adding a small constant
        epsilon = 1e-10
        # Compute the ratio
        ratios = (p_cat.view(-1, 1) + epsilon) / (denominator + 2*epsilon)
        loss = -torch.sum(rank_matrix * ratios)
        return loss

    def train(
        self,
        w_cat: torch.Tensor,
        id_to_element: dict,
        rank_matrix: torch.Tensor,
        optimizer: torch.optim.Optimizer 
    ):
        # Training loop to minimize the loss function
        w_cat = w_cat.to(self.device)
        rank_matrix = rank_matrix.to(self.device)
        for _ in range(1000):
            # Compute the loss
            loss = self.nll_loss_func(w_cat, rank_matrix)

            # Zero gradients, backward pass, optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_p_cat = torch.nn.functional.softmax(w_cat, dim=0)
        id = torch.argmax(final_p_cat)
        answer = id_to_element[int(id)]
        final_p_cat = final_p_cat.cpu().detach().numpy()
        score = final_p_cat[id]

        return answer, score, final_p_cat

    @staticmethod
    def convert_to_rank_matrix(answer_set):
        """
        post-process for evaluating pair-rank results
        """
        num_trail = len(answer_set)
        TOP_K = 5

        # compute the number of unique answers
        unique_elements = np.arange(1, 6)
        NUM_OPTIONS = 5

        # map every item to its unique id
        # element_to_id["A"]=0
        element_to_id = {element: idx for idx, element in enumerate(unique_elements)}
        id_to_element = {idx: element for element, idx in element_to_id.items()}

        rank_matrix = torch.zeros(NUM_OPTIONS, NUM_OPTIONS)
        for trail, answers in answer_set.items():
            # answers[trail_0] = {0:"A", ..."3":"D"}
            mask = torch.ones(NUM_OPTIONS)
            for idx in range(TOP_K):
                # answer["0"] = "A" -> option
                option = answers[str(idx)]
                id_cat = element_to_id[option]
                mask[id_cat] = 0
                rank_matrix[id_cat, :] += mask

        rank_matrix = rank_matrix / num_trail

        # assert rank_matrix.any() >= 0.0 and rank_matrix.any() <=1.0, "rank matrix should be [0,1]"
        return rank_matrix, NUM_OPTIONS, TOP_K, id_to_element


    def aggregate(self, rating_prob_dict: dict):
        pair_rank_dict = {
            str(trail): {
                str(idx): ele for idx, ele in enumerate(rating_prob_dict[trail]['ranked_ratings'])
            } 
            for trail in range(5)
        }

        rank_matrix, NUM_OPTIONS, TOP_K, id_to_element = self.convert_to_rank_matrix(pair_rank_dict)

        # initialize categorical distribution using samples from Gaussian distribution
        w_cat = torch.randn(NUM_OPTIONS, requires_grad=True)
        optimizer = torch.optim.SGD([w_cat], lr=0.01)

        answer, score, final_p_cat = self.train(
            w_cat=w_cat,
            id_to_element=id_to_element,
            rank_matrix=rank_matrix,
            optimizer=optimizer,
        )

        return answer, score, final_p_cat


