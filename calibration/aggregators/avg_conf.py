from aggregators.base_aggregator import BaseAggregator


class AvgConfAggregator(BaseAggregator):

    def __init__(self) -> None:
        super().__init__()

    def aggregate(self, rating_prob_dict: dict):

        aggregated_rating_dict = {}
        rating_count = {}
        total_prob = 0
        for rating_prob in rating_prob_dict:
            rating, prob = rating_prob.values()

            if rating not in rating_count:
                rating_count[rating] = 0

            if rating not in aggregated_rating_dict:
                aggregated_rating_dict[rating] = 0

            aggregated_rating_dict[rating] += prob
            rating_count[rating] += 1
            total_prob += prob

        for rating in aggregated_rating_dict:
            try:
                aggregated_rating_dict[rating] /= total_prob
            except:
                # if probabilities are not correctly generated, backoff to frequency-based probability
                aggregated_rating_dict[rating] = rating_count[rating] / len(rating_prob_dict)

        return aggregated_rating_dict