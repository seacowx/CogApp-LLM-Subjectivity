import ast
import pandas as pd
from scipy.stats import ttest_ind, wasserstein_distance


def main():

    ls_results = pd.read_csv("./results/deberta_large_unimodal_results.csv", sep="\t")
    ls_demo_results = pd.read_csv("./results/deberta_large_unimodal_demo_results.csv", sep="\t")
    ls_traits_results = pd.read_csv("./results/deberta_large_unimodal_traits_results.csv", sep="\t")
    ls_demo_traits_results = pd.read_csv("./results/deberta_large_unimodal_demo_traits_results.csv", sep="\t")

    appraisal_dimensions = ls_results['appraisal_d'].unique()

    # NOTE: llama8: vanilla vs demo
    organize_mean_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_var_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_wasserstein_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    for appraisal_d in appraisal_dimensions:
        cur_ls_results = ls_results[ls_results['appraisal_d'] == appraisal_d]
        cur_ls_demo_results = ls_demo_results[ls_demo_results['appraisal_d'] == appraisal_d]

        cur_mean_mae = abs(cur_ls_results['pred_mean'].values - cur_ls_results['gt_mean'].values)
        cur_var_mae = abs(cur_ls_results['pred_var'].values - cur_ls_results['gt_var'].values)
        cur_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']), 
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_results.iterrows()
        ]

        cur_demo_mean_mae = abs(cur_ls_demo_results['pred_mean'].values - cur_ls_demo_results['gt_mean'].values)
        cur_demo_var_mae = abs(cur_ls_demo_results['pred_var'].values - cur_ls_demo_results['gt_var'].values)
        cur_demo_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']),
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_demo_results.iterrows()
        ]

        # test the mean 
        t_stat, p_value = ttest_ind(cur_mean_mae, cur_demo_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # test the variance
        t_stat, p_value = ttest_ind(cur_var_mae, cur_demo_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # test the wasserstein 
        t_stat, p_value = ttest_ind(cur_wasserstein, cur_demo_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results_df = pd.DataFrame(organize_mean_results)
    organize_var_results_df = pd.DataFrame(organize_var_results)
    organize_wasserstein_results_df = pd.DataFrame(organize_wasserstein_results)

    organize_mean_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_mean_results.csv",
        index=False,
        sep="\t",
    )
    organize_var_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_var_results.csv",
        index=False,
        sep="\t",
    )
    organize_wasserstein_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_wasserstein_results.csv",
        index=False,
        sep="\t",
    )


    # NOTE: llama8: vanilla vs traits
    organize_mean_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_var_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_wasserstein_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    for appraisal_d in appraisal_dimensions:
        cur_ls_results = ls_results[ls_results['appraisal_d'] == appraisal_d]
        cur_ls_traits_results = ls_traits_results[ls_traits_results['appraisal_d'] == appraisal_d]

        cur_mean_mae = abs(cur_ls_results['pred_mean'].values - cur_ls_results['gt_mean'].values)
        cur_var_mae = abs(cur_ls_results['pred_var'].values - cur_ls_results['gt_var'].values)
        cur_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']), 
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_results.iterrows()
        ]

        cur_traits_mean_mae = abs(cur_ls_traits_results['pred_mean'].values - cur_ls_traits_results['gt_mean'].values)
        cur_traits_var_mae = abs(cur_ls_traits_results['pred_var'].values - cur_ls_traits_results['gt_var'].values)
        cur_traits_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']),
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_traits_results.iterrows()
        ]

        # test the mean 
        t_stat, p_value = ttest_ind(cur_mean_mae, cur_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # test the variance
        t_stat, p_value = ttest_ind(cur_var_mae, cur_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # test the wasserstein 
        t_stat, p_value = ttest_ind(cur_wasserstein, cur_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results_df = pd.DataFrame(organize_mean_results)
    organize_var_results_df = pd.DataFrame(organize_var_results)
    organize_wasserstein_results_df = pd.DataFrame(organize_wasserstein_results)

    organize_mean_results_df.to_csv(
        "./results/deberta_large_unimodal_traits_mean_results.csv",
        index=False,
        sep="\t",
    )
    organize_var_results_df.to_csv(
        "./results/deberta_large_unimodal_traits_var_results.csv",
        index=False,
        sep="\t",
    )
    organize_wasserstein_results_df.to_csv(
        "./results/deberta_large_unimodal_traits_wasserstein_results.csv",
        index=False,
        sep="\t",
    )


    # NOTE: llama8: vanilla vs demo + traits
    organize_mean_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_var_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    organize_wasserstein_results = {
        "appraisal_d": [],
        "t_stat": [],
        "p_value": [],
    }
    for appraisal_d in appraisal_dimensions:
        cur_ls_results = ls_results[ls_results['appraisal_d'] == appraisal_d]
        cur_ls_demo_traits_results = ls_demo_traits_results[ls_demo_traits_results['appraisal_d'] == appraisal_d]

        cur_mean_mae = abs(cur_ls_results['pred_mean'].values - cur_ls_results['gt_mean'].values)
        cur_var_mae = abs(cur_ls_results['pred_var'].values - cur_ls_results['gt_var'].values)
        cur_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']), 
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_results.iterrows()
        ]

        cur_demo_traits_mean_mae = abs(cur_ls_demo_traits_results['pred_mean'].values - cur_ls_demo_traits_results['gt_mean'].values)
        cur_demo_traits_var_mae = abs(cur_ls_demo_traits_results['pred_var'].values - cur_ls_demo_traits_results['gt_var'].values)
        cur_demo_traits_wasserstein = [
            wasserstein_distance(
                ast.literal_eval(ele[1]['gt_answer']),
                ast.literal_eval(ele[1]['sampled_rating_lst']),
            )
            for ele in cur_ls_demo_traits_results.iterrows()
        ]

        # test the mean 
        t_stat, p_value = ttest_ind(cur_mean_mae, cur_demo_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # test the variance
        t_stat, p_value = ttest_ind(cur_var_mae, cur_demo_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # test the wasserstein 
        t_stat, p_value = ttest_ind(cur_wasserstein, cur_demo_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results_df = pd.DataFrame(organize_mean_results)
    organize_var_results_df = pd.DataFrame(organize_var_results)
    organize_wasserstein_results_df = pd.DataFrame(organize_wasserstein_results)

    organize_mean_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_traits_mean_results.csv",
        index=False,
        sep="\t",
    )
    organize_var_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_traits_var_results.csv",
        index=False,
        sep="\t",
    )
    organize_wasserstein_results_df.to_csv(
        "./results/deberta_large_unimodal_demo_traits_wasserstein_results.csv",
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    main()
