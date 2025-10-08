import json
import pandas as pd
from scipy.stats import ttest_ind


def convert_to_numeric(results: dict) -> dict:

    for key, val in results.items():
        if key == "appraisal_d":
            continue
        
        new_val = [
            float(ele) for ele in val
        ]

        results[key] = new_val

    return results


def main():
    
    llama8_results = convert_to_numeric(json.load(open("./results/envent_LLAMA8.json")))
    llama8_demo_results = convert_to_numeric(json.load(open("./results/envent_LLAMA8_demo.json")))
    llama8_traits_results = convert_to_numeric(json.load(open("./results/envent_LLAMA8_traits.json")))
    llama8_demo_traits_results = convert_to_numeric(json.load(open("./results/envent_LLAMA8_demo_traits.json")))

    qwen7_results = convert_to_numeric(json.load(open("./results/envent_QWEN7.json")))
    qwen7_demo_results = convert_to_numeric(json.load(open("./results/envent_QWEN7_demo.json")))
    qwen7_traits_results = convert_to_numeric(json.load(open("./results/envent_QWEN7_traits.json")))
    qwen7_demo_traits_results = convert_to_numeric(json.load(open("./results/envent_QWEN7_demo_traits.json")))

    llama8_results_df = pd.DataFrame(llama8_results)
    llama8_demo_results_df = pd.DataFrame(llama8_demo_results)
    llama8_traits_results_df = pd.DataFrame(llama8_traits_results)
    llama8_demo_traits_results_df = pd.DataFrame(llama8_demo_traits_results)

    qwen7_results_df = pd.DataFrame(qwen7_results)
    qwen7_demo_results_df = pd.DataFrame(qwen7_demo_results)
    qwen7_traits_results_df = pd.DataFrame(qwen7_traits_results)
    qwen7_demo_traits_results_df = pd.DataFrame(qwen7_demo_traits_results)

    # ================= HYPOTHESIS TESTING with LLAMA8 =================
    
    appraisal_dimensions = llama8_results_df['appraisal_d'].unique()

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
        cur_llama8_scores = llama8_results_df[llama8_results_df['appraisal_d'] == appraisal_d]
        cur_llama8_demo_scores = llama8_demo_results_df[llama8_demo_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_llama8_mean_mae = abs(
            cur_llama8_scores['human_mean'].values - cur_llama8_scores['llm_mean'].values
        )
        cur_llama8_demo_mean_mae = abs(
            cur_llama8_demo_scores['human_mean'].values - cur_llama8_demo_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_mean_mae, cur_llama8_demo_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_llama8_var_mae = abs(
            cur_llama8_scores['human_var'].values - cur_llama8_scores['llm_var'].values
        )
        cur_llama8_demo_var_mae = abs(
            cur_llama8_demo_scores['human_var'].values - cur_llama8_demo_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_var_mae, cur_llama8_demo_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_llama8_wasserstein = cur_llama8_scores['wasserstein'].values
        cur_llama8_demo_wasserstein = cur_llama8_demo_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_wasserstein, cur_llama8_demo_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/llama8_demo_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/llama8_demo_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/llama8_demo_wasserstein.csv", index=False)


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
        cur_llama8_scores = llama8_results_df[llama8_results_df['appraisal_d'] == appraisal_d]
        cur_llama8_traits_scores = llama8_traits_results_df[llama8_traits_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_llama8_mean_mae = abs(
            cur_llama8_scores['human_mean'].values - cur_llama8_scores['llm_mean'].values
        )
        cur_llama8_traits_mean_mae = abs(
            cur_llama8_traits_scores['human_mean'].values - cur_llama8_traits_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_mean_mae, cur_llama8_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_llama8_var_mae = abs(
            cur_llama8_scores['human_var'].values - cur_llama8_scores['llm_var'].values
        )
        cur_llama8_traits_var_mae = abs(
            cur_llama8_traits_scores['human_var'].values - cur_llama8_traits_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_var_mae, cur_llama8_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_llama8_wasserstein = cur_llama8_scores['wasserstein'].values
        cur_llama8_traits_wasserstein = cur_llama8_traits_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_wasserstein, cur_llama8_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/llama8_traits_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/llama8_traits_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/llama8_traits_wasserstein.csv", index=False)


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
        cur_llama8_scores = llama8_results_df[llama8_results_df['appraisal_d'] == appraisal_d]
        cur_llama8_demo_traits_scores = llama8_demo_traits_results_df[llama8_demo_traits_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_llama8_mean_mae = abs(
            cur_llama8_scores['human_mean'].values - cur_llama8_scores['llm_mean'].values
        )
        cur_llama8_demo_traits_mean_mae = abs(
            cur_llama8_demo_traits_scores['human_mean'].values - cur_llama8_demo_traits_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_mean_mae, cur_llama8_demo_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_llama8_var_mae = abs(
            cur_llama8_scores['human_var'].values - cur_llama8_scores['llm_var'].values
        )
        cur_llama8_demo_traits_var_mae = abs(
            cur_llama8_demo_traits_scores['human_var'].values - cur_llama8_demo_traits_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_var_mae, cur_llama8_demo_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_llama8_wasserstein = cur_llama8_scores['wasserstein'].values
        cur_llama8_demo_traits_wasserstein = cur_llama8_demo_traits_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_llama8_wasserstein, cur_llama8_demo_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/llama8_demo_traits_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/llama8_demo_traits_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/llama8_demo_traits_wasserstein.csv", index=False)


    # ================= HYPOTHESIS TESTING with QWEN7 =================
    
    appraisal_dimensions = qwen7_results_df['appraisal_d'].unique()

    # NOTE: qwen7: vanilla vs demo
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
        cur_qwen7_scores = qwen7_results_df[qwen7_results_df['appraisal_d'] == appraisal_d]
        cur_qwen7_demo_scores = qwen7_demo_results_df[qwen7_demo_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_qwen7_mean_mae = abs(
            cur_qwen7_scores['human_mean'].values - cur_qwen7_scores['llm_mean'].values
        )
        cur_qwen7_demo_mean_mae = abs(
            cur_qwen7_demo_scores['human_mean'].values - cur_qwen7_demo_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_mean_mae, cur_qwen7_demo_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_qwen7_var_mae = abs(
            cur_qwen7_scores['human_var'].values - cur_qwen7_scores['llm_var'].values
        )
        cur_qwen7_demo_var_mae = abs(
            cur_qwen7_demo_scores['human_var'].values - cur_qwen7_demo_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_var_mae, cur_qwen7_demo_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_qwen7_wasserstein = cur_qwen7_scores['wasserstein'].values
        cur_qwen7_demo_wasserstein = cur_qwen7_demo_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_wasserstein, cur_qwen7_demo_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/qwen7_demo_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/qwen7_demo_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/qwen7_demo_wasserstein.csv", index=False)


    # NOTE: qwen7: vanilla vs traits
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
        cur_qwen7_scores = qwen7_results_df[qwen7_results_df['appraisal_d'] == appraisal_d]
        cur_qwen7_traits_scores = qwen7_traits_results_df[qwen7_traits_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_qwen7_mean_mae = abs(
            cur_qwen7_scores['human_mean'].values - cur_qwen7_scores['llm_mean'].values
        )
        cur_qwen7_traits_mean_mae = abs(
            cur_qwen7_traits_scores['human_mean'].values - cur_qwen7_traits_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_mean_mae, cur_qwen7_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_qwen7_var_mae = abs(
            cur_qwen7_scores['human_var'].values - cur_qwen7_scores['llm_var'].values
        )
        cur_qwen7_traits_var_mae = abs(
            cur_qwen7_traits_scores['human_var'].values - cur_qwen7_traits_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_var_mae, cur_qwen7_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_qwen7_wasserstein = cur_qwen7_scores['wasserstein'].values
        cur_qwen7_traits_wasserstein = cur_qwen7_traits_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_wasserstein, cur_qwen7_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/qwen7_traits_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/qwen7_traits_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/qwen7_traits_wasserstein.csv", index=False)


    # NOTE: qwen7: vanilla vs demo + traits
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
        cur_qwen7_scores = qwen7_results_df[qwen7_results_df['appraisal_d'] == appraisal_d]
        cur_qwen7_demo_traits_scores = qwen7_demo_traits_results_df[qwen7_demo_traits_results_df['appraisal_d'] == appraisal_d]

        # point estimate: mean
        cur_qwen7_mean_mae = abs(
            cur_qwen7_scores['human_mean'].values - cur_qwen7_scores['llm_mean'].values
        )
        cur_qwen7_demo_traits_mean_mae = abs(
            cur_qwen7_demo_traits_scores['human_mean'].values - cur_qwen7_demo_traits_scores['llm_mean'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_mean_mae, cur_qwen7_demo_traits_mean_mae)
        organize_mean_results["appraisal_d"].append(appraisal_d)
        organize_mean_results["t_stat"].append(t_stat)
        organize_mean_results["p_value"].append(p_value)

        # point estimate: variance
        cur_qwen7_var_mae = abs(
            cur_qwen7_scores['human_var'].values - cur_qwen7_scores['llm_var'].values
        )
        cur_qwen7_demo_traits_var_mae = abs(
            cur_qwen7_demo_traits_scores['human_var'].values - cur_qwen7_demo_traits_scores['llm_var'].values
        )
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_var_mae, cur_qwen7_demo_traits_var_mae)
        organize_var_results["appraisal_d"].append(appraisal_d)
        organize_var_results["t_stat"].append(t_stat)
        organize_var_results["p_value"].append(p_value)

        # distribution metric: wasserstein
        cur_qwen7_wasserstein = cur_qwen7_scores['wasserstein'].values
        cur_qwen7_demo_traits_wasserstein = cur_qwen7_demo_traits_scores['wasserstein'].values
        # perform t-test
        t_stat, p_value = ttest_ind(cur_qwen7_wasserstein, cur_qwen7_demo_traits_wasserstein)
        organize_wasserstein_results["appraisal_d"].append(appraisal_d)
        organize_wasserstein_results["t_stat"].append(t_stat)
        organize_wasserstein_results["p_value"].append(p_value)

    organize_mean_results = pd.DataFrame(organize_mean_results)
    organize_mean_results.to_csv("../significance_tests/qwen7_demo_traits_mean.csv", index=False)
    organize_var_results = pd.DataFrame(organize_var_results)
    organize_var_results.to_csv("../significance_tests/qwen7_demo_traits_var.csv", index=False)
    organize_wasserstein_results = pd.DataFrame(organize_wasserstein_results)
    organize_wasserstein_results.to_csv("../significance_tests/qwen7_demo_traits_wasserstein.csv", index=False)


if __name__ == "__main__":
    main()
