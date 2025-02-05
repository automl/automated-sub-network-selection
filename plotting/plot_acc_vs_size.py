import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


all_results = pd.read_csv("results_evaluation.csv")

print(len(all_results))
marker = ["x", "o", "D", "s", "v", "^", "<", ">", "p", "S"]
for model, df_model in all_results.groupby("model_type"):
    print(model)
    for task, df_task in df_model.groupby("task"):
        for search_space, df_search_space in df_task.groupby("search_space"):
            for obj, df_objective in df_search_space.groupby("objective"):
                plt.figure()
                for j, (search_strategy, df_search_strategy) in enumerate(
                    df_objective.groupby("search_strategy")
                ):
                    for i, (seed, df_seed) in enumerate(
                        df_search_strategy.groupby("seed")
                    ):
                        print(
                            f"seed={seed}, search_strategy={search_strategy}, N={len(df_seed)}, iters={df_seed['iterations'].unique()}"
                        )
                        if i == 0:
                            plt.scatter(
                                df_seed["params"],
                                1 - np.array(df_seed["accuracy"]),
                                color=f"C{j}",
                                marker=marker[i],
                                label=search_strategy,
                            )
                        else:
                            plt.scatter(
                                df_seed["params"],
                                1 - np.array(df_seed["accuracy"]),
                                color=f"C{j}",
                                marker=marker[i],
                            )
                plt.title(f"{model} {task} {obj} {search_space}")
                plt.xlabel(r"parameter count")
                plt.ylabel(r"downstream_error")
                if task == "arc_easy":
                    plt.plot(0.17073170731, 1 - 0.3522727272727273, "ko")
                if task == "sciq":
                    plt.plot(0.17073170731, 1 - 0.564, "ko")
                plt.grid(linewidth="1", alpha=0.9)
                plt.legend()
                #               plt.xscale('log')
                plt.xlim(10e-4, 1)
                if task == "arc_easy":
                    plt.ylim(0.45, 0.8)
                elif task == "sciq":
                    plt.ylim(0.25, 0.8)
                plt.savefig(
                    f"../figures/acc_vs_size_{model}_{obj}_{task}_{search_space}.png"
                )
#                plt.show()
