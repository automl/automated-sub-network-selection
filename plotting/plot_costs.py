import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


all_results = pd.read_csv("results_evaluation.csv")

marker = ["x", "o", "D", "s", "v", "^", "<", ">", "p", "S"]
for model, df_model in all_results.groupby("model_type"):
    for task, df_task in df_model.groupby("task"):
        for search_space, df_search_space in df_task.groupby("search_space"):
            for obj, df_objective in df_search_space.groupby("objective"):
                plt.figure()
                for j, (method, df_method) in enumerate(df_objective.groupby("method")):
                    for i, (seed, df_seed) in enumerate(df_method.groupby("seed")):
                        print(seed, method, len(df_seed))
                        if i == 0:
                            plt.scatter(
                                df_seed["params"],
                                np.array(df_seed["error"]),
                                color=f"C{j}",
                                marker=marker[i],
                                label=method,
                                s=15,
                            )
                        else:
                            plt.scatter(
                                df_seed["params"],
                                np.array(df_seed["error"]),
                                color=f"C{j}",
                                marker=marker[i],
                                s=15,
                            )
                        idx = np.argsort(df_seed["params"])
                        plt.step(
                            np.array(df_seed["params"])[idx],
                            np.array(df_seed["error"])[idx],
                            color=f"C{j}",
                            alpha=0.4,
                            where="post",
                        )
                plt.title(search_space)
                plt.xlabel(r"parameter count")
                plt.ylabel(obj)
                plt.grid(linewidth="1", alpha=0.9)
                plt.legend()
                plt.savefig(f"../figures/costs_{model}_{obj}_{task}_{search_space}.png")
                plt.show()
