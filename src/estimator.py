import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def wmean(x, w):
    return np.sum(x * w) / np.sum(w)

class SegmentEstimator:
    def __init__(self, data_path, propensity_col="propensity"):
        self.data_path = data_path
        self.propensity_col = propensity_col

    def load_segment(self, segment_rules):
        where_clause = " AND ".join(segment_rules)
        query = f"""
        SELECT *
        FROM read_parquet('{self.data_path}')
        WHERE {where_clause}
        """
        return duckdb.sql(query).to_df()
    
    def apply_overlap(self, df, lower=0.01, upper=0.99):
        e = df[self.propensity_col]
        mask = (e >= lower) & (e <= upper)
        trimmed = df[mask].copy()
        stats = {
            "n_original": len(df),
            "n_trimmed": len(trimmed),
            "pct_dropped": "{} %".format(round(float(100 * (1 - len(trimmed) / len(df)) if len(df) > 0 else None), 5)),
            "propensity_min": round(float(e.min()), 5),
            "propensity_max": round(float(e.max()), 5),
        }
        return trimmed, stats
    
    def compute_weights(self, df):
        e = df[self.propensity_col]
        t = df["is_algo"]
        w = t / e + (1 - t) / (1 - e)
        df = df.copy()
        df["weight"] = w
        ess = (w.sum() ** 2) / (np.sum(w ** 2))
        return df, ess
    
    def compute_outcomes(self, df):
        treated = df[df["is_algo"] == 1]
        control = df[df["is_algo"] == 0]
        like_t = wmean(treated["y_like_24h"], treated["weight"])
        like_c = wmean(control["y_like_24h"], control["weight"])
        dislike_t = wmean(treated["y_dislike_24h"], treated["weight"])
        dislike_c = wmean(control["y_dislike_24h"], control["weight"])
        return {
            "ATE_like": round(float(like_t - like_c), 5),
            "ATE_dislike": round(float(dislike_t - dislike_c), 5),
        }
    
    def run_segment(self, segment_rules):
        df = self.load_segment(segment_rules)
        if len(df) == 0:
            return {"error": "Empty segment"}
        df_trim, overlap_stats = self.apply_overlap(df)
        df_w, ess = self.compute_weights(df_trim)
        outcomes = self.compute_outcomes(df_w)
        return {
            "overlap_stats": overlap_stats,
            "ESS": round(float(ess), 3),
            "outcomes": outcomes,
        }

    def smd(self, df, cols, weighted=False):
        smd_dict = {}
        treated = df[df["is_algo"] == 1]
        control = df[df["is_algo"] == 0]
        for col in cols:
            x_t = treated[col]
            x_c = control[col]
            if weighted:
                w_t = treated["weight"]
                w_c = control["weight"]
                mean_t = wmean(x_t, w_t)
                mean_c = wmean(x_c, w_c)
                var_t = wmean((x_t - mean_t) ** 2, w_t)
                var_c = wmean((x_c - mean_c) ** 2, w_c)
            else:
                mean_t = x_t.mean()
                mean_c = x_c.mean()
                var_t = x_t.var()
                var_c = x_c.var()
            pooled_std = np.sqrt((var_t + var_c) / 2)
            smd_val = 0 if pooled_std == 0 else (mean_t - mean_c) / pooled_std
            smd_dict[col] = smd_val
        return smd_dict

    def smd_plot(self, segment_rules, cols):
        df = self.load_segment(segment_rules)
        df_trim, _ = self.apply_overlap(df)
        df_w, _ = self.compute_weights(df_trim)
        smd_before = self.smd(df_trim, cols, weighted=False)
        smd_after = self.smd(df_w, cols, weighted=True)
        features = list(smd_before.keys())
        before_vals = [smd_before[f] for f in features]
        after_vals = [smd_after[f] for f in features]
        y = np.arange(len(features))
        plt.figure(figsize=(8, 5))
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.scatter(before_vals, y, label="Before weighting")
        plt.scatter(after_vals, y, label="After weighting")
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
        plt.axvline(0.1, color="grey", linestyle="--", linewidth=1)
        plt.axvline(-0.1, color="grey", linestyle="--", linewidth=1)
        plt.yticks(y, features)
        plt.xlabel("Standardized Mean Difference")
        plt.ylabel("Features")
        plt.title("SMD Plot")
        plt.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def propensity_distribution_plot(self, segment_rules):
        df = self.load_segment(segment_rules)
        df_trim, _ = self.apply_overlap(df)
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df_trim,
            x=self.propensity_col,
            hue="is_algo",
            common_norm=False,
            element="step",
            fill=True,
            alpha=0.7,
        )
        plt.xlabel("Propensity")
        plt.ylabel("Count")
        plt.title("Propensity Score Distribution by Treatment Group")
        legend = plt.gca().get_legend()
        legend.set_title("is_algo")
        legend.texts[0].set_text("Organic (0)")
        legend.texts[1].set_text("Algorithmic (1)")
        plt.tight_layout()
        plt.show()
