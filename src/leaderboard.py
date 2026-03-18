import pandas as pd
import numpy as np

class Leaderboard:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df = self.df[["segment","n_trimmed","ATE_like","ATE_dislike"]]

    def net_benefit(self, k=2):
        data = self.df.copy()
        data["NetBenefit"] = 1000 * (data["ATE_like"] - k * data["ATE_dislike"])
        return data["NetBenefit"]

    def impact_score(self, k=2):
        data = self.df.copy()
        data["NetBenefit"] = self.net_benefit(k)
        data["Impact"] = round((data["n_trimmed"] / 1000.0) * data["NetBenefit"], 3)
        return data["Impact"]
    
    def rank_segments(self, k=1, n=-1):
        data = self.df.copy()
        data["NetBenefit"] = self.net_benefit(k)
        data["Impact"] = self.impact_score(k)
        data = data.sort_values(by="Impact", ascending=False).reset_index(drop=True)
        if n != -1:
            data = data.head(n)
        return data 