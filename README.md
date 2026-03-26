# YandexMusic segmentation analysis

This project builds on [Yandex Music recommendation algorithm analysis](https://github.com/kirillous/YAMBDA-algorithm-analysis) and looks at where recommendations work best rather than whether they work overall. Using the same YAMBDA dataset and causal framework, the analysis estimates how recommendation effects vary across different user and item segments, and translates those results into simple policy suggestions.

## Key Findings

- Recommendations are most effective for average users and popular content, where they consistently generate positive engagement with relatively low negative feedback.

- Highly engaged users consuming popular items show the strongest uplift, with very large gains in likes and minimal increase in dislikes.

- Some segments are more sensitive to negative feedback. In particular, low-engagement users exposed to new content experience a higher increase in dislikes relative to likes.

## Project Details

### Goal

The goal of this project is to identify “sweet spots” where algorithmic recommendations deliver the most value. Instead of looking at average effects across all users, the analysis estimates segment-specific impacts and converts them into clear recommendations (e.g., where to increase or reduce algorithmic exposure).

### Methods

The same observational causal framework from [original analysis](https://github.com/kirillous/YAMBDA-algorithm-analysis) is used. Each listen is treated as a decision (algorithmic vs organic), and IPTW propensity score weighting is applied to adjust for non-random assignment.

Users and items are grouped into interpretable segments based on:

- User activity level

- Item popularity

- User like behavior

For each segment, the analysis estimates:

- Change in likes per 1,000 listens

- Change in dislikes per 1,000 listens

This allows direct comparison of positive and negative effects across different contexts.

The analysis also introduces a simple trade-off framework, showing how conclusions change depending on how much the business penalizes negative feedback (dislikes).

### Results

Results show that recommendation impact varies significantly across segments. Segments combining average users and popular content consistently deliver strong positive results, with clear gains in likes and relatively small increases in dislikes. In contrast, segments involving new items and low-engagement users show weaker performance, with dislikes increasing more noticeably. While top-performing segments remain robust, lower-performing ones can become neutral or negative under stricter conditions.

### Limitations

As in the [original analysis](https://github.com/kirillous/YAMBDA-algorithm-analysis), this project is based on the same observational data, not randomized experiments. The results rely on the assumption that all relevant confounding factors are captured in the model. The outcomes focus on short-term feedback (likes/dislikes within 24 hours) and do not account for long-term effects such as user retention or satisfaction over time. Segment-level estimates may also be sensitive to how segments are defined and to the chosen trade-off between likes and dislikes.

## Reproducibility

The analysis is implemented in Python using Jupyter notebooks. The main steps are:

- segments.ipynb - define and explore user/item segments

- segment_est.ipynb - estimate causal effects by segment

- leaderboard.ipynb - compare segments under different dislike penalties

Scripts must be run in that order and as in [original analysis](https://github.com/kirillous/YAMBDA-algorithm-analysis) the full dataset must be obtained separately.

## Sources

https://huggingface.co/datasets/yandex/yambda

[Yandex Music recommendation algorithm analysis](https://github.com/kirillous/YAMBDA-algorithm-analysis)

## Contact

Author: Kirill Markin

Email: kirill.markin18@gmail.com

GitHub: kirillous
