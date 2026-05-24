# Paper Direction

## Working Title

Confident but Collapsed: A Context-Aware Analysis of Small Language Models for Rumour Stance Classification

Alternative:

Label Collapse and Context Sensitivity in Small-LM Rumour Stance Classification

## Core Claim

Small language models do not uniformly benefit from conversational context in RumourEval stance classification. They tend to recover easier labels such as query and majority-prior labels such as comment, while consistently failing on relation-dependent labels such as support. Context can improve query classification but does not solve support collapse, and models may remain overconfident under context-rich inputs.

This is a failure analysis paper, not a performance-improvement paper.

## Required Results

1. Majority baseline and label imbalance
2. Per-class F1 by model and context condition
3. Predicted label distribution for label collapse
4. Confidence, wrong/correct confidence, and ECE
5. Sample-level statistical significance tests
6. Dataset validity statistics for context construction

## Recommended Figures and Examples

1. Confusion matrix for `1.5B_adv c0`
2. Confusion matrix for `1.5B_adv c1`
3. Confusion matrix for `3B_zs c0` or `3B_zs c1`
4. Two or three support failure examples
5. Two or three query improvement examples

## Safe Interpretation

Adversarial context training changes prediction and confidence behavior, but it does not solve RumourEval stance classification. It does not eliminate support collapse. It may reduce overconfident predictions under context-rich inputs.

## Limitations

1. Absolute macro-F1 is low.
2. Support is rarely recovered.
3. RumourEval is label-imbalanced.
4. Context conditions are heuristic.
5. Mixed can resemble conflicting when no parent context exists.
6. Adversarial training is not a new algorithm.
7. Model coverage is limited.
8. Dev-only results should be described as dev results unless test is run.
