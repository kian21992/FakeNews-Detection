# Model Card

## Model Name
VeritasAI Fake News Detection Model

## Overview
This model is designed to help detect whether a news article or news-like text may be fake or misleading. It is part of the VeritasAI project.

## Purpose
The goal of the model is to support users in checking the credibility of news content. It is meant to assist analysis, not replace human judgment.

## Inputs
The model can take:
- raw news text
- article content from a URL
- source/domain information

## Outputs
The system returns:
- a prediction label such as likely real or likely fake
- a confidence score
- supporting signals such as subjectivity and domain reputation

## How It Works
The project uses a combination of:
- a RoBERTa-based fake news classifier
- subjectivity analysis
- domain reputation scoring
- text preprocessing and feature extraction

This combined approach is used to give a more informed prediction.

## Intended Use
This model is intended for:
- academic projects
- research prototypes
- classroom demonstrations
- decision support for reviewing online articles

## Not Intended For
This model should not be used for:
- making final legal or political judgments
- automatically accusing people or organizations of spreading false information
- replacing professional fact-checkers or journalists

## Performance
Initial evaluation metrics will be added as experiments are completed. These may include:
- accuracy
- precision
- recall
- F1-score

## Limitations
The model has several limitations:
- it may misclassify satire or opinion articles
- it may struggle with very recent breaking news
- it may be biased by the training data
- domain reputation does not always reflect truthfulness
- confidence score does not guarantee correctness

## Risks
Possible risks include:
- false positives, where real news is flagged as fake
- false negatives, where fake news is missed
- unfair treatment of lesser-known sources
- over-reliance by users on automated predictions

## Maintenance
This model may be updated over time using user feedback and retraining. Future versions should include better evaluation, more balanced datasets, and clearer reporting.