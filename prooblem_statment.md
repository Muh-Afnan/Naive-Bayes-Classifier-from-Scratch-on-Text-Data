# Day 20 — Naive Bayes Classifier from Scratch on Text Data

## Problem Statement
Build a Naive Bayes text classifier from scratch that learns to classify 
documents by class (e.g. spam vs not spam) using word frequency statistics. 
No sklearn. Pure Python implementation of Bayes theorem applied to text.

## Core Questions
- How does Bayes theorem apply to text classification?
- Why is the independence assumption "naive" and why does it work anyway?
- What is Laplace smoothing and why is it needed?
- Why use log probabilities instead of raw probabilities?

## Requirements
- NaiveBayes with fit, predict, predict_proba
- Laplace smoothing for unseen words
- Log probability scoring to prevent underflow
- Three visualizations: word probabilities, class distribution, confusion matrix