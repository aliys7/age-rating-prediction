# Predicting Content Age Rating Using NLP

**Short description:** Predict multi-class age ratings for movies and TV shows (MPAA / TV Parental Guidelines) from content metadata (title + description) using text features. Aimed at the cold-start scenario (no interaction data).

## Overview
This repository contains the dataset and notebook for predicting content age ratings using NLP + ML. I used the Amazon Prime metadata (`amazon_prime_titles.csv`) and a pipeline that includes text cleaning, TF-IDF, PCA (dimensionality reduction), SMOTE (handle class imbalance), and several classifiers (Random Forest, Logistic Regression, Decision Tree, Naïve Bayes).

## Key findings
- **Best model:** Random Forest (highest accuracy).  
  - Movies: **~41%** accuracy.  
  - TV shows: **~50%** accuracy.

> These scores are modest but expected given the multi-class, imbalanced nature of the problem and limited content features (no user interaction signals). See the notebook for confusion matrices and per-class metrics.

# Data source

**Dataset:** Amazon Prime Movies and TV Shows (Kaggle)  
**Kaggle page:** https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows
