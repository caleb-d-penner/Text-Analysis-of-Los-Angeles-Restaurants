# Text-Analysis-of-Los-Angeles-Restaurants
# CLASS PROJECT — MARKETING 4640 - Text Marketing
# David Eccles School of Business - University of Utah
# Author - Caleb Penner
# Date - 2024-06-01
#Data is sourced from: https://www.kaggle.com/datasets/lorentzyeung/top-240-recommended-restaurants-in-la-2023
#Seed used for reproducibility: 123
###################################################

#This project analyzes Los Angeles restaurant reviews to uncover customer sentiment, dominant topics, geographic patterns, and predictive factors that influence Star Ratings.

#The workflow includes:
#Data cleaning and feature engineering
#Text preprocessing
#Sentiment analysis (lexicon‑based + aspect‑based)
#Topic modeling (LDA)
#Predictive modeling (Linear, Ridge, Lasso, Elastic Net)
#Spatial analysis of topics by ZIP code
#Final business‑focused visualizations

#The goal is to provide actionable insights for restaurant marketing and operations teams.


#Project Workflow Summary
#1. Data Cleaning & Feature Engineering
#Converts price into numeric values
#Converts cuisine style into categorical numeric codes
#Extracts ZIP codes from addresses

#2. Text Preprocessing
#Tokenization
#Lowercasing
#Stopword removal
#Stemming
#DFM creation and trimming

#3. Sentiment Analysis
#Lexicon‑based sentiment using Bing
#Net sentiment score per restaurant
#Aspect‑based sentiment using GPT 

#4. Topic Modeling
#LDA with Gibbs sampling
#K selection using ldatuning
#Extraction of top terms per topic
#Topic visualization
#Popular topics by restaurant types

#5. Predictive Modeling
#Models trained on TF‑IDF features:
#Linear Regression
#Ridge Regression
#Lasso Regression
#Elastic Net
#Models are evaluated on a test set

#6. Spatial Analysis
#Assigns dominant topic per review
#Aggregates topics by ZIP code
#Visualizes topic distribution geographically by ZIP code

#7. Final Visualizations
#Word cloud of the most common terms
#Topic prevalence
#Top terms per topic
#Topic–ZIP heatmap
#Aspect sentiment distribution
#Model performance comparison
#Feature importance from Elastic Net
