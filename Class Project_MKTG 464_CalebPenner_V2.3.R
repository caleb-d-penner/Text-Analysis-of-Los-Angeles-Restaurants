# Class Project - Text Analysis of Restaurant Reviews in Los Angeles
# Caleb Penner - Marketing 4640 
#--------------------------------------------------
# Load Libraries - Make sure the following packages are installed and loaded
#--------------------------------------------------

library(tidyverse)     # ggplot2, dplyr, stringr, tibble, etc.
library(tidytext)      # unnest_tokens, sentiment lexicons
library(textclean)     # text cleaning
library(textstem)      # lemmatization
library(textdata)      # sentiment lexicons
library(topicmodels)   # LDA
library(caret)         # ML models
library(viridis)       # color scales
library(quanteda)      # tokens + dfm
library(quanteda.textplots) # word clouds
library(ldatuning)     # topic tuning
library(stm)           # Structural Topic Models (if needed)
library(tokenizers)    # alternative tokenization
library(tm)            # text mining utilities
library(ellmer)        # Elastic Net utilities

#--------------------------------------------------
# Load Data - Adjust path to where your CSV file is located
#--------------------------------------------------

restaurantRaw <- read.csv("C:/Users/Caleb/OneDrive/Marketing 4640/Class Project/top 240 restaurants in los angeles.csv")

#--------------------------------------------------
# Data Exploration
#--------------------------------------------------

glimpse(restaurantRaw)
summary(restaurantRaw$StarRating)
summary(restaurantRaw$NumberOfReviews)

#Top 10 Restaurants by Number of Reviews
top10 <- restaurantRaw %>%
  count(RestaurantName, sort = TRUE) %>%
  slice_head(n = 10)

ggplot(top10, aes(x = reorder(RestaurantName, n), y = n)) +
  geom_col(fill = "#2c7bb6", width = 0.7) +
  coord_flip() +
  labs(
    title = "Top 10 Restaurants by Number of Reviews",
    x     = NULL,
    y     = "Number of Reviews"
  ) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor   = element_blank())



#--------------------------------------------------
# Feature Engineering
#--------------------------------------------------

# Price → numeric
restaurantRaw <- restaurantRaw %>%
  filter(!is.na(Price), Price != "") %>%
  mutate(PriceNumeric = str_count(Price, "\\$"))

# Style → numeric category
restaurantRaw <- restaurantRaw %>%
  mutate(StyleCategory = as.numeric(as.factor(Style)))

#--------------------------------------------------
# Text Preprocessing
#--------------------------------------------------
restTokens <- quanteda::tokens(restaurantRaw$Comment, what = "word",
  remove_numbers = TRUE,
  remove_punct = TRUE,
  remove_symbols = TRUE)

#Create DFM, but remove terms which occur in less than 2% of all documents and more than 98%
#Clean up this section. Stopword removal and lower should be in the code above.

restDFM <- restTokens %>% 
  tokens_tolower() %>% 
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm() %>%
  dfm_trim(min_docfreq = 0.02, max_docfreq = 0.98, docfreq_type = "prop")
# Return a useful object
restMatrix <- as.matrix(restDFM)
restMatrix[1:3,1:8]

#take a look at how many 0s we have in the DFM as a propportion of the total entries to get a sense of sparsity
sum(restMatrix == 0) / (nrow(restMatrix) * ncol(restMatrix))

#HIGH SPARSITY ~94% According to AI, sparsity of 90-99% is common in TXT data
#Clean empty DFM rows to prepare for LDA
restDFM <- restDFM[ntoken(restDFM) > 0, ]

restaurantRaw$doc_id <- as.character(1:nrow(restaurantRaw))
docnames(restDFM) <- as.character(seq_len(ndoc(restDFM)))

#--------------------------------------------------
# Sentiment Analysis
#--------------------------------------------------

bing <- get_sentiments("bing")

rest_words <- restaurantRaw %>%
  select(RestaurantName, Comment) %>%
  unnest_tokens(word, Comment)

restSentiment <- rest_words %>%
  inner_join(bing, by = "word") %>%
  count(RestaurantName, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(net_sentiment = positive - negative)

#Insert net sentiment back into the main data frame for later analysis
restaurantRaw <- restaurantRaw %>%
  left_join(restSentiment, by = "RestaurantName")

#--------------------------------------------------
# Topic Modeling
#--------------------------------------------------

# Tune to find optimal K value
result <- FindTopicsNumber(
  restDFM,
  topics = seq(2, 25, 3),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 1971),
  mc.cores = 2,
  return_models = TRUE
)

FindTopicsNumber_plot(result)

K <- 7 # chosen from tuning (Range is 7-10 based on metrics)

#Run LDA, 500 iterations with Gibbs sampling, set seed for reproducibility
restLDA <- LDA(restDFM, K, method = "Gibbs",
               control = list(seed = 123, iter = 500))


restTopics <- tidy(restLDA, matrix = "beta")

restTopTerms <- restTopics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  arrange(topic, -beta)

print(restTopTerms)

#Popular Topics by Restaurant Type

#Check that docnames of DFM match with doc_id in restaurantRaw for joining
restaurantRaw$doc_id <- as.character(1:nrow(restaurantRaw))
docnames(restDFM) <- restaurantRaw$doc_id[ntoken(restDFM) > 0]

tidy_terms <- tidy(restDFM) %>%
  rename(doc_id = document)
terms_with_style <- tidy_terms %>%
  left_join(restaurantRaw %>% select(doc_id, Style), by = "doc_id")

#Most common terms by type
common_terms_by_style <- terms_with_style %>%
  group_by(Style, term) %>%
  summarise(total = sum(count), .groups = "drop") %>%
  arrange(Style, desc(total))

top_terms_by_style <- common_terms_by_style %>%
  group_by(Style) %>%
  slice_max(order_by = total, n = 10) %>%
  ungroup()

print(top_terms_by_style)

#--------------------------------------------------
#Bi-Gram LDA
#--------------------------------------------------
# Create bigram tokens using quanteda
biTokens <- restaurantRaw %>%
  select(RestaurantName, Comment) %>%
  unnest_tokens(bigram, Comment, token = "ngrams", n = 2)

# Separate bigrams into two words
bigrams_separated <- biTokens %>%
  separate(bigram, into = c("word1", "word2"), sep = " ")

# Now you can filter out stop words
filteredBiGram <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word)

restBiDFM <- filteredBiGram %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(RestaurantName, bigram) %>%
  cast_dfm(RestaurantName, bigram, n)

#Cut off for bigrams that occur in less than 2% of documents and more than 98%
restBiDFM <- dfm_trim(restBiDFM, min_docfreq = 0.02, max_docfreq = 0.98, docfreq_type = "prop")

# Convert to topicmodels format
restBiDTM <- convert(restBiDFM, to = "topicmodels")

# Remove empty documents if any
restBiDTM <- restBiDTM[rowSums(as.matrix(restBiDTM)) > 0, ]

# Fit LDA model 
restBiLDA <- LDA(restBiDTM, k = 7)
restBiTopics <- tidy(restBiLDA, matrix = "beta")

#Object of top bigrams for each topic
restBiTopTerms <- restBiTopics %>%
  group_by(topic) %>%
  slice_max(beta, n = 5) %>% 
  arrange(topic, -beta)

#Visualize top 5 bigrams for each topic
#Not a great output, stopwords are still included in bigrams. How would we fix this?
restBiTopTerms %>%
  mutate(term = reorder(term, beta)) %>% 
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free")

#Visualize top terms by restaurant type
#--------------------------------------------------
# 1. Sample the top 6 most common restaurant types
#--------------------------------------------------
set.seed(123)

sampled_styles <- restaurantRaw %>%
  count(Style, sort = TRUE) %>%
  slice_head(n = 6) %>%
  pull(Style)

#--------------------------------------------------
# 2. Filter term table to only those sampled styles
#    AND remove common words that do not add insight (e.g., "food", "good", "restaurant", "great")
#--------------------------------------------------
filtered_terms <- common_terms_by_style %>%
  filter(
    Style %in% sampled_styles,
    !term %in% c("food", "good", "like", "us", "realli", "one", "came", "lot", "get", "just", "come", "go", "also", "servic","back", "restaurant","restaur", "great", "order", "place")
  )

#--------------------------------------------------
# 3. Now extract EXACTLY the top 3 terms per style
#--------------------------------------------------
top3_terms_by_style <- filtered_terms %>%
  group_by(Style) %>%
  slice_max(order_by = total, n = 5, with_ties = FALSE) %>%
  ungroup()

#--------------------------------------------------
# 4. Plot
#--------------------------------------------------
ggplot(top3_terms_by_style,
       aes(x = total,
           y = reorder_within(term, total, Style))) +
  geom_col(fill = "steelblue") +
  facet_wrap(~ Style, scales = "free_y", ncol = 2) +
  scale_y_reordered(expand = expansion(mult = c(0.15, 0.15))) +
  labs(title = "Top 5 Terms by Sampled Restaurant Types",
       x = "Frequency",
       y = "Term") +
  theme_minimal(base_size = 14) +
  theme(
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(size = 12, margin = margin(b = 6)),
    panel.spacing = unit(1.5, "lines")
  )

#--------------------------------------------------
# Regression Analysis (Ridge, Lasso, Elastic Net)
#--------------------------------------------------
trainIndex <- createDataPartition(restaurantRaw$StarRating, p = 0.70, list = FALSE)

train_set <- restaurantRaw[trainIndex, ]
test_set  <- restaurantRaw[-trainIndex, ]

# Build DFM for regression
restTrainDFM <- tokens(train_set$Comment,
                       remove_numbers = TRUE,
                       remove_punct = TRUE,
                       remove_symbols = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm() %>%
  dfm_trim(min_termfreq = 10, min_docfreq = 2) %>%
  dfm_tfidf()

restTrainDF <- as.data.frame(as.matrix(restTrainDFM))
restTrainDF$StarRating <- train_set$StarRating

restCvCntrl <- trainControl(method = "cv", number = 5)

# Linear Regression
restLMModel <- train(StarRating ~ ., data = restTrainDF, method = "lm", trControl = restCvCntrl)

# Ridge
restRidge <- train(
  StarRating ~ ., data = restTrainDF, method = "glmnet",
  trControl = restCvCntrl,
  tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4, 1, length = 100))
)

# Lasso
restLassoModel <- train(
  StarRating ~ ., data = restTrainDF, method = "glmnet",
  trControl = restCvCntrl,
  tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, 1, length = 100))
)

# Elastic Net
restEnet <- train(
  StarRating ~ ., data = restTrainDF, method = "glmnet",
  trControl = restCvCntrl,
  tuneLength = 15
)


#--------------------------------------------------
# TEST SET EVALUATION FOR REGRESSION MODELS
#--------------------------------------------------
#Build Test DFM

restTestDFM <- tokens(
  test_set$Comment,
  remove_numbers = TRUE,
  remove_punct = TRUE,
  remove_symbols = TRUE
) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_wordstem() %>%
  dfm() %>%
  dfm_match(features = featnames(restTrainDFM)) %>%   # CRITICAL STEP
  dfm_tfidf()

restTestDF <- as.data.frame(as.matrix(restTestDFM))
restTestDF$StarRating <- test_set$StarRating

#--------------------------------------------------
# Generate Predictions on Test Set
#--------------------------------------------------

pred_lm    <- predict(restLMModel,    restTestDF)
pred_ridge <- predict(restRidge,      restTestDF)
pred_lasso <- predict(restLassoModel, restTestDF)
pred_enet  <- predict(restEnet,       restTestDF)

#--------------------------------------------------
#Compute Test Metrics
#--------------------------------------------------

metrics_lm    <- postResample(pred_lm,    restTestDF$StarRating)
metrics_ridge <- postResample(pred_ridge, restTestDF$StarRating)
metrics_lasso <- postResample(pred_lasso, restTestDF$StarRating)
metrics_enet  <- postResample(pred_enet,  restTestDF$StarRating)

#--------------------------------------------------
#Comparison Table
#--------------------------------------------------

comparison_df <- tibble(
  Model = c("Linear Regression", "Ridge", "Lasso", "Elastic Net"),
  RMSE = c(metrics_lm["RMSE"], metrics_ridge["RMSE"],
           metrics_lasso["RMSE"], metrics_enet["RMSE"]),
  Rsquared = c(metrics_lm["Rsquared"], metrics_ridge["Rsquared"],
               metrics_lasso["Rsquared"], metrics_enet["Rsquared"]),
  MAE = c(metrics_lm["MAE"], metrics_ridge["MAE"],
          metrics_lasso["MAE"], metrics_enet["MAE"])
)

comparison_df

#--------------------------------------------------
# Top Predictive Terms from Ridge (Most Predictive Overall Model)
#--------------------------------------------------
best_lambda_ridge <- restRidge$bestTune$lambda

coef_ridge <- coef(restRidge$finalModel, s = best_lambda_ridge)

coef_df_ridge <- data.frame(
  term = rownames(as.matrix(coef_ridge)),
  coefficient = as.numeric(coef_ridge)
)

coef_df_ridge <- coef_df_ridge[coef_df_ridge$term != "(Intercept)", ]

top_positive_ridge <- coef_df_ridge %>%
  arrange(desc(coefficient)) %>%
  head(20)

top_positive_ridge

#--------------------------------------------------
# Identify Best Model
#--------------------------------------------------

best_model_idx <- which.min(comparison_df$RMSE)
cat("\nBest Model on TEST SET:", comparison_df$Model[best_model_idx], "\n")

#--------------------------------------------------
# LLM Aspect Based Sentiment Analysis
#--------------------------------------------------
chat_aspect <- chat_openai("You are an AI assistant that performs aspect-based sentiment analysis.
For each hotel review provided, identify the key aspects mentioned (e.g., room, staff, location, cleanliness, etc.).
For each aspect, determine the sentiment expressed by the reviewer toward that aspect (Positive, Negative, or Neutral).
Respond with a JSON array of objects, where each object has an 'aspect' and the corresponding 'sentiment'.",
  model = "gpt-4.1-mini"  # using GPT-4 (or choose available model)
)

aspect_schema <- type_array(type_object(
  aspect    = type_string(), 
  sentiment = type_enum(c("Positive", "Negative", "Neutral"))
))

aspect_prompts <- as.list(restaurantRaw$Comment)

#Run LLM in parallel to speed up processing of aspect-based sentiment analysis across all reviews  
aspect_results <- parallel_chat_structured(chat_aspect, aspect_prompts, type = aspect_schema)
restaurantRaw$aspects <- aspect_results

aspect_sentiments <- restaurantRaw %>% 
  select(Comment, aspects) %>% 
  unnest(aspects)
head(aspect_sentiments)

aspect_summary <- aspect_sentiments %>%
  group_by(aspect) %>%
  summarise(
    total_mentions = n(),
    positive_mentions = sum(sentiment == "Positive"),
    negative_mentions = sum(sentiment == "Negative"),
    neutral_mentions  = sum(sentiment == "Neutral")
  ) %>%
  ungroup() %>%
  arrange(desc(total_mentions))
head(aspect_summary, 10)

top5_negative_aspects <- aspect_summary %>% arrange(desc(negative_mentions)) %>% slice(1:5) %>% pull(aspect)
top5_positive_aspects <- aspect_summary %>% arrange(desc(positive_mentions)) %>% slice(1:5) %>% pull(aspect)

top5_negative_aspects
top5_positive_aspects
top_aspects <- aspect_summary %>% slice_max(order_by = total_mentions, n = 10)

ggplot(top_aspects, aes(x = reorder(aspect, total_mentions), y = total_mentions)) +
  geom_col(fill="steelblue") +
  coord_flip() +
  labs(title = "Top 10 Most Mentioned Aspects in Reviews",
       x = "Aspect", y = "Number of Mentions") +
  theme_minimal()


top_aspect_names <- top_aspects$aspect
aspect_sentiments_top <- aspect_sentiments %>% filter(aspect %in% top_aspect_names)

ggplot(aspect_sentiments_top, aes(x = aspect, fill = sentiment)) +
  geom_bar(position = "fill") +  # fill to show proportions
  coord_flip() +
  scale_fill_manual(values=c("Negative"="#F8766D", "Neutral"="gray80", "Positive"="#00BFC4")) +
  labs(title = "Sentiment Distribution for Top Aspects",
       x = "Aspect", y = "Proportion of Mentions") +
  theme_minimal()


#--------------------------------------------------
# Spatial Analysis (Topic by ZIP)
#--------------------------------------------------

restaurantRaw$zip <- str_extract(restaurantRaw$Address, "\\d{5}$")
restaurantRaw$doc_id <- 1:nrow(restaurantRaw)

lda_posterior <- posterior(restLDA)

doc_topics <- lda_posterior$topics %>%
  as.data.frame() %>%
  mutate(doc_id = 1:nrow(.),
         topic = apply(., 1, which.max))

lda_zip <- doc_topics %>%
  left_join(restaurantRaw %>% select(doc_id, zip), by = "doc_id") %>%
  filter(!is.na(zip))

topics_by_zip <- lda_zip %>%
  count(zip, topic)

ggplot(topics_by_zip, aes(x = zip, y = factor(topic), fill = n)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c() +
  labs(x = "ZIP Code", y = "Topic", fill = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#FINAL VISUALIZATIONS
#Word cloud of most common terms
#Topic prevalence
#Top terms per topic
#Topic–ZIP heatmap
#Feature importance from Elastic Net

textplot_wordcloud(restDFM,
  min_size     = 0.5,
  max_size     = 4.5,
  min_count    = 15,
  max_words    = 150,
  color        = c("gray60", "steelblue", "navy"),
  random_order = FALSE,
  rotation     = 0.1
)

# Topic prevalence
topic_prevalence <- tidy(restLDA, matrix = "gamma") %>%
  count(topic, wt = gamma) %>%
  mutate(topic = factor(topic))

ggplot(topic_prevalence, aes(x = topic, y = n, fill = topic)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Most Common Review Topics",
       x = "Topic", y = "Weighted Frequency") +
  theme_minimal()

#Top Words Per Topic
restTopTerms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top Words Defining Each Topic",
       x = "Importance (β)", y = "Term") +
  theme_minimal()

#Feature importance from Elastic Net
enet_importance <- varImp(restEnet)$importance %>%
  rownames_to_column(var = "term") %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 20)
enet_importance %>%
  ggplot(aes(x = Overall, y = reorder(term, Overall))) +
  geom_col(fill = "steelblue") +
  labs(title = "Top 20 Most Predictive Terms (Elastic Net)",
       x = "Importance", y = "Term") + 
  theme_minimal()
