# movie-review-sentiment-analyzer

## Introduction

This project aims to perform sentiment analysis on IMDB reviews to classify them as positive or negative. The process involves extensive data preprocessing, including text cleaning, tokenization, and lemmatization, followed by feature extraction using Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (Tf-IDF) methods. Various machine learning models—Logistic Regression, Support Vector Machine (SVM), and Naive Bayes—are then trained and evaluated on these features. Due to the large volume of data, PySpark is utilized to expedite the data processing, ensuring efficient handling and analysis.

## Data source

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

## Approach

### Data Preprocessing

**Data Preparation:**
- Load textual reviews into a Pandas DataFrame with appropriate labels (positive/negative).
- Save the staged DataFrame in a CSV format.

**Tokenization & Lemmatization:**
- Punctuation Removal: Strip out all punctuation from the reviews to ensure that the words are processed uniformly.
- Stop Words Removal: Eliminate common words (e.g., "and", "the", "is") that do not contribute to the sentiment analysis.
- Lowercasing: Convert all characters in the reviews to lowercase to maintain consistency and reduce the complexity of the vocabulary.

**Vectorization:**
To capture sentiment information in the textual data, we used Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (Tf-IDF), as they are more suitable for sentiment analysis than methods designed to capture semantic meaning.

- **Bag of Words (BoW):** Convert the textual data into a matrix of token counts. This method creates a vocabulary of all the unique words in the reviews and represents each review as a vector indicating the presence or absence (or frequency) of these words.
- **Term Frequency-Inverse Document Frequency (Tf-IDF):** Transform the reviews into a matrix where each term's importance is scaled by how often it appears across all reviews. This method accounts for the frequency of terms in each review and across the entire dataset, providing a balanced view of word importance.

### Model Building

- **Model Selection:**

  - Logistic Regression: A linear model used for binary classification that predicts the probability of a categorical dependent variable.
  - Support Vector Machine (SVM): A classifier that finds the optimal hyperplane which maximizes the margin between different classes.
  - Naive Bayes: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between features.

- **Training Process:**

  - Train each model on both the BoW and Tf-IDF datasets.
  - Split the data into training and testing sets to evaluate the performance of the models.
  - Tune hyperparameters for each model to optimize performance.
  - Evaluate the models using: accuracy, precision, recall, F1-score.

# Results

## Summary table

|                 | BoW               | Tf-IDF            |
|-----------------|-------------------|-------------------|
| **Logistic**    | Accuracy: 0.83    | Accuracy: 0.83    |
|                 | Precision: 0.83   | Precision: 0.83   |
|                 | Recall: 0.83      | Recall: 0.83      |
|                 | Recall: 0.83      | AUC: 0.83         |
|                 | F1-score: 0.83    | F1-score: 0.86    |
| **SVM**         | Accuracy: 0.86    | Accuracy: 0.86    |
|                 | Precision: 0.86   | Precision: 0.86   |
|                 | Recall: 0.86      | Recall: 0.86      |
|                 | F1-score: 0.83    | F1-score: 0.86    |
|                 | AUC: 0.93         | AUC: 0.93         |
| **Naive Bayes** | Accuracy: 0.85    | Accuracy: 0.82    |
|                 | Precision: 0.85   | Precision: 0.82   |
|                 | Recall: 0.85      | Recall: 0.82      |
|                 | F1-score: 0.85    | F1-score: 0.82    |
|                 | Recall: 0.85      | AUC: 0.82         |

## ROC-curves

**Modelling BoW vectorized data frame**

![Logistic Regression](/screenshots/bow-log.png)
![SVM](/screenshots/bow-svm.png)
![Nayev Bayes](/screenshots/bow-nay.png)

**Modelling Tf-IDF vectorized data frame**

![bow-log](/screenshots/tf-log.png)
![SVM](/screenshots/tf-svm.png)
![Nayev Bayes](/screenshots/tf-nay.png)

# Challenges

- The data processing stage was time-consuming due to the large volume of textual data.
- To address this, we utilized PySpark instead of scikit-learn for handling large datasets more efficiently. PySpark's distributed computing capabilities significantly reduced the processing time, making it more manageable for large-scale data.

# Conclusion

**Model Performance:**
- **Logistic Regression:**
  - BoW: Accuracy, Precision, Recall, F1-score all at 0.83.
  - Tf-IDF: Accuracy, Precision, Recall at 0.83, F1-score at 0.86, and AUC at 0.83.
- **SVM:**
  - BoW: Accuracy, Precision, Recall at 0.86, F1-score at 0.83, and AUC at 0.93.
  - Tf-IDF: Accuracy, Precision, Recall, F1-score all at 0.86, and AUC at 0.93.
- **Naive Bayes:**
  - BoW: Accuracy, Precision, Recall, F1-score all at 0.85.
  - Tf-IDF: Accuracy, Precision, Recall, F1-score all at 0.82, and AUC at 0.82.

**Data Processing Time:**
- The data processing stage was time-consuming due to the large volume of textual data.
- To address this, we utilized PySpark instead of scikit-learn for handling large datasets more efficiently. PySpark's distributed computing capabilities significantly reduced the processing time, making it more manageable for large-scale data.
**Data Processing Time:**
- The data processing stage was time-consuming due to the large volume of textual data.
- To address this, we utilized PySpark instead of scikit-learn for handling large datasets more efficiently. PySpark's distributed computing capabilities significantly reduced the processing time, making it more manageable for large-scale data.

# References

[1] Ochilbek Rakhmanov, "A Comparative Study on Vectorization and Classification Techniques in Sentiment Analysis to Classify Student-Lecturer Comments," *Procedia Computer Science*, vol. 178, pp. 194-204, 2020. ISSN: 1877-0509. DOI: [10.1016/j.procs.2020.11.021](https://doi.org/10.1016/j.procs.2020.11.021).

[2] İlhan Tarimer, Adil Çoban, and Arif Kocaman, "Sentiment Analysis on IMDB Movie Comments and Twitter Data by Machine Learning and Vector Space Techniques," 2019.

[3] Denis Cahyani and Irene Patasik, "Performance comparison of TF-IDF and Word2Vec models for emotion text classification," *Bulletin of Electrical Engineering and Informatics*, vol. 10, pp. 2780-2788, 2021. DOI: [10.11591/eei.v10i5.3157](https://doi.org/10.11591/eei.v10i5.3157).

