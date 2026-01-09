# Finding the Correlation & Spam Email Detection

This project contains two tasks completed for the AI/ML midterm assignment.

--------------------------------
1. Finding the Correlation
--------------------------------

The data was collected from the online graph:

http://max.ge/aiml_midterm/31698_html

The graph shows blue data points. Each point was inspected manually by hovering the mouse over it and writing down the X and Y coordinates.

The collected values were saved in an Excel file and loaded into Python using the pandas library.

Pearson’s correlation coefficient was calculated using the .corr() function.

Result:
The correlation coefficient is approximately 0.999, which shows a very strong positive linear correlation between X and Y.

Visualization:
A scatter plot was created to show the relationship between X and Y.

Scatter plot file:
correlation/correlation_scatter.png

--------------------------------
2. Spam Email Detection
--------------------------------

A spam email detection model was built using supervised machine learning.

Dataset:
The dataset contains 2500 emails with the following features:
- words
- links
- capital_words
- spam_word_count
- is_spam (1 = spam, 0 = legitimate)

Model:
- Logistic Regression
- 80% training data, 20% test data

Evaluation:
- Accuracy: 95.2%
- Confusion matrix shows very good classification performance

Visualizations:
- Class distribution: spam_detection/class_distribution.png
- Confusion matrix: spam_detection/confusion_matrix.png

--------------------------------
Manual Email Text Classification
--------------------------------

First, emails were classified manually by providing feature values directly to the model.

After that, a simple text parser was implemented.
The parser extracts features from raw email text:
- number of words
- number of links
- number of capital words
- number of spam-related words

Two email texts were written manually:
- A spam email with spam words, capital letters, and a link
- A legitimate email with neutral language and no spam patterns

The trained model correctly classified both emails.

--------------------------------
Conclusion
--------------------------------

Both tasks were completed successfully.
The project demonstrates correlation analysis and practical spam email detection using machine learning.
