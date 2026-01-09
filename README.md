# Finding the Correlation



The data was collected from the online graph available at:

http://max.ge/aiml\_midterm/31698\_html



The graph shows blue data points. Each data point was inspected by hovering the mouse

over it and manually writing down the X and Y coordinates.



The collected values were saved in an Excel file and then loaded into Python using

the pandas library.



Pearson's correlation coefficient was calculated using the `.corr()` function in pandas.

The result shows a very strong positive correlation between the variables X and Y.

The correlation coefficient is approximately 0.999, which means that when X increases,

Y also increases almost linearly.



\### Visualization



The scatter plot below shows the relationship between X and Y values.

It clearly demonstrates a strong linear trend and supports the calculated correlation result.



!\[Scatter Plot](correlation/correlation\_scatter.png)



 Spam Email Detection

In this task, a spam email detection model was built using supervised machine learning.

The dataset contains 2500 email records with the following features:

words – number of words in the email
links – number of links
capital_words – number of capitalized words
spam_word_count – number of spam-related words
is_spam – target label (1 = spam, 0 = legitimate)

First, the data was loaded and explored using the pandas library.
Then the dataset was split into training (80%) and testing (20%) sets.

A Logistic Regression model was trained on the training data.
After training, the model was evaluated on the test set.

The evaluation results are:

Accuracy: 95.2%
The confusion matrix shows that most emails were classified correctly, with only a small number of misclassifications.

Two visualizations were created:

1. Class distribution graph – shows the number of spam and legitimate emails.
2. Confusion matrix plot – shows true and predicted classifications.

Finally, the trained model was used to classify new example emails, and it successfully identified spam and legitimate messages.
