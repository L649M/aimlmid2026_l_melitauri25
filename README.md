\# Finding the Correlation



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

