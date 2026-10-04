# Task 3 – Simple Linear Regression Model

## Objective
Build a simple linear regression model to predict student marks from study hours using scikit-learn.

## Dataset
`study_marks.csv` contains 12 sample observations with:
- `Study_Hours` – independent variable (X)
- `Marks` – dependent variable (y)

## Method
1. Loaded the CSV using Pandas.
2. Selected study hours as the input feature and marks as the target.
3. Split the data into 75% training and 25% testing using `random_state=42`.
4. Trained `LinearRegression()` from scikit-learn.
5. Evaluated the model using MAE, RMSE and R².
6. Visualized the actual points and fitted regression line.

## Model
The learned equation is approximately:

**Marks = 41.82 + 4.74 × Study_Hours**

## Evaluation
- MAE: 2.600
- RMSE: 2.622
- R² Score: 0.983

## Interpretation
The positive coefficient means that, in this sample, marks tend to increase as study hours increase. The R² score indicates how much of the variation in marks is explained by study hours in the test data.

## Conclusion
The linear regression model provides a simple baseline for predicting marks from study time. In a real-world project, a larger dataset and additional variables such as attendance, previous marks and assignment performance could improve the prediction.
