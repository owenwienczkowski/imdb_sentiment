Distilbert Model

              precision    recall  f1-score   support

           0       0.89      0.92      0.90       250
           1       0.92      0.88      0.90       250

    accuracy                           0.90       500
   macro avg       0.90      0.90      0.90       500
weighted avg       0.90      0.90      0.90       500


Tabularisai Model

-- multiclass categorization with binary labels 

              precision    recall  f1-score   support

           0       0.95      0.37      0.53       250
           1       0.18      0.04      0.07       250
           2       0.00      0.00      0.00         0
           3       0.00      0.00      0.00         0
           4       0.00      0.00      0.00         0

    accuracy                           0.20       500
   macro avg       0.22      0.08      0.12       500
weighted avg       0.56      0.20      0.30       500

-- bucket categorization with neutrals dropped

Number of entries dropped (not negative or positive):  154
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       154
           1       1.00      1.00      1.00       192

    accuracy                           1.00       346
   macro avg       1.00      1.00      1.00       346
weighted avg       1.00      1.00      1.00       346