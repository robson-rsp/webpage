from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np



def show_scores(y_true, y_pred):
    print(f'Mean squared error:      {mean_squared_error(y_true, y_pred): .2f}')
    print(f'Mean root squared error: {mean_squared_error(y_true, y_pred, squared=False): .2f}')
    print(f'Mean absolute error:     {mean_absolute_error(y_true, y_pred): .2f}')
    
    

def show_scores_cross_val(scores):
    rmse_scores = np.sqrt(-scores)
    print(f'Scores: {rmse_scores} \n')
    print(f'Mean scores: {rmse_scores.mean(): .2f}')
    print(f'STD scores:  {rmse_scores.std(): .2f}')