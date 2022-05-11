# TESTING NORMAL DISTRIBUTION OF RESIDUALS 

def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    return df_results

def linear_assumption(model, features, label):
    """
    Linearity: Assumes that there is a linear relationship between the predictors and
               the response variable. If not, either a quadratic term or another
               algorithm should be used.
    """
    print('Assumption 1: Linear Relationship between the Target and the Feature', '\n')
        
    print('Checking with a scatter plot of actual vs. predicted.',
           'Predictions should follow the diagonal line.')
    
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)
    
    # Plotting the actual vs predicted values
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)
        
    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()

def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('Assumption 2: The error terms are normally distributed', '\n')
    
    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    sns.distplot(df_results['Residuals'])
    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')

def homoscedasticity_assumption(model, features, label):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('Assumption 5: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
        
    # Calculating residuals for the plot
    df_results = calculate_residuals(model, features, label)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  

def autocorrelation_assumption(model, features, label):
    """
    Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                     autocorrelation, then there is a pattern that is not explained due to
                     the current value being dependent on the previous value.
                     This may be resolved by adding a lag variable of either the dependent
                     variable or some of the predictors.
    """
    from statsmodels.stats.stattools import durbin_watson
    print('Assumption 4: No Autocorrelation', '\n')
    
    # Calculating residuals for the Durbin Watson-tests
    df_results = calculate_residuals(model, features, label)

    print('\nPerforming Durbin-Watson Test')
    print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
    print('0 to 2< is positive autocorrelation')
    print('>2 to 4 is negative autocorrelation')
    print('-------------------------------------')
    durbinWatson = durbin_watson(df_results['Residuals'])
    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')


# TESTING ASSUMPTIONS ######################################################

#import pandas as pd
#diction = {}
#diction['density'] = density
#diction['age'] = age
#diction['BAME'] = BAME
#diction['deprivation'] = deprivation
#print(diction)


#datasetx = covid_dataframe[['People per sqm', 'Median age', 'Proportion of BAME', 'Proportion of LSAOs with most deprivated neighbourhoods']]
datasetx = covid_dataframe[['Median age', 'Proportion of LSAOs with most deprivated neighbourhoods']]

#print(datasetx.head())
#datasetx['Proportion of LSAOs with most deprivated neighbourhoods'] = np.log([1 + i for i in covid_dataframe['Proportion of LSAOs with most deprivated neighbourhoods']])
#datasetx['People per sqm'] = np.log(datasetx['People per sqm'])
#datasetx['Proportion of BAME'] = np.log(datasetx['Proportion of BAME'])
#print(datasetx.head())

#datasetx = pd.DataFrame(diction)

#deprivation = covid_dataframe['Proportion of LSAOs with most deprivated neighbourhoods']
#deprivation = [1 + i for i in covid_dataframe['Proportion of LSAOs with most deprivated neighbourhoods']]
#deprivation, fitted_lambda = stats.boxcox(deprivation)
#deprivation = np.array(deprivation).reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#regressor.fit(deprivation, infection_rate)
regressor.fit(datasetx, infection_rate)

#Y_pred = regressor.predict(deprivation)
#r2 = regressor.score(deprivation, infection_rate)
Y_pred = regressor.predict(datasetx)
r2 = regressor.score(datasetx, infection_rate)
print(regressor.coef_)
print(r2)


linear_assumption(regressor, datasetx, infection_rate)
normal_errors_assumption(regressor, datasetx, infection_rate)
autocorrelation_assumption(regressor, datasetx, infection_rate)
homoscedasticity_assumption(regressor, datasetx, infection_rate)
