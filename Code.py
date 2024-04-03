import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load data
main_FOLDER = 'C:/Users/Amir Arjmand/Documents/Photodetectors/Plots for dataextraction/12'
excelFileName = 'raman..xlsx'
file = main_FOLDER + '/' + excelFileName
table_vars = pd.read_excel(file)
table_vars.sort_values(by='x', inplace=True)
x = table_vars.x.values
y = table_vars.y.values

# Set parameter ranges
num_exp_terms_range = (20,20,1)  # range for Num_exp_terms
min_separation_range = (3,15,1) # range for MinSeparation
sharpness_range = np.linspace(0.0001, 20, 500)  #range for sharpness

best_r2 = -np.inf
best_params = {}

# Loop over parameter combinations
for num_exp_terms in num_exp_terms_range:
    for min_separation in min_separation_range:
        for sharpness in sharpness_range:
            # Find local extrema
            TF = argrelextrema(np.abs(y), np.greater, order=min_separation)[0]
            num_local_extrema = len(TF)

            # Adjust the number of exponential terms based on the number of local extrema
            num_exp_terms = min(num_exp_terms, num_local_extrema)

            # Prepare the design matrix
            x_unique, indices = np.unique(x, return_index=True)
            x_design = x_unique[np.argsort(indices)]  # Sort x values in ascending order
            shift_vals = x[TF][:num_exp_terms]
            X = np.column_stack((np.ones_like(x_design), 1.0 / x_design, np.exp(-sharpness * np.abs(x_design[:, np.newaxis] - shift_vals))))

            # Adjust the length of y to match X and shift_vals
            y_design = y[indices]
            y_design = y_design[:len(x_design)]

            # Perform linear regression
            reg_coefs, _, _, _ = np.linalg.lstsq(X, y_design, rcond=None)


            # Generate predicted values
            X_hat = np.column_stack((np.ones_like(x), 1.0 / x, np.exp(-sharpness * np.abs(x[:, np.newaxis] - shift_vals))))
            y_exp_hat = np.dot(X_hat, reg_coefs)

            # Calculate R-squared (R2) score
            r2 = r2_score(y, y_exp_hat)

            # Update best score and parameters if current score is better
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'Num_exp_terms': num_exp_terms, 'MinSeparation': min_separation, 'Sharpness': sharpness}
                best_reg_coefs = reg_coefs
                best_shift_vals = shift_vals

# Plotting the best fitted function and the exact data
num_exp_terms = best_params['Num_exp_terms']
min_separation = best_params['MinSeparation']
sharpness = best_params['Sharpness']

plt.figure(figsize=(8,6))
plt.plot(x, y, 'b-', label='Exact data')

# Generate additional x-values for the best fitted function
x_line = np.linspace(min(x), max(x), 1000)

# Calculate the corresponding y-values using the best fitted coefficients and shift values
X_line = np.column_stack((np.ones_like(x_line), 1.0 / x_line, np.exp(-sharpness * np.abs(x_line[:, np.newaxis] - best_shift_vals[:num_exp_terms]))))
y_best_fit = np.dot(X_line, best_reg_coefs)

# Plot the best fitted function
plt.plot(x_line, y_best_fit, 'g-', label='Best fitted function')

r2_text = f"R-squared score: {best_r2:.4f}"
params_text = f"exp_terms: {best_params['Num_exp_terms']:.1f}\nMinSeparation: {best_params['MinSeparation']:.1f}\nSharpness: {best_params['Sharpness']:.2f}"
plt.xlabel('x')
plt.ylabel('y')
plt.text(0.5, 0.75, r2_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.text(0.5, 0.7, params_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.grid(True)
plt.legend()
plt.savefig(r'C:\Users\Amir Arjmand\Documents\Photodetectors\Plots for dataextraction\12\output\XRD.png')
plt.show()
plt.close()


# Export regression coefficients and shift values of the best parameters
output_file_coeffs = r'C:\Users\Amir Arjmand\Documents\Photodetectors\Plots for dataextraction\12\output\XRDcoeff.csv'
output_file_shift_vals = r'C:\Users\Amir Arjmand\Documents\Photodetectors\Plots for dataextraction\12\output\XRDshift.csv'
df_coeffs = pd.DataFrame({'Coefficient': best_reg_coefs}).T
df_shift_vals = pd.DataFrame({'Shift Value': best_shift_vals[:num_exp_terms]}).T
df_coeffs.to_csv(output_file_coeffs, index=False)
df_shift_vals.to_csv(output_file_shift_vals, index=False)


print("Regression coefficients exported to", output_file_coeffs)
print("Shift values exported to", output_file_shift_vals)
