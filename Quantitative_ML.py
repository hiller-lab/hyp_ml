import pandas as pd
import random
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#Tested models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'KNeighbors': KNeighborsRegressor(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Bagging': BaggingRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(random_state=42),
}

results_r2 = {name: [] for name in models.keys()}
results_mse = {name: [] for name in models.keys()}
results_mape = {name: [] for name in models.keys()}
results_rmse = {name: [] for name in models.keys()}
feature_importances = {name: np.zeros(len(['aiso (MHz)', 'delta g', 'Nucleophilicity index', 'Adiabatic Ionization Potential (eV)',
            'Negative Fukui Index', 'logP', 'LUMO-HOMO (hartree)', 'Geminate polarization probability Q'])) for name in models.keys()}

iterations = 100

output_file = "Predictions_results.xlsx" #Predicted SNE for all models, molecules and iterations
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

all_true_values = {name: [] for name in models.keys()}
all_predicted_values = {name: [] for name in models.keys()}

for iteration in range(iterations):

    # Excel input file
    file_path = './Input_file_all_data.xlsx'
    sheet_name = 'Data'
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    features = ['Molecule', 'Atom', 'aiso (MHz)', 'delta g', 'Nucleophilicity index', 'Adiabatic Ionization Potential (eV)',
            'Negative Fukui Index', 'logP', 'LUMO-HOMO (hartree)', 'Geminate polarization probability Q']
    selected_features = features[2:]
    target_variable = 'Absolute SNE'
    set_molecules = ['2M5F', 'I', '4F', '5F', '6F', '7F', '4M', '5M', '6M', '7M', '4OH', '5OH', '6OH', '7OH', '4COOH',
                     '5COOH', '6COOH', '7COOH',
                     '4N', '5N', '6N', '7N', '4OM', '5OM', '6OM', '7OM', 'Y', '4FOH', '34OHF', '3NY', '3MOH', '3OHCOOH',
                     '4MOH', '4OMOH', '2MOH', 'W', '3FY', '4OHCOOH', '2M6F', 'P']

    random.shuffle(set_molecules)
    train_molecules = []
    val_molecules = []
    test_molecules = []

    # Split the shuffled molecules
    test_molecules = set_molecules[:5]
    val_molecules = set_molecules[5:10]
    train_molecules = set_molecules[10:]

    train_data = data[data['Molecule'].isin(train_molecules)]
    val_data = data[data['Molecule'].isin(val_molecules)]
    test_data = data[data['Molecule'].isin(test_molecules)]

    X_train = train_data[selected_features]
    X_val = val_data[selected_features]
    X_test = test_data[selected_features]

    Y_train = train_data[target_variable].values
    Y_val = val_data[target_variable].values
    Y_test = test_data[target_variable].values

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    results_df = pd.DataFrame({'Molecule': test_data['Molecule'], 'Atom': test_data['Atom'], 'True_SNE': Y_test})

    for name, model in models.items():
        model.fit(X_train_scaled, Y_train)

        Y_val_pred = model.predict(X_val_scaled)
        val_r2 = r2_score(Y_val, Y_val_pred)
        val_mse = mean_squared_error(Y_val, Y_val_pred)
        val_mape = np.mean(np.abs((Y_val - Y_val_pred) / Y_val) * 100)
        val_rmse = np.sqrt(val_mse)

        print(f'Validation performance for {name}:')
        print(f'R-squared: {val_r2}, MSE: {val_mse}, MAPE: {val_mape}, RMSE: {val_rmse}')

        Y_test_pred = model.predict(X_test_scaled)
        results_df[name] = Y_test_pred

        all_true_values[name].extend(Y_test)
        all_predicted_values[name].extend(Y_test_pred)

        r2_test = r2_score(Y_test, Y_test_pred)
        mse_test = mean_squared_error(Y_test, Y_test_pred)
        mape_test = np.mean(np.abs((Y_test - Y_test_pred) / Y_test) * 100)
        rmse_test = np.sqrt(mse_test)

        results_r2[name].append(r2_test)
        results_mse[name].append(mse_test)
        results_mape[name].append(mape_test)
        results_rmse[name].append(rmse_test)

        print(f'Model: {name}')
        print(f'Test set Mean Squared Error: {mse_test}')
        print(f'Test set R-squared: {r2_test}')
        print(f'Test set Mean Absolute Percentage Errors: {mape_test}')
        print(f'Training set: {train_molecules}')
        print(f'Test set: {test_molecules}')
        print(f'Validation set: {val_molecules}')
        print(f'Test set RMSE: {rmse_test}')

        # Collect feature importances
        if hasattr(model, 'feature_importances_'):  # For tree-based models
            feature_importances[name] += model.feature_importances_
        elif hasattr(model, 'coef_'):  # For linear models
            feature_importances[name] += np.abs(model.coef_)
        else:  # For other models
            perm_importance = permutation_importance(model, X_test_scaled, Y_test, n_repeats=30, random_state=42)
            feature_importances[name] += perm_importance.importances_mean

    results_df.to_excel(writer, sheet_name=f'Iteration_{iteration + 1}', index=False)
writer.close()

for name in models.keys():
    feature_importances[name] /= iterations

plt.rcParams.update({'font.size': 4.5})

#Print Mean R-squared, MSE and MAPE
for name in models.keys():
    print(f"{name} - Mean R-squared: {np.mean(results_r2[name])}, Std R-squared: {np.std(results_r2[name])}")
    print(f"{name} - Mean MSE: {np.mean(results_mse[name])}, Std MSE: {np.std(results_mse[name])}")
    print(f"{name} - Mean MAPE: {np.mean(results_mape[name])}, Std MAPE: {np.std(results_mape[name])}")

#Plot feature importances
for name in models.keys():
    plt.figure(figsize=(7, 4))
    plt.barh(selected_features, feature_importances[name])
    plt.xlabel('Average Importance')
    plt.title(f'Feature Importance for {name}')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.savefig(f'{name}_importance.pdf', format='pdf', bbox_inches='tight')

# Compute statistical parameters
performance_stats_df = pd.DataFrame()
performance_stats_df['Model'] = list(models.keys())
for metric, results in [('R-squared', results_r2), ('MSE', results_mse), ('MAPE', results_mape), ('RMSE', results_rmse)]:
    mean_values = []
    std_values = []
    min_values = []
    max_values = []
    range_values = []
    cv_values = []
    outliers_count = []

    for name in models.keys():
        mean_val = np.mean(results[name])
        std_val = np.std(results[name])
        min_val = np.min(results[name])
        max_val = np.max(results[name])
        range_val = max_val - min_val
        cv_val = std_val / mean_val if mean_val != 0 else 0

        mean_values.append(mean_val)
        std_values.append(std_val)
        min_values.append(min_val)
        max_values.append(max_val)
        range_values.append(range_val)
        cv_values.append(cv_val)

    performance_stats_df[f'{metric}_Mean'] = mean_values
    performance_stats_df[f'{metric}_Std'] = std_values
    performance_stats_df[f'{metric}_Min'] = min_values
    performance_stats_df[f'{metric}_Max'] = max_values
    performance_stats_df[f'{metric}_Range'] = range_values
    performance_stats_df[f'{metric}_CV'] = cv_values

print(performance_stats_df)

output_file_path = 'Model_performance.xlsx'# Detailed statistics for all models
performance_stats_df.to_excel(output_file_path, index=False)

print(f"Performance saved to {output_file_path}")

# Calculate the performance score for each model
def calculate_model_scores():
    model_scores = {}

    # Loop through each model and calculate the performance score
    for name in models.keys():
        # Calculate average performance metrics
        avg_r2 = np.mean(results_r2[name])
        avg_mse = np.mean(results_mse[name])
        avg_mape = np.mean(results_mape[name])
        avg_rmse = np.mean(results_rmse[name])

        # We assume that higher R² is better, while lower MSE, MAPE, and RMSE are better.
        # So, we normalize the metrics by their ranges or use inverse of their values for MSE, MAPE, and RMSE.

        mse_score = 1 / (avg_mse + 1e-6)  # Avoid division by zero
        mape_score = 1 / (avg_mape + 1e-6)
        rmse_score = 1 / (avg_rmse + 1e-6)

        # Calculate the performance score
        performance_score = avg_r2 + mse_score + mape_score + rmse_score

        # Store the score for the model
        model_scores[name] = performance_score

    return model_scores

# Now calculate the model scores and rank the models
model_scores = calculate_model_scores()

# Sort the models by their performance score (highest first)
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
best_model_name, best_model_score = sorted_models[0]
print(f"The best model is: {best_model_name} with a performance score of {best_model_score:.2f}")
print("\nModel rankings based on performance scores:")
for model_name, score in sorted_models:
    print(f"{model_name}: {score:.2f}")
model_names = [name for name, score in sorted_models]
model_scores = [score for name, score in sorted_models]
norm = mcolors.Normalize(vmin=min(model_scores), vmax=max(model_scores))
cmap = cm.get_cmap('Blues')
bar_colors = [cmap(norm(score)) for score in model_scores]
plt.figure(figsize=(10, 5))
plt.barh(model_names, model_scores, color=bar_colors, edgecolor='black')
plt.xlabel('Performance Score')
plt.ylabel('Models')
plt.title('Performance Scores of Models')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, score in enumerate(model_scores):
    plt.text(score, i, f'{score:.4f}', va='center', fontsize=10)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('Best_models_scores.pdf', format='pdf', bbox_inches='tight')
plt.show()

sorted_indices = np.argsort(Y_test_pred)
sorted_y_true = Y_test[sorted_indices]
sorted_y_pred = Y_test_pred[sorted_indices]
cumulative_true = np.cumsum(sorted_y_true)
cumulative_pred = np.cumsum(sorted_y_pred)

n_models = len(models)
n_rows = int(np.ceil(n_models / 2))
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
axes = axes.flatten()
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap = cm.get_cmap('viridis')

for i, name in enumerate(models.keys()):
    ax = axes[i]

    true_vals = np.array(all_true_values[name])
    pred_vals = np.array(all_predicted_values[name])
    percentage_error = np.abs((true_vals - pred_vals) / true_vals) * 100
    percentage_error = np.clip(percentage_error, 0, 100)
    sc = ax.scatter(true_vals, pred_vals, c=percentage_error, cmap=cmap, norm=norm, edgecolors='k', alpha=1.0, linewidths=0.2)
    ax.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], color='b', lw=0.5, linestyle='--', label='Ideal')
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 125)
    ax.set_xlabel('True SNE')
    ax.set_ylabel('Predicted SNE')
    ax.set_title(f'{name} - True vs Predicted', fontsize=12)
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Percentage Error (%)')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('True_vs_Predicted_Comparison_All_Iterations.pdf', format='pdf', bbox_inches='tight')
plt.show()


