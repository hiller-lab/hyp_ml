from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Classification of signal enhancement based on absolute value of SNE
def classify_sne(actual_value):
    if actual_value < 40:
        return "Small"
    elif 40 <= actual_value < 90:
        return "Medium"
    else:
        return "High"

# Logistic Regression Model
model = LogisticRegression(random_state=42, class_weight='balanced')

# Load the data
file_path = './Input_file_all_data.xlsx'
sheet_name = 'Data'
data = pd.read_excel(file_path, sheet_name=sheet_name)
features = ['Molecule', 'Atom', 'aiso (MHz)', 'delta g', 'Nucleophilicity index', 'Adiabatic Ionization Potential (eV)',
            'Negative Fukui Index', 'logP', 'LUMO-HOMO (hartree)', 'Geminate polarization probability Q']
selected_features = features[2:]
target_variable = 'Absolute SNE'

# Define molecules set
set_molecules = ['2M5F', 'I', '4F', '5F', '6F', '7F', '4M', '5M', '6M', '7M', '4OH', '5OH', '6OH', '7OH', '4COOH',
                 '5COOH', '6COOH', '7COOH',
                 '4N', '5N', '6N', '7N', '4OM', '5OM', '6OM', '7OM', 'Y', '4FOH', '34OHF', '3NY', '3MOH', '3OHCOOH',
                 '4MOH', '4OMOH', '2MOH','W', '3FY', '4OHCOOH', '2M6F', 'P']

# Number of iterations
iterations = 1000000

# Initialize cumulative confusion matrix
cumulative_cm = np.zeros((3, 3))

# Dictionary to store molecule names per confusion matrix cell
molecule_confusion_dict = defaultdict(list)

# Labels for confusion matrix
labels = ["Small", "Medium", "High"]

# List to store feature importances
feature_importances = np.zeros(len(selected_features))

# Loop over iterations
for iteration in range(iterations):
    # Shuffle the molecule list for each iteration to get different splits
    random.shuffle(set_molecules)
    # Split the shuffled molecules
    test_molecules = set_molecules[:5]  # First molecules are for testing
    train_molecules = set_molecules[5:]  # The rest are for training
    train_data = data[data['Molecule'].isin(train_molecules)]
    test_data = data[data['Molecule'].isin(test_molecules)]
    X_train = train_data[selected_features]
    X_test = test_data[selected_features]
    Y_train = train_data[target_variable].apply(classify_sne).values
    Y_test = test_data[target_variable].apply(classify_sne).values
    Y_test_actual = test_data[target_variable].values

    # Check if there are three classes in Y_train
    if len(set(Y_train)) < 3:
        print(f"Skipping iteration {iteration} because the training data only has one class: {set(Y_train)}")
        continue  # Skip this iteration and move to the next one

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Train logistic regression model
    model.fit(X_train_scaled, Y_train)

    # Update feature importance by adding the absolute value of coefficients
    feature_importances += np.abs(model.coef_[0])

    # Predict on test data
    Y_test_pred = model.predict(X_test_scaled)

    # Print out training set details with target variable
    print(f"Iteration {iteration + 1} - Training Set Features and Target Variable:")
    print(train_data[['Molecule'] + features + [target_variable]])

    # Print out the full test set
    print(f"Iteration {iteration + 1} - Full Test Set:")
    print(test_data[['Molecule'] + features + [target_variable]])

    # Compute the confusion matrix for this iteration
    cm = confusion_matrix(Y_test, Y_test_pred, labels=labels)

    # Accumulate confusion matrices
    cumulative_cm += cm

    # Store molecule names in confusion matrix structure
    for mol, atom, sne, true, pred in zip(test_data['Molecule'], test_data['Atom'], Y_test_actual, Y_test, Y_test_pred):
        key = (true, pred)
        molecule_confusion_dict[key].append(mol)

# Normalize the cumulative confusion matrix (for color scale)
cumulative_cm_normalized = cumulative_cm.astype('float') / cumulative_cm.sum(axis=1)[:, np.newaxis]

# Plot cumulative confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cumulative_cm_normalized, display_labels=labels)
disp.plot(cmap="Blues", values_format=".2f", ax=ax)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.title('Confusion Matrix')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('Logistic_regression_confusion_matrix.pdf', format='pdf', bbox_inches='tight')
plt.show()

molecule_matrix = np.empty((3, 3), dtype=object)
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        mol_list = molecule_confusion_dict.get((true_label, pred_label), [])
        mol_counts = {mol: mol_list.count(mol) for mol in set(mol_list)}
        molecule_matrix[i, j] = ", ".join(
            [f"{mol} ({count})" if count > 1 else mol for mol, count in mol_counts.items()])


# Plot feature importance
feature_importances /= iterations
importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Normalized Feature Importance')
plt.title('Feature Importance from Logistic Regression')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('Logistic_reg_loadings_test.pdf', format='pdf', bbox_inches='tight')
plt.show()
