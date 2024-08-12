import pandas as pd  # Make sure pandas is imported

# Step 1: Create a simple dataframe with 'Amount' and 'Time' columns
test_data = pd.DataFrame({
    'Amount': [100.0, 250.0, 15.0, 75.0],
    'Time': [50000, 60000, 70000, 80000]
})

# Step 2: Save this dataframe as a CSV file
test_data.to_csv('new_transactions.csv', index=False)
print("Sample new_transactions.csv file created.")
import pandas as pd  # Ensure pandas is imported

# Create a simple dataframe with 'Amount' and 'Time' columns
test_data = pd.DataFrame({
    'Amount': [100.0, 250.0, 15.0, 75.0],
    'Time': [50000, 60000, 70000, 80000]
})

# Save this dataframe as a CSV file
test_data.to_csv('new_transactions.csv', index=False)
print("Sample new_transactions.csv file created.")
