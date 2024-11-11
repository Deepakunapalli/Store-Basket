# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactional data
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'butter'],
    ['bread', 'butter', 'cheese'],
]

# Convert transactions to a one-hot encoded DataFrame
# Create a set of all unique items
all_items = sorted(set(item for transaction in transactions for item in transaction))

# Create a DataFrame with each item as a column and each transaction as a row
transaction_df = pd.DataFrame(
    [{item: (item in transaction) for item in all_items} for transaction in transactions]
).astype(bool)  # Convert to boolean type

# Print the transaction data for verification
print("One-Hot Encoded Transaction Data:")
print(transaction_df)

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.5  # Minimum support threshold
frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

# Print the frequent itemsets for verification
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets
min_confidence = 0.7  # Minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display association rules with selected columns
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
