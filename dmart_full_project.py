##MARKET BASKET ANALYSIS
# Install necessary libraries
!pip install mlxtend xlsxwriter openpyxl

# Import required libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset (Update the correct file path)
file_path = "/content/DMart(new).xlsx"
df = pd.read_excel(file_path)

# Convert 'Items_list_new_1' column into transactions
df['Items_list_new_1'] = df['Items_list_new_1'].astype(str)
transactions = [items.split(', ') for items in df['Items_list_new_1'].dropna()]

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm with support and confidence
min_support = 0.1 # Minimum support threshold
min_confidence = 0.7 # Minimum confidence threshold
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules with confidence
rules = association_rules(frequent_itemsets, metric="confidence",
                          min_threshold=min_confidence)

# Display all rules with confidence applied
print("\n Association Rules (Support ≥ {:.2f}, Confidence ≥ {:.2f}):".format(min_support,
                                                                            min_confidence))
print(rules)

# Save the rules to an Excel file properly
output_file = "market_basket_rules_confidence.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    rules.to_excel(writer, index=False)
print("\n Rules saved to:", output_file)

# Download the file (For Google Colab users)
from google.colab import files
files.download(output_file)

##Stacked bar chart
import matplotlib.pyplot as plt
import seaborn as sns

# Convert frozensets to strings for better readability
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))

# Select top consequents based on highest lift
top_consequents = rules.groupby("consequents")["lift"].sum().nlargest(10).index
filtered_rules = rules[rules["consequents"].isin(top_consequents)]

# Pivot table for stacked bar chart
pivot_data = filtered_rules.pivot(index='antecedents', columns='consequents',
                                  values='lift').fillna(0)

# Plot
plt.figure(figsize=(12, 6))
pivot_data.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 6))

# Labels and title
plt.xlabel("Antecedents (Items Bought)")
plt.ylabel("Lift Value")
plt.title("Stacked Bar Chart of Lift vs. Consequents")
plt.xticks(rotation=45)
plt.legend(title="Consequents (Recommended Items)", bbox_to_anchor=(1.05, 1),
           loc='upper left')
plt.show()

... (rest of the 1000+ lines code continues) ...
