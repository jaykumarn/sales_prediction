"""
Retrain the model with current sklearn version.
This fixes the pickle incompatibility issue from older sklearn versions.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load training data
train = pd.read_csv("train.csv")
df = pd.DataFrame(train)

# Drop columns with many null values (same as original notebook)
df.drop("Outlet_Size", axis=1, inplace=True)
df.drop("Item_Weight", axis=1, inplace=True)
df.drop("Item_Identifier", inplace=True, axis=1)

# Encode Item_Fat_Content
fat = {"Low Fat": 0, "Regular": 1, "low fat": 0, "LF": 0, "reg": 1}
df.Item_Fat_Content = [fat[item] for item in df.Item_Fat_Content]

# Encode Item_Type
itemtype = {
    'Baking Goods': 0, 'Breads': 1, 'Breakfast': 2, 'Canned': 3, 'Dairy': 4,
    'Frozen Foods': 5, 'Fruits and Vegetables': 6, 'Hard Drinks': 7,
    'Health and Hygiene': 12, 'Household': 11, 'Meat': 10, 'Others': 9, 'Seafood': 8,
    'Snack Foods': 13, 'Soft Drinks': 14, 'Starchy Foods': 15
}
df.Item_Type = [itemtype[item] for item in df.Item_Type]

# Drop Outlet_Identifier
df.drop("Outlet_Identifier", axis=1, inplace=True)

# Create Age_Outlet feature
df["Age_Outlet"] = 2021 - df["Outlet_Establishment_Year"]
df.drop("Outlet_Establishment_Year", axis=1, inplace=True)

# Encode Outlet_Location_Type
tier = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
df.Outlet_Location_Type = [tier[item] for item in df.Outlet_Location_Type]

# Encode Outlet_Type
market_type = {
    'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2,
    'Supermarket Type3': 3
}
df.Outlet_Type = [market_type[item] for item in df.Outlet_Type]

# Handle outliers in Item_Visibility
q3, q1 = np.percentile(df["Item_Visibility"], [75, 25])
iqr = q3 - q1
for i in range(len(df)):
    if df.loc[i, "Item_Visibility"] > 1.5 * iqr:
        df.loc[i, "Item_Visibility"] = 0.066132

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    df.drop("Item_Outlet_Sales", axis=1),
    df["Item_Outlet_Sales"],
    test_size=0.2,
    random_state=42
)

# Train GradientBoostingRegressor (good performance, pure sklearn)
model = GradientBoostingRegressor(
    learning_rate=0.01, 
    n_estimators=500, 
    max_depth=5, 
    min_samples_split=8, 
    min_samples_leaf=100
)
model.fit(x_train, y_train)

# Evaluate
from sklearn import metrics
pred = model.predict(x_test)
print(f"R2 Score: {metrics.r2_score(y_test, pred)*100:.2f}%")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, pred)):.2f}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel retrained and saved to model.pkl")
print(f"Feature order: {list(df.drop('Item_Outlet_Sales', axis=1).columns)}")
