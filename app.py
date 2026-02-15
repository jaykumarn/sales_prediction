from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# Load model with error handling
model = None
MODEL_PATH = 'model.pkl'

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Run retrain_model.py first.")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")

# Mapping dictionaries for display
FAT_CONTENT_MAP = {0: "Low Fat", 1: "Regular"}
ITEM_TYPE_MAP = {
    0: "Baking Goods", 1: "Breads", 2: "Breakfast", 3: "Canned", 4: "Dairy",
    5: "Frozen Foods", 6: "Fruits and Vegetables", 7: "Hard Drinks",
    8: "Seafood", 9: "Others", 10: "Meat", 11: "Household",
    12: "Health and Hygiene", 13: "Snack Foods", 14: "Soft Drinks", 15: "Starchy Foods"
}
LOCATION_TYPE_MAP = {0: "Tier 1", 1: "Tier 2", 2: "Tier 3"}
OUTLET_TYPE_MAP = {0: "Grocery Store", 1: "Supermarket Type1", 2: "Supermarket Type2", 3: "Supermarket Type3"}


def get_sales_category(units_monthly):
    """Categorize sales performance based on monthly units"""
    if units_monthly < 10:
        return "Very Low", "danger"
    elif units_monthly < 30:
        return "Low", "warning"
    elif units_monthly < 60:
        return "Average", "info"
    elif units_monthly < 100:
        return "Good", "primary"
    else:
        return "Excellent", "success"


def get_sales_insights(sales, item_type, outlet_type, mrp):
    """Generate insights based on prediction"""
    insights = []
    
    category, _ = get_sales_category(sales)
    
    if category in ["Very Low", "Low"]:
        insights.append("Consider promotional offers to boost sales")
        insights.append("Review product placement and visibility")
    elif category in ["Good", "Excellent"]:
        insights.append("Strong sales potential - ensure adequate stock")
        insights.append("Consider featuring this item prominently")
    
    if outlet_type == 0:  # Grocery Store
        insights.append("Grocery stores typically have lower footfall than supermarkets")
    elif outlet_type == 3:  # Supermarket Type3
        insights.append("Supermarket Type3 outlets show highest sales potential")
    
    if mrp > 200:
        insights.append("Premium priced item - target quality-conscious customers")
    elif mrp < 50:
        insights.append("Budget item - high volume potential")
    
    return insights


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                               error_text="Model not loaded. Please run retrain_model.py first.")
    
    try:
        # Parse and validate inputs
        fat = request.form.get('Item_Fat_Content', '')
        item_type = request.form.get('Item_Type', '')
        location = request.form.get('Outlet_Location_Type', '')
        outlet_type = request.form.get('Outlet_Type', '')
        age = request.form.get('Age_Outlet', '')
        price = request.form.get('Item_MRP', '')
        
        # Check for empty fields
        if not all([fat, item_type, location, outlet_type, age, price]):
            return render_template('index.html', 
                                   error_text="Please fill in all fields")
        
        # Convert to numeric
        fat = int(fat)
        item_type = int(item_type)
        location = int(location)
        outlet_type = int(outlet_type)
        age = float(age)
        price = float(price)
        
        # Validate ranges
        if age < 0 or age > 100:
            return render_template('index.html', 
                                   error_text="Outlet age must be between 0 and 100 years")
        if price < 0:
            return render_template('index.html', 
                                   error_text="Price cannot be negative")
        
        # Fixed visibility value (median from training data)
        visibility = 0.0539
        
        # Feature order: Fat, Visibility, Item_Type, MRP, Location, Outlet_Type, Age
        features = [[fat, visibility, item_type, price, location, outlet_type, age]]
        prediction = model.predict(features)
        sales = max(0, prediction[0])  # Sales cannot be negative
        
        # Calculate units sold (predicted sales amount / item price)
        monthly_units = sales / price if price > 0 else 0
        daily_units = monthly_units / 30
        weekly_units = monthly_units / 4
        yearly_units = monthly_units * 12
        
        # Calculate revenue
        daily_revenue = daily_units * price
        weekly_revenue = weekly_units * price
        monthly_revenue = sales  # This is what model predicts
        yearly_revenue = yearly_units * price
        
        # Get category and insights
        category, badge_class = get_sales_category(monthly_units)
        insights = get_sales_insights(sales, item_type, outlet_type, price)
        
        # Build detailed result
        result = {
            'item_price': round(price, 2),
            # Units sold
            'daily_units': round(daily_units, 1),
            'weekly_units': round(weekly_units, 1),
            'monthly_units': round(monthly_units, 1),
            'yearly_units': round(yearly_units, 0),
            # Revenue
            'daily_revenue': round(daily_revenue, 2),
            'weekly_revenue': round(weekly_revenue, 2),
            'monthly_revenue': round(monthly_revenue, 2),
            'yearly_revenue': round(yearly_revenue, 2),
            'category': category,
            'badge_class': badge_class,
            'insights': insights,
            'input_details': {
                'fat_content': FAT_CONTENT_MAP.get(fat, "Unknown"),
                'item_type': ITEM_TYPE_MAP.get(item_type, "Unknown"),
                'location_type': LOCATION_TYPE_MAP.get(location, "Unknown"),
                'outlet_type': OUTLET_TYPE_MAP.get(outlet_type, "Unknown"),
                'outlet_age': int(age),
                'item_mrp': round(price, 2)
            }
        }
        
        return render_template('index.html', result=result)
        
    except ValueError as e:
        return render_template('index.html', 
                               error_text=f"Invalid input: Please enter valid numbers")
    except Exception as e:
        return render_template('index.html', 
                               error_text=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
