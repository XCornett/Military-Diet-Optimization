import pandas as pd
import pulp
from datetime import datetime

# Loading in data
df = pd.read_excel("insert file location here").fillna(0)

# Droping the last two rows (they contain min/max nutrient constraints, not food items)
food_df = df.iloc[:-2].copy()

# Extracting the last two rows for min/max nutrient requirements
min_req = df.iloc[-2, 3:].to_dict()  # Minimum daily intake
max_req = df.iloc[-1, 3:].to_dict()  # Maximum daily intake

# Extracting food names and costs
foods = list(food_df["Foods"])
cost = dict(zip(food_df["Foods"], food_df["Price/ Serving"]))

# Getting the list of nutrients 
nutrients = df.columns[3:]

# Storing nutrient content for each food in dictionary
nutrient_data = {n: dict(zip(food_df["Foods"], food_df[n])) for n in nutrients}

# Initializeing the optimization problem
problem = pulp.LpProblem("Cheapest_Diet", pulp.LpMinimize)

# Defining decision variables (how much of each food to include in the diet)
x = pulp.LpVariable.dicts("x", foods, cat="Continuous", lowBound=0)

# Defining objective function
problem += pulp.lpSum(cost[i] * x[i] for i in foods), "Total_Cost"

# Adding nutrient constraints
for nutrient in nutrients:
    problem += pulp.lpSum(nutrient_data[nutrient][i] * x[i] for i in foods) >= min_req[nutrient], f"Min_{nutrient}"
    problem += pulp.lpSum(nutrient_data[nutrient][i] * x[i] for i in foods) <= max_req[nutrient], f"Max_{nutrient}"

# Solving the optimization problem
problem.solve()

# Storing the results
result = {"status": pulp.LpStatus[problem.status]}

if result["status"] == "Optimal":
    result["total_cost"] = round(pulp.value(problem.objective), 2)
    result["servings"] = {i: round(x[i].varValue, 2) for i in foods if x[i].varValue and x[i].varValue > 0}

else:
    result["message"] = "No optimal solution found."

# Saving the model to a file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
lp_filename = f"military_meal_model_{timestamp}.lp"
problem.writeLP(lp_filename)

# Printing the results
print(f"Status: {result['status']}")
if "total_cost" in result:
    print(f"Total cost: ${result['total_cost']}")
    print("Optimal servings:")
    for food, amount in result["servings"].items():
        print(f"  {food}: {amount} servings")
else:
    print(result["message"])


# Defining second model with additional constraints to make meal "more balanced"
problem2 = pulp.LpProblem("Diet Optimization with Extra Constraints", pulp.LpMinimize)
y = pulp.LpVariable.dicts("y", foods, cat="Binary")  # Whether a food is chosen
z = pulp.LpVariable.dicts("z", foods, lowBound=0, cat="Continuous")  # How much is eaten

problem2 += pulp.lpSum(cost[i] * z[i] for i in foods), "Total_Cost"

for nutrient in nutrients:
    problem2 += pulp.lpSum(nutrient_data[nutrient][i] * z[i] for i in foods) >= min_req[nutrient], f"Min_{nutrient}"
    problem2 += pulp.lpSum(nutrient_data[nutrient][i] * z[i] for i in foods) <= max_req[nutrient], f"Max_{nutrient}"

# Adding protein, veggie, and serving constraints
for food in foods:
    problem2 += z[food] >= 0.1 * y[food], f"Min_Serving_{food}"  # If selected, at least 1/10 serving
    problem2 += z[food] <= y[food] * 100, f"Max_Serving_{food}"  # Ensure logical bounds

problem2 += y["Celery, Raw"] + y["Frozen Broccoli"] <= 1, "Celery_Broccoli_Constraint"

protein_sources = [
    "Roasted Chicken", "Poached Eggs", "Scrambled Eggs", "Bologna,Turkey",
    "Frankfurter, Beef", "Ham,Sliced,Extralean", "Kielbasa,Prk", "Hamburger W/Toppings",
    "Hotdog, Plain", "Pork", "Sardines in Oil", "White Tuna in Water",
    "Neweng Clamchwd", "New E Clamchwd,W/Mlk", "Beanbacn Soup,W/Watr",
    "Chicknoodl Soup", "Splt Pea&Hamsoup", "Vegetbeef Soup"
]
problem2 += pulp.lpSum(y[p] for p in protein_sources) >= 3, "Protein_Variety_Constraint"

# Solving and printing the second model
problem2.solve()
print("Enhanced Model Result:")
for food in foods:
    if z[food].varValue > 0:
        print(f"{food}: {z[food].varValue:.2f} servings")
print(f"Total Cost: ${pulp.value(problem2.objective):.2f}")