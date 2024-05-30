import pandas as pd
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

# Load the CSV files
cards_df = pd.read_csv('AA222FinalProjectProblem2.csv')
spend_df = pd.read_csv('Spend.csv')
existing_cards_df = pd.read_csv('ExistingCardSetup.csv')

# Extract Data
card_ids = cards_df['Card'].tolist()  # Cards available for selection
num_cards = len(card_ids)  # Number of cards available for selection
categories = spend_df['Categories'].tolist()  # Categories of spend
spend_amounts = spend_df['Spend'].tolist()  # Spending per category
point_values = cards_df['PointValue'].tolist()  # $ Value per point for card i (P_i)
point_values_CashValue = cards_df['PointValue_CashValue'].tolist()  # $ Cash Value per point for card i (P_i)
annual_fees = cards_df['AnnualFee'].tolist()  # Annual fee for card i (A_i)
earn_rates = cards_df[categories].values  # Points earned per dollar spend on card i for category j (E_ij)
sign_up_bonus_points = cards_df["Sign_Up_Bonus_Points"].tolist()  # Sign up bonus for card i (B_i)
sign_up_bonus_spending_requirements = cards_df["Spending_Requirements"].tolist()  # Sign up bonus spending requirement for card i (R_i)
existing_card_ids = existing_cards_df['ExistingCards'].tolist()  # Existing cards in the setup
num_existing_cards = len(existing_card_ids)

# Create a dictionary to indicate if a card is in the existing setup
existing_card_indicator = {card_id: 1 if card_id in existing_card_ids else 0 for card_id in card_ids}

# Function to optimize the model with a given constraint on the number of selected cards
def optimize_with_constraint(num_selected_cards, point_values):
    # Initialize the model
    model = Model("CreditCardOptimization")

    # Create decision variables
    x = model.addVars(num_cards, vtype=GRB.BINARY, name="x")  # x_i: if card i is chosen
    S = model.addVars(num_cards, len(categories), vtype=GRB.CONTINUOUS, name="S")  # S_ij: spend on card i in category j

    # Auxiliary variables for spending requirements
    Spending_Requirement_Met = model.addVars(num_cards, vtype=GRB.BINARY, name="Spending_Requirement_Met")

    # Objective function: Maximize the total value of points earned minus total annual fees plus sign-up bonuses
    objective = quicksum(
        x[i] * point_values[i] * quicksum(earn_rates[i][j] * S[i, j] for j in range(len(categories))) 
        - x[i] * annual_fees[i]
        + Spending_Requirement_Met[i] * point_values[i] * sign_up_bonus_points[i] * (1 - existing_card_indicator[card_ids[i]])
        for i in range(num_cards)
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # Constraint: Spend allocated to card i for spend category j cannot be negative
    for i in range(num_cards):
        for j in range(len(categories)):
            model.addConstr(S[i, j] >= 0, name=f"SpendNonNegative_{i}_{j}")

    # Constraint: The total spend on spend category j across all cards must match the user-specified spend allocation
    for j in range(len(categories)):
        model.addConstr(quicksum(S[i, j] for i in range(num_cards)) == spend_amounts[j], name=f"TotalSpend_{j}")

    # Constraint: Ensure no spend is allocated to a card that is not chosen
    for i in range(num_cards):
        for j in range(len(categories)):
            model.addConstr(S[i, j] <= spend_amounts[j] * x[i], name=f"SpendIfChosen_{i}_{j}")

    # Constraint: Limit the number of selected cards
    model.addConstr(quicksum(x[i] for i in range(num_cards)) == num_selected_cards, name="NumSelectedCards")

    # Constraint: Ensure the Spending_Requirement_Met variable reflects whether the spending requirement is met
    for i in range(num_cards):
        model.addConstr(quicksum(S[i, j] for j in range(len(categories))) >= sign_up_bonus_spending_requirements[i] * Spending_Requirement_Met[i], name=f"SignUpMetRequirement_{i}")
        model.addConstr(Spending_Requirement_Met[i] <= x[i], name=f"SignUpMetOnlyIfChosen_{i}")

    # Constraint: Ensure cards in the existing credit card setup are selected
    for i in range(num_cards):
        if existing_card_indicator[card_ids[i]] == 1:
            model.addConstr(x[i] == 1, name=f"ExistingCardSelected_{i}")

    # Optimize the model
    model.optimize()

    # Return the objective value and selected cards if the solution is optimal
    if model.status == GRB.Status.OPTIMAL:
        selected_cards = [card_ids[i] for i in range(num_cards) if x[i].x > 0.5]
        return model.objVal, selected_cards, model
    else:
        return None, None, None

# Function to identify Pareto optimal points
def identify_pareto(points):
    pareto_points = []
    for i, point in enumerate(points):
        dominated = False
        for j, other_point in enumerate(points):
            if i != j and other_point[0] <= point[0] and other_point[1] >= point[1] and (other_point[0] < point[0] or other_point[1] > point[1]):
                dominated = True
                break
        if not dominated:
            pareto_points.append(point)
    return pareto_points

# Generate results for each value of num_selected_cards for the original point values
results_original = []
pareto_frontier_original = []
for num_selected_cards in range(max(1, num_existing_cards), num_cards + 1):
    obj_val, selected_cards, model = optimize_with_constraint(num_selected_cards, point_values)
    if obj_val is not None:
        for selected_card in selected_cards:
            # Find the index of the selected card in the DataFrame
            card_index = card_ids.index(selected_card)
            # Extract spending breakdown for the selected card
            spending_breakdown = [model.getVarByName(f"S[{card_index},{j}]").x for j in range(len(categories))]
            # Create a dictionary to store the results
            result_dict = {'NumSelectedCards': num_selected_cards, 
                           'ObjectiveValue': obj_val, 
                           'SelectedCard': selected_card}
            # Add spending breakdown for each category
            for category_index, category in enumerate(categories):
                result_dict[category] = spending_breakdown[category_index]
            # Append the result to the results list
            results_original.append(result_dict)
        pareto_frontier_original.append((num_selected_cards, obj_val))

# Generate results for each value of num_selected_cards for the cash value point values
results_cash_value = []
pareto_frontier_cash_value = []
for num_selected_cards in range(max(1, num_existing_cards), num_cards + 1):
    obj_val, selected_cards, model = optimize_with_constraint(num_selected_cards, point_values_CashValue)
    if obj_val is not None:
        for selected_card in selected_cards:
            # Find the index of the selected card in the DataFrame
            card_index = card_ids.index(selected_card)
            # Extract spending breakdown for the selected card
            spending_breakdown = [model.getVarByName(f"S[{card_index},{j}]").x for j in range(len(categories))]
            # Create a dictionary to store the results
            result_dict = {'NumSelectedCards': num_selected_cards, 
                           'ObjectiveValue': obj_val, 
                           'SelectedCard': selected_card}
            # Add spending breakdown for each category
            for category_index, category in enumerate(categories):
                result_dict[category] = spending_breakdown[category_index]
            # Append the result to the results list
            results_cash_value.append(result_dict)
        pareto_frontier_cash_value.append((num_selected_cards, obj_val))

# Identify Pareto optimal points
pareto_optimal_original = identify_pareto(pareto_frontier_original)
pareto_optimal_cash_value = identify_pareto(pareto_frontier_cash_value)

# Convert results to DataFrame and save to CSV
results_original_df = pd.DataFrame(results_original)
results_original_df.to_csv('optimization_results_original.csv', index=False)

results_cash_value_df = pd.DataFrame(results_cash_value)
results_cash_value_df.to_csv('optimization_results_cash_value.csv', index=False)

# Convert pareto frontier to DataFrame and save to CSV
pareto_frontier_original_df = pd.DataFrame(pareto_frontier_original, columns=['NumSelectedCards', 'ObjectiveValue'])
pareto_frontier_original_df.to_csv('pareto_frontier_original.csv', index=False)

pareto_frontier_cash_value_df = pd.DataFrame(pareto_frontier_cash_value, columns=['NumSelectedCards', 'ObjectiveValue'])
pareto_frontier_cash_value_df.to_csv('pareto_frontier_cash_value.csv', index=False)

# Plot the Pareto frontier
selected_cards_original, obj_values_original = zip(*pareto_frontier_original)
selected_cards_cash_value, obj_values_cash_value = zip(*pareto_frontier_cash_value)

pareto_cards_original, pareto_obj_values_original = zip(*pareto_optimal_original)
pareto_cards_cash_value, pareto_obj_values_cash_value = zip(*pareto_optimal_cash_value)

plt.figure(figsize=(10, 6))
plt.plot(selected_cards_original, obj_values_original, marker='x', linestyle='None', label='Design Points (ThePointsGuy Valuation)', color='red')
plt.plot(selected_cards_cash_value, obj_values_cash_value, marker='x', linestyle='None', label='Design Points (Cash Valuation)', color='blue')

# Highlight Pareto optimal points
plt.plot(pareto_cards_original, pareto_obj_values_original, marker='None', linestyle='-', linewidth=10, alpha=0.2, label='Pareto Frontier (ThePointsGuy Valuation)', color='red')
plt.plot(pareto_cards_cash_value, pareto_obj_values_cash_value, marker='None', linestyle='-', linewidth=10, alpha=0.2, label='Pareto Frontier (Cash Valuation)', color='blue')

# Plot the utopia point for original
utopia_point_original = (1, max(obj_values_original))
plt.scatter(*utopia_point_original, color='magenta', zorder=5, label='Utopia Point (ThePointsGuy Valuation)')

# Plot the utopia point for cash value
utopia_point_cash_value = (1, max(obj_values_cash_value))
plt.scatter(*utopia_point_cash_value, color='cyan', zorder=5, label='Utopia Point (Cash Valuation)')

plt.title('Plot of Annual Net Positive Value Gained vs Number of Cards in Setup')
plt.xlabel('Number of Selected Cards')
plt.ylabel('Annual Net Positive Value Gained ($)')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('pareto_frontier_comparison_plot.png')

# Show the plot
plt.show()
