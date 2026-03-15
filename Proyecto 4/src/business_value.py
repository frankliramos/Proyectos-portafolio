def calculate_roi(y_test, preds, monthly_charges):
    """
    Calculate potential savings based on the model.
    Assumes a successful retention campaign saves 50% of identified customers.
    """
    # Customers correctly identified by the model (true positives)
    tp_mask = (y_test == 1) & (preds == 1)
    revenue_at_risk = monthly_charges[tp_mask].sum()

    # If we save 50% with an offer
    potential_savings = revenue_at_risk * 0.5

    print(f"💰 Monthly Revenue at Risk Identified: ${revenue_at_risk:,.2f}")
    print(f"📈 Estimated Monthly Savings (50% retention): ${potential_savings:,.2f}")
    print(f"🚀 Projected Annual Savings: ${potential_savings * 12:,.2f}")

    return potential_savings
