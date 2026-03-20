import pandas as pd


def calculate_business_impact(y_test, preds, monthly_charges):
    """
    Calculate the financial impact of the model.
    """
    # Identify correctly detected churners (true positives)
    tp_mask = (y_test == 1) & (preds == 1)

    # Monthly revenue the model identified as "at risk"
    revenue_at_risk = monthly_charges[tp_mask].sum()

    # Assume a 40% retention success rate with an offer
    retention_rate = 0.40
    potential_monthly_savings = revenue_at_risk * retention_rate
    annual_savings = potential_monthly_savings * 12

    metrics = {
        "revenue_at_risk_monthly": revenue_at_risk,
        "potential_savings_monthly": potential_monthly_savings,
        "potential_savings_annual": annual_savings,
    }

    return metrics
