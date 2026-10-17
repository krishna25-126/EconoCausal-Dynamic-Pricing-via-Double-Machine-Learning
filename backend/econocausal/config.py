"""Shared constants for the EconoCausal pipeline."""

# Pre-treatment customer features. These are the confounders: they influence
# both who received a discount historically and whether the customer buys.
FEATURES = ["age", "income_k", "tenure_months", "past_purchases", "engagement", "cart_value"]

TREATMENT = "discount"      # dollars off, continuous (0 = no discount)
OUTCOME = "purchased"       # 1 if the customer bought, else 0
ID_COL = "user_id"
MARGIN_COL = "margin"       # profit in dollars if the customer buys (before discount)

# Discount levels (in dollars) the optimizer may assign.
DISCOUNT_LEVELS = [0, 5, 10, 15, 20]

DEFAULT_BUDGET = 5000.0
RANDOM_STATE = 42