"""Mock banking tools. All data is synthetic and generated locally — no backend calls."""

import random
from datetime import datetime, timedelta

from langchain_core.tools import tool

_MERCHANTS = [
    "Amazon", "Whole Foods", "Shell Gas", "Netflix", "Uber",
    "Starbucks", "Rent Payment", "Payroll Deposit", "Spotify", "Target",
]


@tool
def check_balance(account_id: str) -> str:
    """Check the current balance of a bank account. Example account_id: 'CHK-1001'."""
    rng = random.Random(account_id)
    balance = round(rng.uniform(50, 25000), 2)
    return f"Account {account_id} balance: ${balance:,.2f} USD (as of now)."


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between two bank accounts and return a synthetic confirmation."""
    rng = random.Random(f"{from_account}:{to_account}:{amount}")
    confirmation_id = f"TXN-{rng.randint(100000, 999999)}"
    return (
        f"Transferred ${amount:,.2f} from {from_account} to {to_account}. "
        f"Confirmation ID: {confirmation_id}. Status: SUCCESS (synthetic)."
    )


@tool
def get_transaction_history(account_id: str, limit: int = 5) -> str:
    """Get recent transaction history for an account, most recent first."""
    rng = random.Random(f"{account_id}:{limit}")
    today = datetime.now()
    lines = []
    for _ in range(limit):
        day = today - timedelta(days=rng.randint(0, 30))
        amount = round(rng.uniform(-500, 3000), 2)
        merchant = rng.choice(_MERCHANTS)
        lines.append(f"{day.date()} | {merchant} | ${amount:,.2f}")
    return "\n".join(lines)


TOOLS = [check_balance, transfer_funds, get_transaction_history]
