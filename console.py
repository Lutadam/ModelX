"""Terminal output: section headers, progress lines and the results tables.

Only plain ASCII characters, so it prints the same in any Windows terminal.
"""
import time

import pandas as pd

LINE_WIDTH = 100


def print_title(title: str) -> None:
    print("=" * LINE_WIDTH)
    print(f" {title}")
    print("=" * LINE_WIDTH)


def print_step(step_number: int, step_count: int, title: str) -> None:
    print(f"\n[{step_number}/{step_count}] {title}")
    print("-" * LINE_WIDTH)


def print_values(values: dict) -> None:
    """One aligned 'name : value' line per entry."""
    name_width = max(len(name) for name in values)
    for name, value in values.items():
        print(f"  {name:<{name_width}} : {value}")


def print_table(table: pd.DataFrame, title: str = None) -> None:
    if title:
        print(f"\n  {title}")
    for line in table.to_string().splitlines():
        print(f"  {line}")


def run_with_progress(method_name: str, judge):
    """Run one method on one fold, then print its time and outcome on one line.

    Args:
        method_name: Shown at the start of the line.
        judge: Function with no arguments that returns one results row.

    Returns:
        The results row.
    """
    print(f"    {method_name:<32}", end="", flush=True)
    start = time.perf_counter()
    result = judge()
    outcome = "CAUGHT" if result["caught"] else "missed"
    budget_note = "" if result["within_budget"] else "  (over budget)"
    print(f"{time.perf_counter() - start:7.1f} s   {outcome:<6}  {result['false_alerts']:>3} false alerts{budget_note}")
    return result


# ============================================================ results tables

def overview_table(summary: pd.DataFrame, budget_per_operating_hour: float) -> pd.DataFrame:
    """One row per method, all folds pooled."""
    rate = summary["false alerts per 100 operating hours"].astype(float)
    return pd.DataFrame({
        "caught": summary["failures caught"],
        "caught (known onset)": summary["caught, known onset"],
        "false alerts": summary["false alerts"],
        "per 100 op. h": rate.round(2),
        "within budget": (rate <= 100 * budget_per_operating_hour).map({True: "yes", False: "NO"}),
        "P(luck)": summary["P(this many catches by luck)"],
        "F4 (time-ordered)": summary["time-ordered fold"],
    })


def by_fold(per_fold: pd.DataFrame, describe_row) -> pd.DataFrame:
    """Method x failure table, one text cell per fold, methods in run order."""
    cells = per_fold.assign(cell=per_fold.apply(describe_row, axis=1))
    table = cells.pivot(index="method", columns="failure", values="cell")
    table = table.reindex(per_fold["method"].unique())
    table.index.name, table.columns.name = None, None
    return table


def describe_outcome(row: pd.Series) -> str:
    if row["caught"]:
        warning = f" {row['warning_hours']:.1f} h" if pd.notna(row["warning_hours"]) else ""
        status = f"CAUGHT{warning}"
    else:
        status = "missed"
    over = " OVER" if not row["within_budget"] else ""
    return f"{status}, {row['false_alerts']} FA{over}"


def pr_auc_table(per_fold: pd.DataFrame) -> pd.DataFrame:
    """PR-AUC per method and fold, with the base rate (what a random score gets) as the last row."""
    table = by_fold(per_fold, lambda row: "-" if pd.isna(row["pr_auc"]) else f"{row['pr_auc']:.4f}")
    base_rate = per_fold.drop_duplicates("failure").set_index("failure")["risk_base_rate"]
    table.loc["random score (base rate)"] = base_rate.map("{:.4f}".format)
    return table


def print_results(per_fold: pd.DataFrame, summary: pd.DataFrame, budget_per_operating_hour: float) -> None:
    print()
    print_title("RESULTS")
    print(f"  Budget: {100 * budget_per_operating_hour:g} false alert per 100 operating hours. "
          "FA = false alerts. OVER = above the budget in that fold.")
    print_table(overview_table(summary, budget_per_operating_hour), "Overview (all four folds pooled)")
    print_table(by_fold(per_fold, describe_outcome), "Each fold: caught or missed, false alerts")
    print_table(pr_auc_table(per_fold), "Each fold: PR-AUC (higher than the base rate = better than random)")
