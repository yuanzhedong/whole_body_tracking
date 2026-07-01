"""Delete older duplicate 'BONES-SEED ... Data Generation Pipeline' W&B reports.

Keeps the single newest one (reports list is newest-first). Loads each report to read its
title (the list API doesn't populate it). Run in .venv (wandb 0.27). Dry-run by default.
"""
import argparse

import wandb
from wandb_gql import gql

ENTITY, PROJECT = "toddler_tracking", "g1-sim2sim"
MATCH = "Data Generation Pipeline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()
    api = wandb.Api()
    mut = gql('mutation deleteView($id: ID!) { deleteView(input: {id: $id}) { success } }')

    matches = []
    for r in api.reports(f"{ENTITY}/{PROJECT}"):
        try:
            title = wandb.apis.reports.Report.from_url(r.url).title
        except Exception:
            title = getattr(r, "title", "") or ""
        if MATCH in title:
            matches.append(r.id)
    if not matches:
        print("no matching reports found"); return
    keep, drop = matches[0], matches[1:]          # newest-first -> keep [0]
    print(f"keep: {keep}\ndrop ({len(drop)}): {drop}")
    if not args.apply:
        print("(dry-run; pass --apply to delete)"); return
    for rid in drop:
        api.client.execute(mut, variable_values={"id": rid}); print("deleted", rid)


if __name__ == "__main__":
    main()
