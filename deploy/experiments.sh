#!/bin/bash
# MODE → BRANCH mapping. Single source of truth for which git branch each
# experiment mode corresponds to.
#
# Sourced by deploy/init.sh BEFORE git operations (to know which branch to
# check out). Each branch holds its own deploy/config.sh with the values it
# wants to run with (including EXTRA_VLLM_FLAGS). This file does NOT carry
# any runtime config — that's a property of the branch, not of the mode.
#
# To repoint a mode at a different branch:
#   1. Edit the case entry below
#   2. Commit and push to every branch we might init.sh from
#      (init.sh reads experiments.sh from whatever's currently checked out)

case "$MODE" in
    control)
        export BRANCH=control
        ;;
    treatment)
        # Currently pointed at dev-treatment while active dev happens there.
        # Flip to BRANCH=treatment once dev-treatment is merged in.
        export BRANCH=dev-treatment
        ;;
    *)
        echo "ERROR: unknown MODE='$MODE' (no entry in deploy/experiments.sh)" >&2
        echo "Valid modes: control, treatment" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
