#!/usr/bin/env bash
# Start the headquarters (HQ) node. See scripts/hq_start.sh for details.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hq_start.sh" "$@"
