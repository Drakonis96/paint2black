#!/bin/sh
set -e

# Ensure the temporary directory exists and has correct ownership
mkdir -p /app/temp
chown -R appuser:appuser /app/temp

# Execute the given command as appuser
exec /usr/sbin/runuser -u appuser -- "$@"
