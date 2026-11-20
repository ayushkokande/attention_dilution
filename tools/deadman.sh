#!/bin/bash
# Auto-stop guard: run ON THE POD so it stops itself at a fixed UTC time even if the controller disappears.
# Needs a RunPod API key with GraphQL write access in /root/.rp/key (chmod 600). RUNPOD_POD_ID is set by RunPod.
# Usage (detached, survives the launching shell):  STOP_AT=23:59 python launch.py deadman.sh /root/.rp/deadman.log
STOP_AT=${STOP_AT:-23:59}
target=$(date -u -d "$STOP_AT" +%s); now=$(date -u +%s); [ $target -le $now ] && target=$((target+86400))
sleep $((target-now))
curl -s -X POST https://api.runpod.io/graphql -H "Content-Type: application/json" -H "Authorization: Bearer $(cat /root/.rp/key)" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}){ id desiredStatus } }\"}" >> /root/.rp/deadman.log 2>&1
