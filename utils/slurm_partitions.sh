#!/usr/bin/env bash
set -euo pipefail

# Try to get partitions from associations; fallback to visible partitions
get_parts() {
  if sacctmgr show assoc where user=$USER format=Partition -n -P 2>/dev/null | grep -q .; then
    sacctmgr show assoc where user=$USER format=Partition -n -P | sed 's/|//g' | sort -u
  else
    sinfo -h -o "%P" | sed 's/*$//' | sort -u
  fi
}

sum_gpus_in_gres() {
  # Input is a GRES string like: gpu:V100:4 or gpu:A100:2(S:0-1),gpu:MIG-1g.5gb:4
  # We sum the numeric counts after the last colon for every 'gpu...' item.
  awk -F',' '
    {
      total=0
      for (i=1; i<=NF; i++) {
        item=$i
        # Keep only gpu-* entries
        if (item ~ /^gpu/) {
          # Drop any parentheses suffix like (S:0-3)
          gsub(/\(.*\)/, "", item)
          # Count is the field after the last colon; if missing, assume 1
          n = split(item, a, ":")
          if (n >= 3 && a[n] ~ /^[0-9]+$/) total += a[n]
          else total += 1
        }
      }
      print total
    }'
}

for p in $(get_parts); do
  echo "=== Partition: $p ==="
  # List nodes with their GRES
  sinfo -p "$p" -N -o "%N %G" | sed '1d' | awk 'NF' | while read -r node gres; do
    printf "%-24s %s\n" "$node" "${gres:-none}"
  done

  # Total GPUs across nodes in this partition
  total=$(sinfo -p "$p" -N -o "%G" | sed '1d' | awk 'NF' | sum_gpus_in_gres | awk '{s+=$1} END{print s+0}')
  echo "Total GPUs in $p: ${total}"
  echo
done
