#!/usr/bin/env bash
url="${1:?usage: smoke.sh <base_url> [expected_version]}"
expected="$2"
timeout="${SMOKE_TIMEOUT:-60}"
payload='{"Administrative":9,"Administrative_Duration":192.92,"Informational":2,"Informational_Duration":114.5,"ProductRelated":60,"ProductRelated_Duration":1394.24,"BounceRates":0.0,"ExitRates":0.003,"PageValues":63.65,"SpecialDay":0.0,"OperatingSystems":2,"Browser":2,"Region":7,"TrafficType":2,"VisitorType":"Returning_Visitor","Weekend":false}'
end=$((SECONDS + timeout))

while [ "$SECONDS" -lt "$end" ]; do
  if curl -fsS --max-time 10 "$url/ready" > /dev/null 2>&1; then
    response=$(curl -fsS --max-time 10 -X POST "$url/predict" -H "Content-Type: application/json" -d "$payload" 2> /dev/null)
    if echo "$response" | grep -q '"probability"'; then
      if [ -z "$expected" ] || echo "$response" | grep -Eq "\"model_version\": *\"$expected\""; then
        echo "$response"
        exit 0
      fi
    fi
  fi
  sleep 2
done

echo "Smoke test failed for $url (expected version: ${expected:-any})" >&2
exit 1
