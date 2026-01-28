#!/bin/bash
# Script to import LLM Gateway dashboard into Grafana
# Usage: ./import_dashboard.sh [GRAFANA_URL] [ADMIN_USER] [ADMIN_PASSWORD]

set -e

GRAFANA_URL="${1:-http://localhost:3000}"
ADMIN_USER="${2:-admin}"
ADMIN_PASSWORD="${3:-${GRAFANA_ADMIN_PASSWORD}}"
DASHBOARD_FILE="./grafana-dashboard.json"

echo "📊 Importing LLM Gateway dashboard into Grafana..."
echo "   URL: $GRAFANA_URL"
echo "   File: $DASHBOARD_FILE"
echo ""

# Check if dashboard file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "❌ Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi

# Get Grafana version to ensure it's running
echo "🔍 Checking Grafana connection..."
HEALTH=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" "$GRAFANA_URL/api/health")
if echo "$HEALTH" | grep -q '"database":"ok"'; then
    echo "✓ Grafana is healthy"
else
    echo "⚠️  Grafana might not be fully initialized yet"
    echo "   Response: $HEALTH"
fi
echo ""

# Create Prometheus datasource if it doesn't exist
echo "📦 Creating/updating Prometheus datasource..."
DS_RESPONSE=$(curl -s -X POST -u "$ADMIN_USER:$ADMIN_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true,
    "basicAuth": false
  }' \
  "$GRAFANA_URL/api/datasources")

if echo "$DS_RESPONSE" | grep -q '"id"'; then
    DS_ID=$(echo "$DS_RESPONSE" | grep -o '"id":[0-9]*' | cut -d':' -f2)
    echo "✓ Prometheus datasource created/updated: ID $DS_ID"
elif echo "$DS_RESPONSE" | grep -q "already exists"; then
    echo "✓ Prometheus datasource already exists"
fi
echo ""

# Import the dashboard (wrapped in correct format)
echo "📥 Importing dashboard..."

# Create a temporary file with the wrapped dashboard
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE" << 'EOF'
{
  "dashboard": $(cat $DASHBOARD_FILE),
  "overwrite": true
}
EOF

# Actually, let's use a simpler approach - just wrap the JSON correctly
IMPORT_RESPONSE=$(curl -s -X POST -u "$ADMIN_USER:$ADMIN_PASSWORD" \
  -H "Content-Type: application/json" \
  --data-binary @- \
  "$GRAFANA_URL/api/dashboards/db" << JSONEOF
{
  "dashboard": $(cat "$DASHBOARD_FILE"),
  "overwrite": true
}
JSONEOF
)

rm -f "$TEMP_FILE"

echo "Response: $IMPORT_RESPONSE"
echo ""

if echo "$IMPORT_RESPONSE" | grep -q '"id"'; then
    DASH_ID=$(echo "$IMPORT_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
    DASH_UID=$(echo "$IMPORT_RESPONSE" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "✅ Dashboard imported successfully!"
    echo "   Dashboard ID: $DASH_ID"
    echo "   Dashboard UID: $DASH_UID"
    echo ""
    echo "📊 Dashboard URL:"
    echo "   $GRAFANA_URL/d/$DASH_UID"
elif echo "$IMPORT_RESPONSE" | grep -q "already exists"; then
    echo "✓ Dashboard already exists"
    # Try to get the UID
    EXISTING=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" \
      "$GRAFANA_URL/api/dashboards/uid/llm-gateway-metrics")
    if echo "$EXISTING" | grep -q '"uid"'; then
        DASH_UID=$(echo "$EXISTING" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
        DASH_ID=$(echo "$EXISTING" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
        echo "   Dashboard ID: $DASH_ID"
        echo "   Dashboard UID: $DASH_UID"
        echo "   URL: $GRAFANA_URL/d/$DASH_UID"
    fi
else
    echo "❌ Failed to import dashboard"
    echo "   Full response: $IMPORT_RESPONSE"
    exit 1
fi
