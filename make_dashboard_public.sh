#!/bin/bash
# Script to make the LLM Gateway dashboard public and read-only in Grafana
# Usage: ./make_dashboard_public.sh [GRAFANA_URL] [ADMIN_USER] [ADMIN_PASSWORD]

set -e

GRAFANA_URL="${1:-http://localhost:3000}"
ADMIN_USER="${2:-admin}"
ADMIN_PASSWORD="${3:-${GRAFANA_ADMIN_PASSWORD}}"
DASHBOARD_UID="llm-gateway-metrics"

echo "🔧 Configuring public dashboard in Grafana..."
echo "   URL: $GRAFANA_URL"
echo "   Dashboard UID: $DASHBOARD_UID"
echo ""

# Get dashboard info
echo "📊 Fetching dashboard..."
DASHBOARD=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" \
  "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID")

# If not found by UID, search by title
if echo "$DASHBOARD" | grep -q 'Dashboard not found'; then
    SEARCH=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" \
      "$GRAFANA_URL/api/search?query=LLM%20Gateway%20Metrics")
    FOUND_UID=$(echo "$SEARCH" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -n "$FOUND_UID" ]; then
        DASHBOARD_UID="$FOUND_UID"
        DASHBOARD=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" \
          "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID")
    fi
fi

if [ -z "$DASHBOARD" ] || echo "$DASHBOARD" | grep -q 'Dashboard not found'; then
    echo "❌ Dashboard not found. Make sure it exists in Grafana (try running setup_grafana_dashboard.py)."
    exit 1
fi

DASHBOARD_ID=$(echo "$DASHBOARD" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
echo "✓ Found dashboard ID: $DASHBOARD_ID (UID: $DASHBOARD_UID)"
echo ""

# Enable public sharing for the dashboard
echo "🔓 Making dashboard public..."
PUBLIC_RESPONSE=$(curl -s -X POST -u "$ADMIN_USER:$ADMIN_PASSWORD" \
  -H "Content-Type: application/json" \
  "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID/public-dashboards" \
  -d '{"isEnabled": true}')

PUBLIC_UID=$(echo "$PUBLIC_RESPONSE" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$PUBLIC_UID" ]; then
    echo "⚠️  Could not create public dashboard. It may already exist."
    echo "   Response: $PUBLIC_RESPONSE"
    # Try to get existing public dashboard
    EXISTING=$(curl -s -u "$ADMIN_USER:$ADMIN_PASSWORD" \
      "$GRAFANA_URL/api/dashboards/uid/$DASHBOARD_UID/public-dashboards")
    PUBLIC_UID=$(echo "$EXISTING" | grep -o '"uid":"[^"]*"' | head -1 | cut -d'"' -f4)
else
    echo "✓ Public dashboard created"
fi

if [ -n "$PUBLIC_UID" ]; then
    echo ""
    echo "✅ Dashboard is now public!"
    echo ""
    echo "📱 Public dashboard URL:"
    echo "   $GRAFANA_URL/public-dashboards/$PUBLIC_UID"
    echo ""
    echo "📌 You can:"
    echo "   1. Share this URL with others"
    echo "   2. Embed it in a webpage using:"
    echo "      <iframe src=\"$GRAFANA_URL/public-dashboards/$PUBLIC_UID?refresh=10s\" "
    echo "              width=\"100%\" height=\"600\"></iframe>"
    echo "   3. Access it without login required"
    echo ""
    echo "🔒 The dashboard is READ-ONLY - users cannot modify it"
else
    echo "❌ Failed to create public dashboard"
    exit 1
fi
