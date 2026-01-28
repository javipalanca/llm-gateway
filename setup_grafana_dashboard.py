#!/usr/bin/env python3
"""
Script to import and publish LLM Gateway dashboard in Grafana.
Usage: python3 setup_grafana_dashboard.py [GRAFANA_URL] [ADMIN_USER] [ADMIN_PASSWORD]
"""

import json
import sys
import requests
from pathlib import Path

def setup_grafana(grafana_url="http://localhost:3000", admin_user="admin", admin_password=None):
    """Setup Grafana dashboard and make it public"""
    
    if not admin_password:
        print("❌ Error: GRAFANA_ADMIN_PASSWORD not provided")
        print("Usage: python3 setup_grafana_dashboard.py [GRAFANA_URL] [ADMIN_USER] [ADMIN_PASSWORD]")
        print("Or set GRAFANA_ADMIN_PASSWORD environment variable")
        return False
    
    auth = (admin_user, admin_password)
    dashboard_file = Path("./grafana-dashboard.json")
    
    print("🔧 Setting up Grafana dashboard...")
    print(f"   URL: {grafana_url}")
    print()
    
    # Check health
    print("🔍 Checking Grafana connection...")
    try:
        health = requests.get(f"{grafana_url}/api/health", auth=auth, timeout=5).json()
        if health.get("database") == "ok":
            print(f"✓ Grafana v{health.get('version')} is healthy")
        else:
            print(f"⚠️  Grafana may not be ready: {health}")
    except Exception as e:
        print(f"❌ Cannot connect to Grafana: {e}")
        return False
    print()
    
    # Create/update datasource
    print("📦 Creating Prometheus datasource...")
    ds_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
        "basicAuth": False
    }
    try:
        ds_resp = requests.post(
            f"{grafana_url}/api/datasources",
            json=ds_payload,
            auth=auth,
            timeout=5
        )
        if ds_resp.status_code in [200, 409]:  # 409 = already exists
            ds_data = ds_resp.json()
            print(f"✓ Prometheus datasource ready (ID: {ds_data.get('id', 'existing')})")
        else:
            print(f"⚠️  Datasource response: {ds_resp.text}")
    except Exception as e:
        print(f"⚠️  Datasource error: {e}")
    print()
    
    # Load and import dashboard
    print("📥 Importing dashboard...")
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        with open(dashboard_file) as f:
            dashboard = json.load(f)
        
        # Wrap dashboard for API
        import_payload = {
            "dashboard": dashboard,
            "overwrite": True,
            "message": "Auto-imported via setup script"
        }
        
        import_resp = requests.post(
            f"{grafana_url}/api/dashboards/db",
            json=import_payload,
            auth=auth,
            timeout=10
        )
        
        if import_resp.status_code == 200:
            import_data = import_resp.json()
            dash_id = import_data.get("id")
            dash_uid = import_data.get("uid")
            print(f"✅ Dashboard imported!")
            print(f"   ID: {dash_id}")
            print(f"   UID: {dash_uid}")
            print()
        else:
            print(f"❌ Import failed: {import_resp.status_code}")
            print(f"   Response: {import_resp.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error importing dashboard: {e}")
        return False
    
    # Get dashboard UID if not from import response
    if not dash_uid:
        print("🔍 Finding dashboard UID...")
        try:
            list_resp = requests.get(
                f"{grafana_url}/api/dashboards/search?query=llm-gateway",
                auth=auth,
                timeout=5
            )
            dashboards = list_resp.json()
            if dashboards:
                dash_uid = dashboards[0].get("uid")
                dash_id = dashboards[0].get("id")
                print(f"✓ Found dashboard UID: {dash_uid}")
            else:
                print("❌ Dashboard not found in search")
                return False
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    print()
    
    # Make dashboard public
    print("🔓 Making dashboard public...")
    try:
        public_resp = requests.post(
            f"{grafana_url}/api/dashboards/uid/{dash_uid}/public-dashboards",
            json={"isEnabled": True},
            auth=auth,
            timeout=5
        )
        
        if public_resp.status_code == 200:
            public_data = public_resp.json()
            public_uid = public_data.get("uid")
            print(f"✅ Dashboard is now public!")
            print(f"   Public UID: {public_uid}")
            print()
            print(f"📱 Public dashboard URL:")
            print(f"   {grafana_url}/public/dashboards/{public_uid}")
            print()
            print(f"📌 Share options:")
            print(f"   1. Direct URL (no login required)")
            print(f"   2. Embed in webpage:")
            print(f"      <iframe src=\"{grafana_url}/public/dashboards/{public_uid}?refresh=10s\"")
            print(f"              width=\"100%\" height=\"600\"></iframe>")
            print(f"   3. Kiosk mode (TV display):")
            print(f"      {grafana_url}/public/dashboards/{public_uid}?kiosk=tv&refresh=10s")
            return True
        else:
            print(f"❌ Public sharing failed: {public_resp.status_code}")
            print(f"   Response: {public_resp.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error making dashboard public: {e}")
        return False

if __name__ == "__main__":
    import os
    grafana_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    admin_user = sys.argv[2] if len(sys.argv) > 2 else "admin"
    admin_password = sys.argv[3] if len(sys.argv) > 3 else os.getenv("GRAFANA_ADMIN_PASSWORD")
    
    success = setup_grafana(grafana_url, admin_user, admin_password)
    sys.exit(0 if success else 1)
