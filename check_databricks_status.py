"""
Simple script to orchestrate running Databricks notebooks via REST API
"""
import os
import requests
import time
import json

# Databricks configuration - read from environment or .databrickscfg
DATABRICKS_HOST = "https://dbc-a7612980-d814.cloud.databricks.com"
DATABRICKS_TOKEN = None  # Will need to get this from the config

# Paths to notebooks in the workspace
WORKSPACE_PATH = "/Workspace/Users/adventureworksanalytics/databricks_rag_solution"

NOTEBOOKS = [
    "01_ingest_data",
    "02_generate_chunks",
    "03_generate_embeddings",
    "04_create_vector_index",
    "05_rag_retrieval_and_agent"
]

def get_databricks_token():
    """Read token from .databrickscfg file"""
    config_path = os.path.expanduser("~/.databrickscfg")
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'token' in line.lower():
                    return line.split('=')[1].strip()
    except Exception as e:
        print(f"Error reading config: {e}")
    return None

def run_notebook(notebook_name, cluster_id=None):
    """Run a notebook using Databricks Jobs API"""
    token = get_databricks_token()
    if not token:
        print("Could not get Databricks token")
        return None
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Create a one-time job run
    notebook_path = f"{WORKSPACE_PATH}/{notebook_name}"
    
    payload = {
        "run_name": f"Run {notebook_name}",
        "notebook_task": {
            "notebook_path": notebook_path
        }
    }
    
    # If we have a cluster ID, use it, otherwise use serverless
    if cluster_id:
        payload["existing_cluster_id"] = cluster_id
    else:
        # Try serverless compute
        payload["new_cluster"] = {
            "spark_version": "13.3.x-scala2.12",
            "node_type_id": "Standard_DS3_v2",
            "num_workers": 0
        }
    
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit"
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error running notebook: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return None

def check_run_status(run_id):
    """Check the status of a notebook run"""
    token = get_databricks_token()
    if not token:
        return None
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error checking status: {e}")
        return None

def main():
    print("Starting Databricks RAG Solution execution...")
    print(f"Host: {DATABRICKS_HOST}")
    print(f"Workspace path: {WORKSPACE_PATH}")
    
    for notebook in NOTEBOOKS:
        print(f"\n{'='*60}")
        print(f"Running notebook: {notebook}")
        print('='*60)
        
        result = run_notebook(notebook)
        if not result:
            print(f"Failed to start {notebook}")
            break
        
        run_id = result.get('run_id')
        print(f"Started run with ID: {run_id}")
        
        # Poll for completion
        while True:
            time.sleep(10)
            status = check_run_status(run_id)
            if not status:
                print("Failed to get status")
                break
            
            life_cycle_state = status.get('state', {}).get('life_cycle_state')
            result_state = status.get('state', {}).get('result_state')
            
            print(f"Status: {life_cycle_state}")
            
            if life_cycle_state in ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']:
                if result_state == 'SUCCESS':
                    print(f"✓ {notebook} completed successfully!")
                else:
                    print(f"✗ {notebook} failed with state: {result_state}")
                    print(f"Error: {status.get('state', {}).get('state_message', 'No error message')}")
                    return False
                break
        
    print("\n" + "="*60)
    print("All notebooks completed successfully!")
    print("="*60)
    return True

if __name__ == "__main__":
    main()
