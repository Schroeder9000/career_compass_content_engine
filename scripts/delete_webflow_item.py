#!/usr/bin/env python3
"""Delete a Webflow CMS item by id.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/delete_webflow_item.py <item_id>
If no arg provided, the script will look for `WEBFLOW_ITEM_ID` in `../.env`.
"""
import sys
from dotenv import load_dotenv
load_dotenv("../.env")
import os
import requests

def main():
    item_id = sys.argv[1] if len(sys.argv) > 1 else os.getenv("WEBFLOW_ITEM_ID")
    if not item_id:
        print("Provide item id as arg or set WEBFLOW_ITEM_ID in env")
        return
    token = os.getenv("WEBFLOW_API_TOKEN")
    if not token:
        print("WEBFLOW_API_TOKEN missing in env")
        return
    headers = {"Authorization": f"Bearer {token}", "accept-version": "1.0.0"}
    # We need the collection id to build delete url; Webflow delete endpoint is /collections/:collection_id/items/:item_id
    coll = os.getenv("WEBFLOW_COLLECTION_ID")
    if not coll:
        print("WEBFLOW_COLLECTION_ID missing in env; cannot delete")
        return
    url = f"https://api.webflow.com/v2/collections/{coll}/items/{item_id}"
    r = requests.delete(url, headers=headers, timeout=10)
    if r.status_code in (200,204):
        print("Deleted item", item_id)
    else:
        print("Delete failed", r.status_code, r.text)

if __name__ == '__main__':
    main()
