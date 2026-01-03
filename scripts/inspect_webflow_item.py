from career_compass_content_engine.agents.publishing_agent import WebflowCMS
from career_compass_content_engine.config import Settings

s = Settings.from_env()
if not s.webflow_api_token or not s.webflow_collection_id:
    raise SystemExit("WEBFLOW_API_TOKEN and WEBFLOW_COLLECTION_ID must be set in environment")

cms = WebflowCMS(s.webflow_api_token)
item_id = "6850224f5cb4c81d89373e19"

# Try the configured collection first, then iterate site collections
def try_fetch(collection_id):
    url = f"https://api.webflow.com/collections/{collection_id}/items/{item_id}"
    try:
        resp = cms._request_with_retries('GET', url)
        return resp
    except Exception as exc:
        return None

print('Trying configured collection:', s.webflow_collection_id)
resp = try_fetch(s.webflow_collection_id)
if resp is None or resp.status_code != 200:
    # fetch all collections for the site
    cols_url = f"https://api.webflow.com/sites/{s.webflow_site_id}/collections"
    cols_resp = cms._request_with_retries('GET', cols_url)
    cols = cols_resp.json()
    found = False
    # cols may be list of dicts or list of strings
    for c in cols:
        if isinstance(c, dict):
            cid = c.get('id')
        else:
            cid = c
        if not cid:
            continue
        r = try_fetch(cid)
        if r and r.status_code == 200:
            print('Found in collection:', cid)
            print(r.text)
            found = True
            break
    if not found:
        print('Item not found in any collections on the site')
else:
    print('Found in configured collection:')
    print(resp.text)

# If specific item not found, inspect latest item in the configured collection as a fallback
print('\nFetching latest item from configured collection as fallback...')
cols_items_url = f"https://api.webflow.com/collections/{s.webflow_collection_id}/items?limit=1"
ri = cms._request_with_retries('GET', cols_items_url)
if ri.status_code == 200:
    items = ri.json().get('items') or ri.json().get('data') or ri.json()
    if items and len(items) > 0:
        latest = items[0]
        print('Latest item id:', latest.get('id'))
        fd = latest.get('fieldData') or latest.get('fields') or latest
        print('\nField data:')
        print(json.dumps(fd, indent=2))
        # search for the provided short summary string
        target = "Finding a mentor can change your career—but how do you actually build a real relationship? This guide breaks down how to find, connect with, and keep a mentor who gets you."
        matches = []
        for k, v in (fd.items() if isinstance(fd, dict) else []):
            if isinstance(v, str) and target in v:
                matches.append(k)
        if matches:
            print('\nFound target string in fields:', matches)
        else:
            print('\nTarget string not found in latest item field values.')
    else:
        print('No items found in configured collection')
else:
    print('Failed to fetch collection items:', ri.status_code, ri.text)
