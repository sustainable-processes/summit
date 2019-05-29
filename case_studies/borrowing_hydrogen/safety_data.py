import requests
import json
import pubchempy as pcp

def get_cid(name):
    res = pcp.get_compounds(name, 'name')
    return res[0].record['id']['id']['cid']

def get_safety_data(name):
    cid = get_cid(name)
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/'
    compound_url = f"{base_url}{cid}/JSON/"
    payload = {'response_type': 'save',
               'response_basename': f"compound_CID_{cid}"}
    r = requests.get(compound_url, params=payload)
    res = json.loads(r.text)
    for section in res['Record']['Section']:
        if section['TOCHeading'] == "Safety and Hazards":
            hazards_section = section['Section'][0]['Section'][0]['Information']
            sds = {}
            try:
                sds['hazards'] = [hazard['String'] for hazard in hazards_section[2]['Value']['StringWithMarkup']][1:-1]
                sds['exposure_limits'] = section['Section'][6]['Section'][0]['Information'][0]['Value']['StringWithMarkup'][0]['String']
                sds['reactivity'] = section['Section'][7]['Section'][3]['Information'][0]['Value']['StringWithMarkup'][0]['String']
            except IndexError:
                pass
            break
        else:
            sds = {}
    return sds