import requests
import time


def search_location(name, 
                    username,
                    max_results=5,
                    sleep=2):
    '''
    Search for locations on Geonames.
    Params:
        name (str): name of the location to search
        username (str): your registered username on geonames
        max_results (int): the maximum number of results to return
        sleep (int): waiting time in second between two GeoNames API calls
    '''
    url = f"http://api.geonames.org/searchJSON?q={name}&maxRows={max_results}&username={username}"
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    data = response.json()    
    if data['totalResultsCount'] > 0:
        results = []
        for d in range(len(data['geonames'])):
            if name.lower() in data['geonames'][d]['name'].lower() or data['geonames'][d]['name'].lower() in name.lower():
                results.append({
                    'query':name,
                    'geonameId': data['geonames'][d]['geonameId'],
                    'name': data['geonames'][d]['name'],
                    'toponymName' : data['geonames'][d].get('toponymName'),
                    'countryName': data['geonames'][d].get('countryName'),
                    'coordinates':(float(data['geonames'][d]['lat']),float(data['geonames'][d]['lng'])),
                    'fcl':data['geonames'][d].get('fcl'),
                    'fclName':data['geonames'][d].get('fclName'),
                    'hierarchy':get_geonames_hierarchy(data['geonames'][d]['geonameId'],username),
                    'adminName1': data['geonames'][d].get('adminName1')
                })
                time.sleep(sleep)
        return results
    else:
        return []
    

def get_geonames_hierarchy(geoname_id, username):
    '''
    Return the hierarchy corresponding to a GeoName ID.
    Params:
        geoname_id (int): the unique ID for the GeoName place
        username (str): your registered username on geonames
    '''
    url = f"http://api.geonames.org/hierarchyJSON?geonameId={geoname_id}&username={username}"
    response = requests.get(url)
    data = response.json()
    hierarchy = []
    for item in data.get('geonames', []):
        hierarchy.append(item['name'])
        
    return hierarchy
