import urllib.request
import urllib.parse
import json

# API endpoint
serviceurl = 'http://py4e-data.dr-chuck.net/opengeo?'

# Prompt for the location
address = input('Enter location: ')

# Encode the address parameter
params = {'q': address}
url = serviceurl + urllib.parse.urlencode(params)

print('Retrieving', url)
# Open the URL and read the data
with urllib.request.urlopen(url) as response:
    data = response.read().decode()
    print('Retrieved', len(data), 'characters')

# Parse the JSON data
try:
    js = json.loads(data)
except:
    js = None

# Extract the first plus_code
if not js or 'features' not in js or len(js['features']) == 0:
    print('No plus_code found')
else:
    plus_code = js['features'][0]['properties']['plus_code']
    print('Plus code', plus_code)