"""
Test Flightradar24 with price data - they have some pricing
Also test EVA Air official site API
"""
from curl_cffi import requests
import re, json

# Try EVA Air direct booking API
url = 'https://www.evaair.com/en-global/book-and-plan/flights/search-flights/'
res = requests.get(url, impersonate='chrome110', timeout=10)
print('EVA status:', res.status_code)

# Try Google Flights JSON endpoint via fast-flights' approach
# Actually let's check what fast_flights gives us for flight source URL
from fast_flights import get_flights, FlightData, Passengers
try:
    result = get_flights(
        flight_data=[FlightData(date='2026-06-08', from_airport='TPE', to_airport='PUS')],
        trip='one-way',
        seat='economy',
        passengers=Passengers(adults=1, children=0, infants_in_seat=0, infants_on_lap=0),
        fetch_mode='fallback'
    )
    if result and result.flights:
        f = result.flights[0]
        print('Fast-flights result:', f.name, f.price, f.departure, f.arrival)
        print('All attributes:', [a for a in dir(f) if not a.startswith('_')])
except Exception as e:
    print('fast-flights error:', e)
