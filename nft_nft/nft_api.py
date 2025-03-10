import requests
from nft_config import MAGICEDENAPIKEY_SOLANA, DEXTOOLS_API_KEY
import json

class SolanaAPI:
    BASE_URL = "https://api-mainnet.magiceden.dev/v2/collections"

    def __init__(self):
        self.api_key = MAGICEDENAPIKEY_SOLANA
        self.dextools_api_key = DEXTOOLS_API_KEY

    def fetch_holder_stats(self, symbol):
        url = f"{self.BASE_URL}/{symbol}/holder_stats"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            response = requests.get(url, headers=headers)
            print(f"\nHolder Stats API Response for {symbol}:")
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text[:500]}...")  # First 500 chars of response
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded. Consider adding delay between requests.")
                # Could add time.sleep(1) here to automatically handle rate limiting
            else:
                print(f"Unexpected status code: {response.status_code}")
            return None
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return None

    def fetch_collection_stats(self, symbol):
        url = f"{self.BASE_URL}/{symbol}/stats"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            response = requests.get(url, headers=headers)
            print(f"\nCollection Stats API Response for {symbol}:")
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text[:500]}...")  # First 500 chars of response
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded. Consider adding delay between requests.")
                # Could add time.sleep(1) here to automatically handle rate limiting
            else:
                print(f"Unexpected status code: {response.status_code}")
            return None
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return None

    def fetch_memecoin_stats(self, chain, address):
        """Fetch memecoin statistics from DEXTools API for associated token."""
        url = f"https://public-api.dextools.io/trial/v2/token/{chain}/{address}/info"
        headers = {'X-API-KEY': self.dextools_api_key}
        
        print(f"\nFetching memecoin stats for:")
        print(f"Chain: {chain}")
        print(f"Address: {address}")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Received data: {data}")
                return data
            elif response.status_code == 401:
                print("Error: Unauthorized. Check your API key.")
            elif response.status_code == 403:
                print("Error: Forbidden. Your API key might not have access to this endpoint.")
            elif response.status_code == 429:
                print("Error: Too many requests. Rate limit exceeded.")
            else:
                print(f"Error: Unexpected status code {response.status_code}")
                print(f"Response text: {response.text}")
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response.text}")
            return None

    def fetch_complete_project_data(self, symbol, chain=None, token_address=None):
        """Fetch both NFT and memecoin data if available."""
        print(f"[DEBUG] Fetching complete project data for symbol: {symbol}, chain: {chain}")  # Debugging line
        data = {
            'holder_stats': self.fetch_holder_stats(symbol),
            'collection_stats': self.fetch_collection_stats(symbol)
        }
        
        print(f"[DEBUG] Holder stats for {symbol}: {data['holder_stats']}")  # Debugging line
        print(f"[DEBUG] Collection stats for {symbol}: {data['collection_stats']}")  # Debugging line
        
        # Calculate NFT FDV
        if data['collection_stats'] and data['holder_stats']:
            floor_price = data['collection_stats'].get('floorPrice', 0)
            total_supply = data['holder_stats'].get('totalSupply', 0)
            
            print(f"[DEBUG] Initial floor price for {symbol}: {floor_price}")  # Debugging line
            
            # Conditional logic for floor price adjustment based on chain
            if chain == 'solana':
                floor_price /= 1e9  # Convert from lamports to SOL
                print(f"[DEBUG] Converted floor price for {symbol} (Solana): {floor_price}")  # Debugging line
            
            nft_fdv = (floor_price * total_supply)   # Calculate NFT FDV
            data['nft_fdv'] = nft_fdv
            data['floorPrice'] = floor_price  # Store the adjusted floor price
            print(f"[DEBUG] NFT FDV for {symbol}: {nft_fdv}")  # Debugging line
        
        # If this is a memecoin project, fetch token data and calculate ratio
        if chain and token_address:
            memecoin_stats = self.fetch_memecoin_stats(chain, token_address)
            data['memecoin_stats'] = memecoin_stats
            
            if memecoin_stats and 'fdv' in memecoin_stats:
                token_fdv = memecoin_stats['fdv']
                data['token_fdv'] = token_fdv
                
                # Calculate FDV ratio if both FDVs are available
                if nft_fdv > 0:
                    data['fdv_ratio'] = token_fdv / nft_fdv
                    print(f"\nCalculated FDV Ratio:")
                    print(f"Token FDV: {token_fdv}")
                    print(f"NFT FDV: {nft_fdv}")
                    print(f"Ratio: {data['fdv_ratio']}")
        
        return data

    def fetch_listings(self, symbol):
        """Fetch listings count from collection stats."""
        url = f"{self.BASE_URL}/{symbol}/stats"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        print(f"\nFetching listings stats for {symbol}")
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, headers=headers)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                stats = response.json()
                listings_count = stats.get('listedCount', 0)
                print(f"Found {listings_count} listings for {symbol}")
                print(f"Raw stats response: {stats}")
                return listings_count
            else:
                print(f"Error fetching stats for {symbol}: {response.status_code}")
                print(f"Response text: {response.text}")
                return 0
        except Exception as e:
            print(f"Exception while fetching stats for {symbol}: {str(e)}")
            return 0

    def fetch_collections(self, offset=0, limit=500):
        """Fetch collections from Magic Eden with pagination."""
        url = "https://api-mainnet.magiceden.dev/v2/collections"
        params = {
            "offset": offset,
            "limit": limit
        }
        headers = {"accept": "application/json"}
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching collections: {e}")
            return None

    def fetch_most_popular_collections(self, time_frame, offset=0, limit=100):
        """Fetch most popular collections based on the specified time frame."""
        url = f"{self.BASE_URL}/popular"
        params = {
            "timeFrame": time_frame,
            "offset": offset,
            "limit": limit
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        print(f"\nMaking API request:")
        print(f"URL: {url}")
        print(f"Parameters: {params}")
        print(f"Headers: {{'Authorization': 'Bearer ***'}}")  # Hide actual API key
        
        try:
            response = requests.get(url, params=params, headers=headers)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"Error Response Body: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"\nDetailed error information:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response Status Code: {e.response.status_code}")
                print(f"Response Headers: {dict(e.response.headers)}")
                print(f"Response Body: {e.response.text}")
            return None

    def get_sol_price(self):
        """Get current SOL/USD price from CoinGecko."""
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "solana",
            "vs_currencies": "usd"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()
            return data['solana']['usd']
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code} - Response: {response.text}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            print(f"Request URL: {response.url}")
            print(f"Request Params: {params}")
            if response is not None:
                print(f"Response Status Code: {response.status_code}")
                print(f"Response Content: {response.text}")
        return None
