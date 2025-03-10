import requests
import time
from nft_config import MAGICEDENAPIKEY

class EthereumAPI:
    def __init__(self):
        self.api_key = MAGICEDENAPIKEY
        self.base_url = "https://api-mainnet.magiceden.dev/v3/rtp"

    def fetch_collection_stats(self, contract_address):
        """Fetch collection stats for an Ethereum collection."""
        url = f"{self.base_url}/ethereum/collections/v7"
        params = {
            "id": contract_address,
            "includeMintStages": "false",
            "includeSecurityConfigs": "false",
            "normalizeRoyalties": "false",
            "useNonFlaggedFloorAsk": "false"
        }
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            print(f"\nCollection Stats API Response for {contract_address}:")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data and 'collections' in data and len(data['collections']) > 0:
                    collection = data['collections'][0]
                    
                    # Extract relevant stats
                    stats = {
                        'floorPrice': collection.get('floorAsk', {}).get('price', {}).get('amount', {}).get('usd', 0),
                        'listedCount': collection.get('onSaleCount', 0),
                        'volume': collection.get('volume', {}).get('allTime', 0)
                    }
                    return stats
            return None
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return None

    def fetch_holder_stats(self, contract_address):
        """Fetch holder stats for an Ethereum collection."""
        url = f"{self.base_url}/ethereum/collections/v7"
        params = {
            "id": contract_address,
            "includeMintStages": "false",
            "includeSecurityConfigs": "false"
        }
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and 'collections' in data and len(data['collections']) > 0:
                    collection = data['collections'][0]
                    return {
                        'totalSupply': int(collection.get('supply', 0)),
                        'uniqueHolders': collection.get('ownerCount', 0)
                    }
            return None
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return None

    def fetch_complete_project_data(self, contract_address):
        """Fetch all relevant data for an Ethereum collection in a single API call."""
        url = f"{self.base_url}/ethereum/collections/v7"
        params = {
            "id": contract_address,
            "includeMintStages": "false",
            "includeSecurityConfigs": "false"
        }
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and 'collections' in data and len(data['collections']) > 0:
                    collection = data['collections'][0]
                    
                    # Get floor price in USD from the nested structure
                    floor_price = collection.get('floorAsk', {}).get('price', {}).get('amount', {}).get('usd', 0)
                    
                    return {
                        'collection_stats': {
                            'floorPrice': floor_price,  # Already in USD
                            'listedCount': collection.get('onSaleCount', 0),
                            'volume': collection.get('volume', {}).get('allTime', 0)
                        },
                        'holder_stats': {
                            'totalSupply': int(collection.get('supply', 0)),
                            'uniqueHolders': collection.get('ownerCount', 0)
                        }
                    }
            return None
        except Exception as e:
            print(f"Error fetching complete project data: {str(e)}")
            return None

    def fetch_listings(self, contract_address):
        """Fetch number of listings for an Ethereum collection."""
        url = f"{self.base_url}/ethereum/collections/v7"
        params = {"id": contract_address}
        headers = {
            "accept": "*/*",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data and 'collections' in data and len(data['collections']) > 0:
                    return data['collections'][0].get('onSaleCount', 0)
            return 0
        except Exception as e:
            print(f"Error fetching listings: {str(e)}")
            return 0
