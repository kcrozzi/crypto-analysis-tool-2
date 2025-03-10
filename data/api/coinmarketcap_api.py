import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from config import CMC_API_KEY  # Ensure the correct API key is imported

class CoinMarketCapAPI:
    BASE_URL = "https://pro-api.coinmarketcap.com"

    def __init__(self, api_key=CMC_API_KEY):
        self.api_key = api_key
        self.headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json"
        }

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def get_coinmarketcap_id(self, contract_address):
        """Fetch the CoinMarketCap ID using the contract address."""
        url = f"{self.BASE_URL}/v1/cryptocurrency/map"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            data = response.json().get("data", [])
            for crypto in data:
                if crypto.get("platform", {}).get("token_address") == contract_address:
                    return crypto["id"]
            print("Contract address not found.")
            return None
        else:
            print(f"Failed to fetch CoinMarketCap ID: {response.status_code} - {response.text}")
            response.raise_for_status()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def fetch_categories(self):
        """Fetch a list of all available categories."""
        url = f"{self.BASE_URL}/v1/cryptocurrency/categories"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Failed to fetch categories: {response.status_code} - {response.text}")
            response.raise_for_status()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def fetch_coins_from_category(self, category_id, start=1, limit=100):
        """Fetch coins from a specific category."""
        url = f"{self.BASE_URL}/v1/cryptocurrency/category"
        params = {
            "id": category_id,
            "start": start,
            "limit": limit
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json().get("data", {})
            return data.get("coins", [])
        else:
            print(f"Failed to fetch coins from category: {response.status_code} - {response.text}")
            response.raise_for_status()

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def fetch_fear_and_greed_index(self, historical=False, start=1, limit=50):
        """Fetch the latest or historical Fear and Greed Index."""
        endpoint = "/v3/fear-and-greed/historical" if historical else "/v3/fear-and-greed/latest"
        url = f"{self.BASE_URL}{endpoint}"
        params = {
            "start": start,
            "limit": limit
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json().get("data", [])
        else:
            print(f"Failed to fetch Fear and Greed Index: {response.status_code} - {response.text}")
            response.raise_for_status()

# Example usage
if __name__ == "__main__":
    api_key = CMC_API_KEY  # Use the API key from config
    cmc_api = CoinMarketCapAPI(api_key)
    
    # Fetch categories
    categories = cmc_api.fetch_categories()
    if categories:
        print(categories)

    # Fetch latest Fear and Greed Index
    fear_greed_index = cmc_api.fetch_fear_and_greed_index()
    if fear_greed_index:
        print(fear_greed_index)