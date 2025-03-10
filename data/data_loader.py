# data/data_loader.py
import requests
import sys
import time

# Add the root directory to the system path to access config.py
sys.path.append("C:/Users/Kyle/crypto-analysis-tool")  # Adjust path as needed

from config import DEXTOOLS_API_KEY

class DataLoader:
    RATE_LIMIT_DELAY = 1.1  # Slightly over 1 second to be safe

    def __init__(self, api_key):
        self.base_url = "https://public-api.dextools.io/trial/v2/token"
        self.headers = {
            "x-api-key": api_key
        }

    def fetch_token_data(self, token_address, chain_id):
        try:
            print("\n[DEBUG] Starting fetch_token_data")
            
            # Validate parameters
            if not self._validate_parameters(token_address, chain_id):
                print("[DEBUG] Invalid parameters for token data fetch.")
                return None


            # Fetch token financial info
            print("[DEBUG] About to fetch financial data")
            financial_data = self._fetch_financial_data(token_address, chain_id)
            
            # Add delay before second API call
            print("[DEBUG] Waiting for rate limit...")
            time.sleep(self.RATE_LIMIT_DELAY)
            
            # Fetch token description info
            print("[DEBUG] About to fetch description data")
            description_data = self._fetch_token_description(token_address, chain_id)
            print("[DEBUG] Description data:", description_data)
            
            # Combine data
            token_data = {}
            if financial_data:
                token_data.update(financial_data)
            if description_data:
                token_data.update(description_data)
            
            print("[DEBUG] Final combined data:", token_data)
            return token_data if token_data else None
            
        except Exception as e:
            print(f"[DEBUG] Exception in fetch_token_data: {str(e)}")
            return None

    def _validate_parameters(self, token_address, chain_id):
        """Validate token address and chain ID."""
        if not token_address or not isinstance(token_address, str):
            print("[DEBUG] Invalid token address.")
            return False
        if not chain_id or not isinstance(chain_id, str):
            print("[DEBUG] Invalid chain ID.")
            return False
        return True

    def _fetch_financial_data(self, token_address, chain_id):
        url = f"{self.base_url}/{chain_id}/{token_address}/info"
        print(f"\nFinancial data URL: {url}")
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            financial_data = {
                "circulatingSupply": data.get("circulatingSupply"),
                "totalSupply": data.get("totalSupply"),
                "mcap": data.get("mcap"),
                "fdv": data.get("fdv"),
                "holders": data.get("holders"),
                "transactions": data.get("transactions")
            }
            print("Financial data retrieved:", financial_data)
            return financial_data
        else:
            print(f"Failed to fetch financial data: Status {response.status_code}")
            print(f"Response: {response.text}")
            return None

    def _fetch_token_description(self, token_address, chain_id):
        url = f"{self.base_url}/{chain_id}/{token_address}"
        print(f"\nDEBUG: Attempting to fetch description data")
        print(f"URL: {url}")
        print(f"Headers: {self.headers}")
        
        try:
            response = requests.get(url, headers=self.headers)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            
            if response.status_code == 200:
                raw_data = response.json()
                print("\nRaw Description API Response:", raw_data)
                return {
                    "address": raw_data.get("data", {}).get("address"),
                    "name": raw_data.get("data", {}).get("name"),
                    "symbol": raw_data.get("data", {}).get("symbol"),
                    "decimals": raw_data.get("data", {}).get("decimals"),
                    "creationTime": raw_data.get("data", {}).get("creationTime"),
                    "creationBlock": raw_data.get("data", {}).get("creationBlock")
                }
            else:
                print(f"Error Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Exception occurred while fetching description data: {str(e)}")
            return None

    def fetch_float_data(self, token_address, chain_id):
        """Fetch detailed float information for a token with rate limiting."""
        try:
            url = f"{self.base_url}/{chain_id}/{token_address}/info"
            print(f"\n[DEBUG] Fetching float data from: {url}")
            
            # Add consistent rate limiting delay
            time.sleep(self.RATE_LIMIT_DELAY)  # 1.1 seconds
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json().get("data", {})
                float_data = {
                    "totalSupply": data.get("totalSupply"),
                    "circulatingSupply": data.get("circulatingSupply"),
                    "lockedSupply": data.get("lockedSupply", 0),
                    "burnedSupply": data.get("burnedSupply", 0),
                    "lockScore": data.get("lockScore", 0)
                }
                print(f"[DEBUG] Float data retrieved: {float_data}")
                return float_data
            else:
                print(f"[DEBUG] Failed to fetch float data: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[DEBUG] Error fetching float data: {str(e)}")
            return None