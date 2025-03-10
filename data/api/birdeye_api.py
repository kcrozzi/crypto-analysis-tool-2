import requests
from config import BIRDEYE_API_KEY
from datetime import datetime, timedelta


class BirdeyeAPI:
    def __init__(self):
        self.base_url = "https://public-api.birdeye.so/defi/history_price"

    def get_historical_price(self, token_address, time_from, time_to, data_type,chain_id):
        headers = {
            "accept": "application/json",
            "x-chain": chain_id,  # Add the chain_id as a header
            "X-API-KEY": BIRDEYE_API_KEY
        }
        params = {
            "address": token_address,
            "address_type": "token",
            "type": data_type,
            "time_from": time_from,
            "time_to": time_to
        }
        
        # Debugging output to print the full request details
        print(f"Request URL: {self.base_url}")
        print(f"Request Headers: {headers}")
        print(f"Request Params: {params}")
        
        response = requests.get(self.base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            print("Full API response:", response.json())  # Print the full response
            return response.json()
        else:
            print(f"Error response: {response.text}")  # Debugging output
            response.raise_for_status()

    def calculate_price_changes(self, token_address, chain_id=None):
        # Adjust chain_id if it's "ether"
        print(chain_id)

        print(chain_id == 'ether')


        if chain_id == "ether":
            chain_id = "ethereum"

        print(chain_id)

        # Calculate the Unix timestamps for today and 1 year ago
        time_to = int(datetime.now().timestamp())
        time_from_4h = int((datetime.now() - timedelta(days=5)).timestamp())  # 5 days for 4H data
        time_from_1w = int((datetime.now() - timedelta(weeks=52)).timestamp())  # 52 weeks for 1W data
        
        # Fetch historical price data for 4H and 1W intervals
        price_data_4h = self.get_historical_price(token_address, time_from_4h, time_to, '4H', chain_id)
        price_data_1w = self.get_historical_price(token_address, time_from_1w, time_to, '1W',chain_id)
        
        changes = {}

        # Calculate changes using 4H data
        if price_data_4h and "data" in price_data_4h and "items" in price_data_4h["data"]:
            prices_4h = price_data_4h["data"]["items"]
            intervals_4h = {
                "24-hour": 6,  # 6 intervals of 4 hours each
                "48-hour": 12,
                "72-hour": 18,
                "96-hour": 24,
            }
            
            for label, intervals_count in intervals_4h.items():
                if len(prices_4h) < intervals_count + 1:
                    print(f"Not enough data to calculate {label} change.")
                    continue
                
                start_index = len(prices_4h) - intervals_count - 1
                start_price = prices_4h[start_index]["value"]
                end_price = prices_4h[-1]["value"]
                percentage_change = ((end_price - start_price) / start_price) * 100
                changes[label] = percentage_change

        # Calculate changes using 1W data
        if price_data_1w and "data" in price_data_1w and "items" in price_data_1w["data"]:
            prices_1w = price_data_1w["data"]["items"]
            intervals_1w = {
                "1-week": 1,
                "2-week": 2,
                "3-week": 3,
                "4-week": 4,
                "5-week": 5,
                "6-week": 6,
                "3-month": 12,  # Approximately 3 months
                "6-month": 26,  # Approximately 6 months
            }
            
            for label, intervals_count in intervals_1w.items():
                if len(prices_1w) < intervals_count + 1:
                    print(f"Not enough data to calculate {label} change.")
                    continue
                
                start_index = len(prices_1w) - intervals_count - 1
                start_price = prices_1w[start_index]["value"]
                end_price = prices_1w[-1]["value"]
                percentage_change = ((end_price - start_price) / start_price) * 100
                changes[label] = percentage_change

        return changes

    def get_24h_volume(self, token_address, chain_id):
        headers = {
            "accept": "application/json",
            "x-chain": chain_id,
            "X-API-KEY": BIRDEYE_API_KEY
        }
        params = {
            "address": token_address,
            "address_type": "token",
            "type": "24H"
        }
        
        response = requests.get(self.base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Assuming the response contains a field 'volume' for 24H volume
            return data.get('volume', 0)
        else:
            print(f"Error fetching 24H volume: {response.text}")
            return 0

# Example usage
if __name__ == "__main__":
    birdeye_api = BirdeyeAPI()
    
    # Example token address
    token_address = "So11111111111111111111111111111111111111112"  # Use default address for testing
    
    try:
        changes = birdeye_api.calculate_price_changes(token_address)
        for interval, change in changes.items():
            print(f"{interval} price change: {change:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")