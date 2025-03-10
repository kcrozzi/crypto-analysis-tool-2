from data.data_loader import DataLoader
from data.database import Database
from analysis.regression import RegressionAnalyzer
from data.api.birdeye_api import BirdeyeAPI
from config import BIRDEYE_API_KEY
import sqlite3
import json
from pathlib import Path
import requests
import sys

import time



class DataSetManager:
    def __init__(self, api_key):
        print("Loading DataSetManager from:", __file__)
        self.data_loader = DataLoader(api_key)
        self.db = Database()
        self.birdeye_api = BirdeyeAPI()

    def _map_chain_ids(self, platform_name):
        """Map CoinMarketCap platform names to the correct chain IDs."""
        mapping = {
            "Solana": "solana",
            "Ethereum": "ether",
            "Arbitrum": "arbitrum",
            "Avalanche": "avalanche",
            "BNB Chain": "bsc",
            "Optimism": "optimism",
            "Polygon": "polygon",
            "Base": "base",
            "Zksync": "zksync"
        }
        return mapping.get(platform_name, None)
    

    def create_dataset_from_category(self, category_id, cmc_api):
        """Create a dataset from a CoinMarketCap category."""
        try:
            print(f"[DEBUG] Fetching coins for category ID: {category_id}")
            coins = cmc_api.fetch_coins_from_category(category_id)
            
            if not coins:
                print("No coins found in the category.")
                return

            dataset_name = input("Enter the name for the new dataset: ").strip()
            if not self.db.create_dataset(dataset_name):
                print(f"Failed to create dataset '{dataset_name}'. It may already exist.")
                return

            for coin in coins:
                if coin is None:
                    print("[DEBUG] Skipping None coin entry")
                    continue

                print("[DEBUG] Raw coin data from CoinMarketCap:", coin)

                platform = coin.get("platform")
                if platform:
                    chain_id = self._map_chain_ids(platform.get("name"))
                    token_address = platform.get("token_address")
                else:
                    print("[DEBUG] No platform data for coin:", coin.get("name", "Unknown"))
                    continue

                if chain_id and token_address:
                    try:
                        self.add_project_to_dataset(dataset_name, chain_id, token_address)
                    except Exception as e:
                        print(f"[DEBUG] Error adding project to dataset: {str(e)}")
                else:
                    print("[DEBUG] Missing chain_id or token_address for coin:", coin.get("name", "Unknown"))

            print(f"Dataset '{dataset_name}' created with coins from category '{category_id}'.")
        
        except Exception as e:
            print(f"[DEBUG] Exception in create_dataset_from_category: {str(e)}")

    def create_dataset(self, dataset_name):
        """Create a new dataset if it doesn't already exist."""
        if self.db.dataset_exists(dataset_name):
            print(f"Dataset '{dataset_name}' already exists.")
            return False  # Indicate failure due to existing dataset

        # Proceed with dataset creation
        if self.db.create_dataset(dataset_name):
            print(f"Dataset '{dataset_name}' created successfully.")
            return True
        else:
            print(f"Failed to create dataset '{dataset_name}'.")
            return False

    def delete_dataset(self, dataset_name):
        print(f"[DEBUG] Attempting to delete dataset: {dataset_name}")
        if self.db.delete_dataset(dataset_name):
            print(f"[DEBUG] Dataset '{dataset_name}' deleted successfully.")
            return True
        else:
            print(f"[DEBUG] Dataset '{dataset_name}' does not exist or could not be deleted.")
            return False

    def add_project_to_dataset(self, dataset_name, chain_id, token_address):
        try:
            print("\n[DEBUG] Starting add_project_to_dataset")
            
            # Fetch token data using DataLoader
            token_data = self.data_loader.fetch_token_data(token_address, chain_id)
            
            # If token data is not found, try with "Solana" and then "Ethereum"
            if not token_data:
                print("[DEBUG] Original chain_id failed, trying with 'solana'")
                token_data = self.data_loader.fetch_token_data(token_address, "solana")
            
            if not token_data:
                print("[DEBUG] 'Solana' chain_id failed, trying with 'Ethereum'")
                token_data = self.data_loader.fetch_token_data(token_address, "ether")

            if token_data:
                # Calculate FDV per holder
                fdv = token_data.get('fdv')
                holders = token_data.get('holders')
                if fdv is not None and holders is not None and holders > 0:
                    token_data['fdv_per_holder'] = round(fdv / holders, 2)
                
                # Retrieve price change data from BirdeyeAPI using the correct chain ID
                #price_changes = self.birdeye_api.calculate_price_changes(token_address, chain_id)
                #token_data['price_changes'] = price_changes  # Add price change data to token data
                
                # Extract and log project name
                project_name = token_data.get('name', 'Unknown')
                print(f"[DEBUG] Project name: {project_name}")

                print("[DEBUG] Data being sent to database:", {
                    'chain_id': chain_id,
                    'token_address': token_address,
                    'data': token_data
                })
                
                # Store in database
                success = self.db.add_project(dataset_name, chain_id, token_address, token_data)
                print(f"[DEBUG] Database storage success: {success}")
                
                if success:
                    print(f"Added project '{token_data.get('name', 'Unknown')}' to dataset '{dataset_name}'.")
                else:
                    print(f"Dataset '{dataset_name}' does not exist.")
            else:
                print("Failed to retrieve token data.")
                
        except Exception as e:
            print(f"[DEBUG] Exception in add_project_to_dataset: {str(e)}")

    def _attempt_fetch_with_chain_id(self, chain_id, token_address):
        """Attempt to fetch token data with a specific chain_id."""
        try:
            print(f"[DEBUG] Attempting to fetch data with chain_id: {chain_id}")
            print(f"[DEBUG] Attempting to fetch data with token_address: {token_address}")
            
            return self.data_loader.fetch_token_data(token_address, chain_id)
        except Exception as e:
            print(f"[DEBUG] Error fetching data with chain_id '{chain_id}': {str(e)}")
            return 
        

    def list_datasets(self):
        datasets = self.db.list_datasets()
        if datasets:
            print("\nAvailable Datasets:")
            print("-" * 50)
            for name, created_at, project_count in datasets:
                print(f"Name: {name}")
                print(f"Created: {created_at}")
                print(f"Projects: {project_count}")
                print("-" * 30)
        else:
            print("No datasets available.")

    def view_dataset(self, dataset_name):
        """View a specific dataset after listing all available datasets."""
        # List all available datasets
        datasets = self.db.list_datasets()
        if not datasets:
            print("No datasets available.")
            return

        print("\nAvailable Datasets:")
        print("-" * 50)
        for name, created_at, project_count in datasets:
            print(f"Name: {name}")
            print(f"Created: {created_at}")
            print(f"Projects: {project_count}")
            print("-" * 30)

        # Fetch and display the dataset
        print(f"\nViewing dataset '{dataset_name}':")
        projects = self.db.get_dataset(dataset_name)
        
        if projects:
            for idx, project in enumerate(projects, 1):
                data = project['data']
                
                print(f"\nProject {idx}:")
                print(f"Name: {data.get('name', 'Unknown')} ({data.get('symbol', 'Unknown')})")
                print(f"Chain ID: {project['chain_id']}")
                print(f"Token Address: {project['token_address']}")
                
                print("\nFinancial Data:")

                price_changes = data.get('price_changes', {})


                metrics = {
                    "FDV": data.get('fdv'),
                    "Market Cap": data.get('mcap'),
                    "Holders": data.get('holders'),
                    "FDV per Holder": data.get('fdv_per_holder'),
                    "Total Supply": data.get('totalSupply'),
                    "Circulating Supply": data.get('circulatingSupply'),
                    "24-HR": price_changes.get('24-hour'),
                    "48-HR": price_changes.get('48-hour'),
                    "72-HR": price_changes.get('72-hour'),
                    "96-HR": price_changes.get('96-hour'),
                    "1-week": price_changes.get('1-week'),
                    "2-week": price_changes.get('2-week'),
                    "3-week": price_changes.get('3-week'),
                    "4-week": price_changes.get('4-week'),
                    "5-week": price_changes.get('5-week'),
                    "6-week": price_changes.get('6-week'),
                    "3-month": price_changes.get('3-month'),
                    "6-month": price_changes.get('6-month'),
                }

                for key, value in metrics.items():
                    if value is not None:
                        if 'FDV' in key or 'Cap' in key:
                            print(f"  {key}: ${value:,.2f}")
                        elif 'Supply' in key:
                            print(f"  {key}: {value:,.0f}")
                        else:
                            print(f"  {key}: {value:,}")
                
                print("-" * 50)
        else:
            print(f"Dataset '{dataset_name}' is empty or does not exist.")

    def analyze_fdv_relationship(self, dataset_name):
        """Analyze the relationship between FDV and FDV per holder in a dataset."""
        projects = self.db.get_dataset(dataset_name)
        if not projects:
            return None, f"Dataset '{dataset_name}' does not exist or is empty."

        # Default to 'all' regression types and full analysis
        reg_type = 'all'
        analyzer = RegressionAnalyzer()
        results = analyzer.analyze_dataset(projects, reg_type)
        
        if not results:
            return None, "No results from analysis."

        # Prepare data for plotting
        plot_data = {
            'fdv': [project['data'].get('fdv') for project in projects],
            'fdv_per_holder': [project['data'].get('fdv_per_holder') for project in projects]
        }

        # Prepare deviations text
        deviations_text = "\nDeviation Analysis (Ranked by Absolute Deviation):\n" + "-" * 50
        data_points = results['data_points']
        
        # Calculate average deviation across all regression types
        for point in data_points:
            deviations = []
            if 'linear_deviation' in point:
                deviations.append(abs(point['linear_deviation']))
            if 'poly_deviation' in point:
                deviations.append(abs(point['poly_deviation']))
            if 'power_deviation' in point:
                deviations.append(abs(point['power_deviation']))
            if 'ridge_deviation' in point:
                deviations.append(abs(point['ridge_deviation']))
            if 'log_deviation' in point:
                deviations.append(abs(point['log_deviation']))
            
            point['avg_deviation'] = sum(deviations) / len(deviations) if deviations else 0
        
        # Sort by average deviation
        sorted_points = sorted(data_points, key=lambda x: x['avg_deviation'], reverse=True)
        
        # Create ranked results text
        for rank, point in enumerate(sorted_points, 1):
            deviations_text += (
                f"\n#{rank}: {point['name']} ({point['symbol']})\n"
                f"Average Deviation: {point['avg_deviation']:.2f}%\n"
                f"FDV: ${point['fdv']:,.2f}\n"
                f"Actual FDV per Holder: ${point['fdv_per_holder']:,.2f}\n"
            )
            if 'linear_deviation' in point:
                deviations_text += f"Linear Deviation: {point['linear_deviation']:.2f}%\n"
            if 'poly_deviation' in point:
                deviations_text += f"Polynomial Deviation: {point['poly_deviation']:.2f}%\n"
            if 'power_deviation' in point:
                deviations_text += f"Power Deviation: {point['power_deviation']:.2f}%\n"
            if 'ridge_deviation' in point:
                deviations_text += f"Ridge Deviation: {point['ridge_deviation']:.2f}%\n"
            if 'log_deviation' in point:
                deviations_text += f"Logarithmic Deviation: {point['log_deviation']:.2f}%\n"
            deviations_text += "-" * 30

        return plot_data, deviations_text

    def analyze_with_custom_regression(self, dataset_name, custom_analyzer):
        projects = self.db.get_dataset(dataset_name)
        if not projects:
            print(f"Dataset '{dataset_name}' does not exist or is empty.")
            return

        selected_equation = custom_analyzer.select_custom_equation()
        if not selected_equation:
            print("No valid equation selected.")
            return

        print(f"\nAnalyzing with custom {selected_equation['type']} equation: {selected_equation['coefficients']}")
        results = []
        for project in projects:
            fdv = project['data'].get('fdv')
            if fdv is None:
                continue

            if selected_equation['type'] == 'power':
                a, b = selected_equation['coefficients']
                expected_value = a * (fdv ** b)
            elif selected_equation['type'] == 'polynomial':
                a, b, c = selected_equation['coefficients']
                expected_value = a * (fdv ** 2) + b * fdv + c

            fdv_per_holder = project['data'].get('fdv_per_holder')
            if fdv_per_holder is not None:
                deviation = ((fdv_per_holder - expected_value) / expected_value) * 100
                results.append({
                    'name': project['data'].get('name', 'Unknown'),
                    'symbol': project['data'].get('symbol', 'Unknown'),
                    'fdv': fdv,
                    'fdv_per_holder': fdv_per_holder,
                    'expected_value': expected_value,
                    'deviation': deviation
                })

        filtered_results = self.filter_results(results, deviation_threshold=1000)
        for result in filtered_results:
            print(f"Token: {result['name']} ({result['symbol']})")
            print(f"FDV: {result['fdv']}, Expected: {result['expected_value']:.2f}, Actual: {result['fdv_per_holder']:.2f}, Deviation: {result['deviation']:.2f}%")
            print("-" * 30)

    def filter_results(self, results, deviation_threshold=100):
        """Filter results based on deviation."""
        return [x for x in results if abs(x['deviation']) <= deviation_threshold]

    def copy_dataset_with_fdv_filter(self, source_dataset_name, new_dataset_name, min_fdv, max_fdv):
        """Copy projects from one dataset to a new one, filtered by FDV range."""
        print(f"[DEBUG] Starting copy_dataset_with_fdv_filter")
        print(f"[DEBUG] Source dataset: {source_dataset_name}, New dataset: {new_dataset_name}")
        print(f"[DEBUG] FDV range: {min_fdv} to {max_fdv}")

        try:
            projects = self.db.get_dataset(source_dataset_name)
            if not projects:
                print(f"[DEBUG] Source dataset '{source_dataset_name}' does not exist or is empty.")
                return

            print(f"[DEBUG] Number of projects in source dataset: {len(projects)}")

            # Filter projects by FDV
            filtered_projects = []
            for project in projects:
                fdv = project['data'].get('fdv')
                if fdv is None:
                    print(f"[DEBUG] Project '{project['data'].get('name', 'Unknown')}' has no FDV value and is excluded from filter.")
                    continue

                if min_fdv <= fdv <= max_fdv:
                    filtered_projects.append(project)
                else:
                    print(f"[DEBUG] Project '{project['data'].get('name', 'Unknown')}' with FDV {fdv} excluded from filter.")

            if not filtered_projects:
                print(f"[DEBUG] No projects found with FDV between {min_fdv} and {max_fdv}.")
                return

            print(f"[DEBUG] Number of filtered projects: {len(filtered_projects)}")

            # Create new dataset
            if not self.db.create_dataset(new_dataset_name):
                print(f"[DEBUG] Failed to create new dataset '{new_dataset_name}'. It may already exist.")
                return

            # Add filtered projects to the new dataset
            for project in filtered_projects:
                self.db.add_project(new_dataset_name, project['chain_id'], project['token_address'], project['data'])
                print(f"[DEBUG] Added project '{project['data'].get('name', 'Unknown')}' to new dataset '{new_dataset_name}'.")

            print(f"[DEBUG] New dataset '{new_dataset_name}' created with {len(filtered_projects)} projects.")

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")

    def refresh_dataset(self, dataset_name):
        """Refresh all projects in a dataset with the latest data."""
        projects = self.db.get_dataset(dataset_name)
        RATE_LIMIT_DELAY = 1.3

        if not projects:
            print(f"Dataset '{dataset_name}' does not exist or is empty.")
            return False  # Indicate failure if no projects are found

        all_successful = True  # Track overall success

        for project in projects:
            chain_id = project['chain_id']
            token_address = project['token_address']

            # Fetch the latest token data
            print(f"Refreshing data for token: {token_address} on chain: {chain_id}")
            token_data = self.data_loader.fetch_token_data(token_address, chain_id)

            if token_data:
                # Calculate FDV per holder
                fdv = token_data.get('fdv')
                holders = token_data.get('holders')
                if fdv is not None and holders is not None and holders > 0:
                    token_data['fdv_per_holder'] = round(fdv / holders, 2)

                # Fetch and add price change data
                #try:
                #    price_changes = self.birdeye_api.calculate_price_changes(token_address)
                #    token_data['price_changes'] = price_changes
                #except Exception as e:
                #    print(f"Failed to fetch price changes for token '{token_address}': {e}")

                # Update the project in the database
                self.db.update_project(dataset_name, chain_id, token_address, token_data)
                print(f"Updated project '{token_address}' in dataset '{dataset_name}'.")
            else:
                print(f"Failed to refresh data for token '{token_address}'.")
                all_successful = False  # Mark as unsuccessful if any project fails

            # Wait for the rate limit delay
            time.sleep(RATE_LIMIT_DELAY)

        return all_successful  # Return overall success status

    def delete_project(self, dataset_name, token_address):
        """Delete a project from a dataset."""
        result = self.db.delete_project(dataset_name, token_address)
        if result:
            print(f"Project {token_address} deleted from dataset '{dataset_name}'.")
        else:
            print(f"Project not found or could not be deleted.")
        return result

    def analyze_dataset_by_volume(self, dataset_name):
        projects = self.db.get_dataset(dataset_name)
        if not projects:
            print(f"Dataset '{dataset_name}' does not exist or is empty.")
            return

        results = []
        RATE_LIMIT_DELAY = 1.3  # Add delay between requests
        
        for project in projects:
            try:
                token_address = project.get('token_address')  # Use .get() instead of direct access
                chain_id = project.get('chain_id')
                fdv = project.get('data', {}).get('fdv')  # Safely access nested data

                if not all([token_address, chain_id, fdv]):  # Check if we have all required data
                    print(f"Skipping project due to missing data: {project.get('data', {}).get('name', 'Unknown')}")
                    continue

                if fdv <= 0:
                    continue

                # Fetch 24H volume data with error handling
                try:
                    volume_24h = self.fetch_24h_volume(token_address, chain_id)
                    if volume_24h is None or volume_24h == 0:
                        print(f"No volume data for: {project.get('data', {}).get('name', 'Unknown')}")
                        continue
                    
                    # Add delay to respect rate limits
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"Error fetching volume data: {str(e)}")
                    continue

                # Calculate 24H V / FDV
                volume_fdv_ratio = volume_24h / fdv

                results.append({
                    'name': project.get('data', {}).get('name', 'Unknown'),
                    'symbol': project.get('data', {}).get('symbol', 'Unknown'),
                    'fdv': fdv,
                    'volume_fdv_ratio': volume_fdv_ratio
                })

            except Exception as e:
                print(f"Error processing project: {str(e)}")
                continue

        if not results:
            print("No valid data points for analysis.")
            return

        # Perform regression analysis
        try:
            analyzer = RegressionAnalyzer()
            regression_results = analyzer.analyze_dataset(results, 'all')

            if regression_results:
                analyzer.print_full_analysis(regression_results)
            else:
                print("No regression results available.")
        except Exception as e:
            print(f"Error in regression analysis: {str(e)}")

    def fetch_24h_volume(self, token_address, chain_id='solana'):
        """Fetch 24h volume for a token from Birdeye API."""
        chain_mapping = {
            'ether': 'ethereum',
            'ethereum': 'ethereum',
            'solana': 'solana',
            'arbitrum': 'arbitrum',
            'base': 'base',
            'optimism': 'optimism',
            'polygon': 'polygon'
        }
        
        chain = chain_mapping.get(chain_id.lower(), chain_id)
        
        url = f"https://public-api.birdeye.so/public/token_overview"
        
        headers = {
            "accept": "application/json",
            "x-chain": chain,
            "X-API-KEY": BIRDEYE_API_KEY
        }
        
        params = {
            "address": token_address,
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            # Extract volume from the correct path in the response
            volume = data.get("data", {}).get("volume", {}).get("h24", 0)
            return volume
            
        except Exception as e:
            print(f"Error fetching volume data: {str(e)}")
            return None

    def refresh_dataset_remove_duplicates(self, original_dataset_name, new_dataset_name):
        """Refresh dataset and remove duplicate projects, keeping one instance of each."""
        try:
            print(f"\n[DEBUG] Starting refresh with duplicate removal")
            print(f"Original dataset: '{original_dataset_name}'")
            print(f"New dataset: '{new_dataset_name}'")
            
            # Get all projects in the dataset
            projects = self.db.get_dataset(original_dataset_name)
            if not projects:
                print(f"Dataset '{original_dataset_name}' does not exist or is empty.")
                return False

            # Create a dictionary to track unique token addresses
            unique_projects = {}
            duplicates_found = False

            # Identify duplicates while keeping the first instance
            for project in projects:
                token_address = project['token_address'].lower()  # Normalize address
                if token_address not in unique_projects:
                    unique_projects[token_address] = project
                else:
                    duplicates_found = True
                    print(f"Found duplicate for token address: {token_address}")

            if not duplicates_found:
                print("No duplicates found in original dataset.")
                return False

            # Create the new dataset
            if not self.db.create_dataset(new_dataset_name):
                print("Failed to create new dataset.")
                return False

            # Add unique projects to new dataset
            for project in unique_projects.values():
                self.db.add_project(
                    new_dataset_name,
                    project['chain_id'],
                    project['token_address'],
                    project['data']
                )

            # Now refresh the new dataset
            print("Refreshing new dataset...")
            return self.refresh_dataset(new_dataset_name)

        except Exception as e:
            print(f"Error in refresh_dataset_remove_duplicates: {str(e)}")
            return False

    def analyze_float_metrics(self, dataset_name):
        """Analyze float metrics for all tokens in a dataset."""
        print(f"\nAnalyzing float metrics for dataset: {dataset_name}")
        
        dataset = self.db.get_dataset(dataset_name)
        if not dataset:
            print("Dataset not found.")
            return None
        
        results = []
        for project in dataset:
            try:
                token_data = project.get('data', {})
                chain_id = project.get('chain_id')
                token_address = project.get('token_address')
                
                # Fetch fresh float data
                float_data = self.data_loader.fetch_float_data(token_address, chain_id)
                if not float_data:
                    continue
                    
                # Calculate float metrics with safe conversion
                total_supply = float(float_data.get('totalSupply', 0) or 0)
                circ_supply = float(float_data.get('circulatingSupply', 0) or 0)
                locked_supply = float(float_data.get('lockedSupply', 0) or 0)
                burned_supply = float(float_data.get('burnedSupply', 0) or 0)
                
                if total_supply > 0:
                    metrics = {
                        'name': token_data.get('name', 'Unknown'),
                        'symbol': token_data.get('symbol', 'Unknown'),
                        'float_ratio': circ_supply / total_supply if total_supply > 0 else 0,
                        'locked_ratio': locked_supply / total_supply if total_supply > 0 else 0,
                        'burned_ratio': burned_supply / total_supply if total_supply > 0 else 0,
                        'lock_score': float(float_data.get('lockScore', 0) or 0),
                        'raw_data': float_data
                    }
                    results.append(metrics)
                    print(f"Processed {metrics['symbol']}: Float ratio = {metrics['float_ratio']*100:.2f}%")
                    
            except Exception as e:
                print(f"Error processing project {project.get('token_address')}: {str(e)}")
                continue

        if results:
            self._print_float_analysis(results)
            self._plot_float_distribution(results)
            return results  # Return the results
        else:
            print("No valid float data found for analysis.")
            return None

    def _print_float_analysis(self, results):
        """Print float analysis results."""
        print("\nFloat Analysis Results:")
        print("-" * 50)
        
        # Sort by float ratio (ascending - lower float first)
        results.sort(key=lambda x: x['float_ratio'])
        
        for rank, project in enumerate(results, 1):
            print(f"\n#{rank}: {project['name']} ({project['symbol']})")
            print(f"Float Ratio: {project['float_ratio']*100:.2f}%")
            print(f"Locked Ratio: {project['locked_ratio']*100:.2f}%")
            print(f"Burned Ratio: {project['burned_ratio']*100:.2f}%")
            print(f"Lock Score: {project['lock_score']}")

    def _plot_float_distribution(self, results):
        """Plot float distribution analysis."""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with multiple subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Extract data
            names = [r['symbol'] for r in results]
            float_ratios = [r['float_ratio'] * 100 for r in results]
            locked_ratios = [r['locked_ratio'] * 100 for r in results]
            
            # Plot float ratio distribution
            ax1.bar(names, float_ratios, color='blue', alpha=0.6)
            ax1.set_title('Float Ratio Distribution')
            ax1.set_ylabel('Float Ratio (%)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.2)  # Add grid to first subplot
            
            # Plot locked ratio distribution
            ax2.bar(names, locked_ratios, color='green', alpha=0.6)
            ax2.set_title('Locked Ratio Distribution')
            ax2.set_ylabel('Locked Ratio (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.2)  # Add grid to second subplot
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating float distribution plot: {str(e)}")

    def create_memecoin_dataset(self, original_dataset_name, new_dataset_name):
        """Create a new memecoin dataset from an existing dataset."""
        if self.db.create_memecoin_dataset(original_dataset_name, new_dataset_name):
            print(f"Memecoin dataset '{new_dataset_name}' created from '{original_dataset_name}'.")
        else:
            print(f"Failed to create memecoin dataset.")

    def get_memecoin_dataset(self, dataset_name):
        """Get all projects in a memecoin dataset."""
        return self.db.get_memecoin_dataset(dataset_name)