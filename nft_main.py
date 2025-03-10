import sys
import time
from nft_data.nft_database import SolanaDatabase
from nft_analysis.solana_regression import SolanaRegressionAnalyzer
from nft_nft.nft_api import SolanaAPI
from nft_analysis.market_analyzer import MarketAnalyzer
from nft_project import Project
import json
from nft_nft.ethereum_api import EthereumAPI

def main():
    # Initialize modules
    solana_db = SolanaDatabase()
    solana_api = SolanaAPI()
    ethereum_api = EthereumAPI()
    regression_analyzer = SolanaRegressionAnalyzer()
    market_analyzer = MarketAnalyzer(solana_db)

    while True:
        print("\nNFT Analysis Tool")
        print("1. Create a new Solana NFT dataset")
        print("2. Create a new Ethereum NFT dataset")
        print("3. Create a new Solana NFT+Memecoin dataset")
        print("4. Add a project to a dataset")
        print("5. View a dataset")
        print("6. Analyze a dataset with regression")
        print("7. Analyze dataset listings")
        print("8. Refresh a dataset")
        print("9. Delete a project from a dataset")
        print("10. Find undervalued projects")
        print("11. Duplicate dataset")
        print("12. Analyze memecoin dataset")
        print("13. Exit")
        print("14. Return to Main Dashboard")
        
        choice = input("Enter your choice: ").strip()

        if choice == "14":
            break
        elif choice == "1":
            dataset_name = input("Enter the name for the new dataset: ").strip()
            solana_db.create_dataset(dataset_name, dataset_type='solana')
        elif choice == "2":
            dataset_name = input("Enter the name for the new dataset: ").strip()
            solana_db.create_dataset(dataset_name, dataset_type='ethereum')
        elif choice == "3":
            dataset_name = input("Enter the name for the new dataset: ").strip()
            solana_db.create_dataset(dataset_name, dataset_type='solana', is_memecoin=True)
        elif choice == "4":
            dataset_name = input("Enter the dataset name: ").strip()
            
            # Get dataset type
            dataset_type = solana_db.get_dataset_type(dataset_name)
            if not dataset_type:
                print("Dataset not found.")
                continue
            
            if dataset_type == 'ethereum':
                print("\nEnter Ethereum contract addresses (one per line, blank line to finish):")
            else:
                print("\nEnter project symbols (one per line, blank line to finish):")
            
            symbols = []
            while True:
                if dataset_type == 'ethereum':
                    symbol = input("Contract address: ").strip()
                else:
                    symbol = input().strip()
                if not symbol:
                    break
                symbols.append(symbol)
            
            if not symbols:
                print("No symbols/addresses entered.")
                continue
            
            if dataset_type == 'ethereum':
                solana_db.add_projects(dataset_name, symbols, ethereum_api)
            else:  # solana
                solana_db.add_projects(dataset_name, symbols, solana_api)
        elif choice == "5":
            available_datasets = solana_db.list_datasets()
            if not available_datasets:
                print("No datasets available.")
                continue
            
            print("\nAvailable datasets:")
            for dataset in available_datasets:
                print(f"- {dataset}")
            
            while True:
                dataset_name = input("\nEnter the dataset name to view (or 'cancel' to go back): ").strip()
                if dataset_name.lower() == 'cancel':
                    break
                if dataset_name in available_datasets:
                    solana_db.view_dataset(dataset_name)
                    break
                else:
                    print(f"Invalid dataset name. Please choose from the available datasets.")
        elif choice == "6":
            dataset_name = input("Enter the dataset name to analyze: ").strip()
            dataset = solana_db.get_dataset(dataset_name)
            if dataset:
                results = regression_analyzer.analyze_dataset(dataset)
                regression_analyzer.print_results(results)
                regression_analyzer.plot_results(results)
            else:
                print("No dataset found.")
        elif choice == "7":
            dataset_name = input("Enter the dataset name to analyze listings: ").strip()
            dataset = solana_db.get_dataset(dataset_name)
            
            if dataset:
                print("\nFetching listings data for all collections...")
                listings_data = {}
                for project in dataset:
                    symbol = project["symbol"]
                    listings = solana_api.fetch_listings(symbol)
                    listings_data[symbol] = listings
                
                results = regression_analyzer.analyze_listings(dataset, listings_data)
                if results:
                    regression_analyzer.print_listings_results(results)
                    regression_analyzer.plot_listings_analysis(results)
            else:
                print("No dataset found.")
        elif choice == "8":
            dataset_name = input("Enter the dataset name to refresh: ").strip()
            solana_db.refresh_dataset(dataset_name, solana_api)
        elif choice == "9":
            dataset_name = input("Enter the dataset name: ").strip()
            
            # Get dataset type
            dataset_type = solana_db.get_dataset_type(dataset_name)
            if not dataset_type:
                print("Dataset not found.")
                continue
            
            if dataset_type == 'ethereum':
                print("\nEnter Ethereum contract addresses to delete (one per line, blank line to finish):")
            else:
                print("\nEnter project symbols to delete (one per line, blank line to finish):")
            
            symbols = []
            while True:
                if dataset_type == 'ethereum':
                    symbol = input("Contract address: ").strip()
                else:
                    symbol = input().strip()
                if not symbol:
                    break
                symbols.append(symbol)
            
            if not symbols:
                print("No symbols/addresses entered.")
                continue
            
            solana_db.delete_project(dataset_name, symbols)
        elif choice == "10":
            # Get SOL price first
            sol_price = solana_api.get_sol_price()
            if not sol_price:
                print("Error: Could not fetch SOL price. Analysis will use SOL units.")
            else:
                print(f"Current SOL price: ${sol_price}")

            # Add floor price filter option
            filter_choice = input("\nWould you like to filter the output by floor price (USD)? (y/n): ").lower()
            min_floor = None
            max_floor = None
            
            if filter_choice == 'y':
                try:
                    min_input = input("Enter minimum floor price in USD (press Enter to skip): ").strip()
                    max_input = input("Enter maximum floor price in USD (press Enter to skip): ").strip()
                    
                    if min_input:
                        min_floor = float(min_input)
                    if max_input:
                        max_floor = float(max_input)
                        
                    if min_floor is not None and max_floor is not None and min_floor > max_floor:
                        print("Error: Minimum floor price cannot be greater than maximum floor price.")
                        continue
                        
                    print(f"\nWill show results for projects with floor price:", end=" ")
                    if min_floor is not None:
                        print(f"above ${min_floor:,.2f}", end="")
                    if min_floor is not None and max_floor is not None:
                        print(" and", end=" ")
                    if max_floor is not None:
                        print(f"below ${max_floor:,.2f}", end="")
                    print()
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")
                    continue

            # Let user select multiple datasets
            available_datasets = solana_db.list_datasets()
            if not available_datasets:
                print("No datasets available.")
                continue
            
            print("\nAvailable datasets:")
            for dataset in available_datasets:
                dataset_type = solana_db.get_dataset_type(dataset)
                print(f"- {dataset} ({dataset_type})")
            
            print("\nEnter dataset names (one per line, blank line to finish):")
            selected_datasets = []
            while True:
                dataset_name = input().strip()
                if not dataset_name:
                    break
                if dataset_name in available_datasets:
                    selected_datasets.append(dataset_name)
                else:
                    print(f"Dataset '{dataset_name}' not found. Skipping.")
            
            if not selected_datasets:
                print("No valid datasets selected.")
                continue
            
            # Ask about refreshing before processing
            update_choice = input("\nWould you like to refresh the datasets before analysis? (y/n): ").lower()
            
            # Process each dataset
            combined_dataset = []
            listings_data = {}
            
            for dataset_name in selected_datasets:
                dataset_type = solana_db.get_dataset_type(dataset_name)
                
                # Only refresh if user chose to
                if update_choice == 'y':
                    if dataset_type == 'solana':
                        print(f"\nRefreshing Solana dataset '{dataset_name}'...")
                        solana_db.refresh_dataset(dataset_name, solana_api)
                    else:  # ethereum
                        print(f"\nRefreshing Ethereum dataset '{dataset_name}'...")
                        solana_db.refresh_dataset(dataset_name, ethereum_api)
                
                dataset = solana_db.get_dataset(dataset_name)
                if dataset:
                    # Get listings data for ALL projects (for regression analysis)
                    for project in dataset:
                        symbol = project["symbol"]
                        if dataset_type == 'ethereum':
                            listings_data[symbol] = int(project['collection_stats'].get('listedCount', 0))
                        else:
                            listings = solana_api.fetch_listings(symbol)
                            listings_data[symbol] = int(listings)
                    
                    combined_dataset.extend(dataset)
            
            if combined_dataset:
                # Convert all floor prices to USD before analysis
                for project in combined_dataset:
                    dataset_type = project.get('dataset_type')
                    if dataset_type == 'solana':
                        # Convert lamports to SOL first (1 SOL = 1e9 lamports)
                        if 'collection_stats' in project:
                            lamports_floor = project['collection_stats'].get('floorPrice', 0)
                            sol_floor = lamports_floor / 1e9  # Convert lamports to SOL
                            
                            # Store both the original SOL price and USD price
                            project['floor_price_sol'] = sol_floor
                            project['floor_price'] = sol_floor  # Don't multiply by sol_price here
                            project['fdv'] = sol_floor * project.get('total_supply', 0)  # Don't multiply by sol_price here
                    else:  # ethereum projects are already in USD
                        project['floor_price'] = project['collection_stats'].get('floorPrice', 0)
                        project['fdv'] = project['floor_price'] * project.get('total_supply', 0)

                # Perform regression analysis using ALL projects
                analysis_results = regression_analyzer.find_undervalued_projects(
                    combined_dataset, 
                    listings_data,
                    solana_api=solana_api,
                    dataset_type=dataset_type
                )
                
                # Filter results for display if requested
                if filter_choice == 'y':
                    filtered_results = []
                    for result in analysis_results:
                        floor_price = result['floor_price']  # Now consistently in USD
                        
                        if ((min_floor is None or floor_price >= min_floor) and 
                            (max_floor is None or floor_price <= max_floor)):
                            filtered_results.append(result)
                    
                    if not filtered_results:
                        print("\nNo projects match the specified floor price criteria.")
                        continue
                        
                    print(f"\nShowing {len(filtered_results)} projects that match the floor price criteria")
                    print(f"(Regression analysis was performed using all {len(analysis_results)} projects)")
                    
                    # Display filtered results while keeping original regression data
                    regression_analyzer._print_deviation_results(filtered_results)
                else:
                    # Display all results
                    regression_analyzer._print_deviation_results(analysis_results)
                
                # Add resorting option
                while True:
                    resort_choice = input("\nWould you like to resort with different weights? (y/n): ").lower()
                    if resort_choice != 'y':
                        break
                        
                    try:
                        print("\nEnter new weights for analysis:")
                        new_listings_weight = float(input("Listings weight (default=3): ") or 3)
                        new_fdv_weight = float(input("FDV weight (default=1): ") or 1)
                        new_ownership_weight = float(input("Ownership weight (default=1): ") or 1)
                        
                        # Normalize weights to sum to 5
                        total_weight = new_listings_weight + new_fdv_weight + new_ownership_weight
                        new_listings_weight = (new_listings_weight / total_weight) * 5
                        new_fdv_weight = (new_fdv_weight / total_weight) * 5
                        new_ownership_weight = (new_ownership_weight / total_weight) * 5
                        
                        print(f"\nNormalized weights: Listings={new_listings_weight:.2f}, FDV={new_fdv_weight:.2f}, Ownership={new_ownership_weight:.2f}")
                        
                        # Get sort order
                        while True:
                            sort_order = input("\nSort by (a)scending or (d)escending deviation? (a/d): ").lower()
                            if sort_order in ['a', 'd']:
                                break
                            print("Invalid choice. Please enter 'a' for ascending or 'd' for descending.")
                        
                        # Recalculate combined deviations with new weights
                        for result in analysis_results:
                            result['combined_deviation'] = (
                                (new_listings_weight * result['listings_deviation']) + 
                                (new_fdv_weight * result['fdv_deviation']) +
                                (new_ownership_weight * result['ownership_deviation'])
                            ) / 5
                        
                        # Resort results
                        analysis_results.sort(
                            key=lambda x: x['combined_deviation'],
                            reverse=(sort_order == 'd')
                        )
                        
                        # Print resorted results
                        regression_analyzer._print_deviation_results(analysis_results)
                        
                    except ValueError:
                        print("Invalid weight values. Please try again.")
            else:
                print("No valid data found in selected datasets.")
        elif choice == "11":
            available_datasets = solana_db.list_datasets()
            if not available_datasets:
                print("No datasets available to duplicate.")
                continue
            
            print("\nAvailable datasets:")
            for dataset in available_datasets:
                print(f"- {dataset}")
            
            source_name = input("\nEnter the name of the dataset to duplicate: ").strip()
            if source_name not in available_datasets:
                print(f"Dataset '{source_name}' not found.")
                continue
                
            target_name = input("Enter the name for the new dataset: ").strip()
            solana_db.duplicate_dataset(source_name, target_name)
        elif choice == "12":
            # Get all datasets and filter for memecoin ones
            available_datasets = solana_db.list_memecoin_datasets()
            
            if not available_datasets:
                print("No memecoin datasets available.")
                continue
            
            print("\nAvailable memecoin datasets:")
            for dataset in available_datasets:
                print(f"- {dataset}")
            
            dataset_name = input("\nEnter the memecoin dataset name to analyze (or 'cancel' to go back): ").strip()
            if dataset_name.lower() == 'cancel' or dataset_name not in available_datasets:
                if dataset_name not in available_datasets:
                    print(f"Dataset '{dataset_name}' not found.")
                continue
            
            # Ask about refreshing the dataset
            refresh_choice = input("\nWould you like to refresh the dataset before analysis? (y/n): ").lower()
            if refresh_choice == 'y':
                print(f"\nRefreshing dataset '{dataset_name}'...")
                solana_db.refresh_dataset(dataset_name, solana_api)
            
            # Perform the analysis
            dataset = solana_db.get_dataset(dataset_name)
            if dataset:
                results = regression_analyzer.analyze_memecoin_dataset(dataset)
                if results:
                    print("\n=== Memecoin Analysis Results ===")
                    print(f"Average FDV Ratio: {results['metrics']['avg_fdv_ratio']:.2f}")
                    print(f"Median FDV Ratio: {results['metrics']['median_fdv_ratio']:.2f}")
                    print(f"Average Token Holder Value: ${results['metrics']['avg_token_holder_value']:,.2f}")
                    print(f"Median Token Holder Value: ${results['metrics']['median_token_holder_value']:,.2f}")
                    
                    # Plot regression results
                    regression_analyzer.plot_results(results)
            else:
                print("No dataset found or dataset is not a memecoin dataset.")
        elif choice == "13":
            print("Exiting Solana NFT Analysis Tool.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
