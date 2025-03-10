# main.py
import sys
sys.path.append("C:/Users/Kyle/crypto-analysis-tool")
from data.data_manager import DataSetManager
from data.database import Database
from config import DEXTOOLS_API_KEY, CMC_API_KEY
from analysis.custom_regression import CustomRegressionAnalyzer
from data.api.coinmarketcap_api import CoinMarketCapAPI


# Add this to the top of main.py, right after the imports
print("Running from:", __file__)

def main():
    # Initialize managers with the API key
    dataset_manager = DataSetManager(api_key=DEXTOOLS_API_KEY)
    database = Database()
    custom_analyzer = CustomRegressionAnalyzer()
    cmc_api = CoinMarketCapAPI(api_key=CMC_API_KEY)

    while True:
        print("\nOptions:")
        print("1. Create a new dataset")
        print("2. Delete a dataset")
        print("3. Add a project to a dataset")
        print("4. View a dataset")
        print("5. Analyze FDV Relationship")
        print("6. Delete a project from dataset")
        print("7. Copy dataset with FDV filter")
        print("8. Add custom regression equation")
        print("9. List custom regression equations")
        print("10. Analyze with Custom Regression")
        print("11. Refresh a dataset")
        print("12. Create dataset from CoinMarketCap category")

        print("21. Duplicate a dataset")  # New option for duplicating a dataset
        print("22. Exit")

        print("24. Analyze Dataset By Volume Data")  # New option
        print("25. Refresh Dataset - Remove Duplicates")  # New option
        print("27. Analyze Dataset Float Metrics")  # Existing option
        print("28. Create Memecoin Dataset")  # New option

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            dataset_name = input("Enter the name for the new dataset: ").strip()
            dataset_manager.create_dataset(dataset_name)
        elif choice == "2":
            dataset_name = input("Enter the name of the dataset to delete: ").strip()
            dataset_manager.delete_dataset(dataset_name)
        elif choice == "3":
            specify_chain = input("Would you like to enter tokens from a specific chain? (y/n): ").strip().lower()
            if specify_chain == 'y':
                chain_id = input("Enter the chain ID: ").strip()
            else:
                chain_id = input("Enter the chain ID (default): ").strip()

            dataset_name = input("Enter the dataset name: ").strip()
            while True:
                token_address = input("Enter the token address (enter 'qpb' to return home): ").strip()
                
                if token_address.lower() == 'qpb':
                    break
                
                dataset_manager.add_project_to_dataset(dataset_name, chain_id, token_address)
                print(f"Added project to dataset '{dataset_name}'.")
        elif choice == "4":
            dataset_manager.list_datasets()
            dataset_name = input("Enter the name of the dataset to view: ").strip()
            dataset_manager.view_dataset(dataset_name)
        elif choice == "5":
            dataset_name = input("Enter the name of the dataset to analyze: ").strip()
            dataset_manager.analyze_fdv_relationship(dataset_name)
        elif choice == "6":
            dataset_name = input("Enter the name of the dataset: ").strip()
            token_address = input("Enter the token address to delete: ").strip()
            dataset_manager.delete_project(dataset_name, token_address)
        elif choice == "7":
            source_dataset_name = input("Enter the source dataset name: ").strip()
            new_dataset_name = input("Enter the new dataset name: ").strip()
            min_fdv = float(input("Enter the minimum FDV: ").strip())
            max_fdv = float(input("Enter the maximum FDV: ").strip())
            dataset_manager.copy_dataset_with_fdv_filter(source_dataset_name, new_dataset_name, min_fdv, max_fdv)
        elif choice == "8":
            custom_analyzer.add_custom_equation()
        elif choice == "9":
            custom_analyzer.list_custom_equations()
        elif choice == "10":
            dataset_name = input("Enter the name of the dataset to analyze with custom regression: ").strip()
            dataset_manager.analyze_with_custom_regression(dataset_name, custom_analyzer)
        elif choice == "11":
            dataset_name = input("Enter the name of the dataset to refresh: ").strip()
            dataset_manager.refresh_dataset(dataset_name)
        elif choice == "12":
            categories = cmc_api.fetch_categories()
            if categories:
                print("\nAvailable Categories:")
                for category in categories:
                    print(f"ID: {category['id']}, Name: {category['name']}")
                category_id = input("Enter the category ID to create a dataset from: ").strip()
                dataset_manager.create_dataset_from_category(category_id, cmc_api)
       
        elif choice == "21":  # New option for duplicating a dataset
            original_dataset_name = input("Enter the name of the dataset to duplicate: ").strip()
            new_dataset_name = input("Enter the new name for the duplicated dataset: ").strip()
            if database.duplicate_dataset(original_dataset_name, new_dataset_name):
                print(f"Dataset '{original_dataset_name}' duplicated as '{new_dataset_name}'.")
            else:
                print("Failed to duplicate dataset.")
        elif choice == "22":
            print("Exiting.")
            break
        elif choice == "27":
            dataset_name = input("Enter the name of the dataset to analyze float metrics: ").strip()
            dataset_manager.analyze_float_metrics(dataset_name)
        elif choice == "28":  # New option for creating memecoin dataset
            print("\nCreate Memecoin Dataset")
            print("-" * 20)
            # List available datasets
            dataset_manager.list_datasets()
            original_dataset_name = input("\nEnter the name of the dataset to convert to memecoin dataset: ").strip()
            new_dataset_name = input("Enter the name for the new memecoin dataset: ").strip()
            dataset_manager.create_memecoin_dataset(original_dataset_name, new_dataset_name)
        else:
            print("Invalid choice. Please try again.")
    
    
if __name__ == "__main__":    
    main()