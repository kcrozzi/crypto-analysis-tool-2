import sqlite3
from nft_project import Project
import json
from pathlib import Path
import time
# nft_data/nft_database.py

# Add to the top of the existing file, after other imports
from nft_nft.ethereum_api import EthereumAPI

class SolanaDatabase:
    def __init__(self):
        Path("solana_data").mkdir(exist_ok=True)
        self.db_path = "solana_data/solana_nft_analysis.db"
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update datasets table to include dataset_type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                dataset_type TEXT NOT NULL DEFAULT 'solana',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_memecoin BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        
        # Add dataset_type column if it doesn't exist
        cursor.execute("PRAGMA table_info(datasets)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'dataset_type' not in columns:
            cursor.execute("ALTER TABLE datasets ADD COLUMN dataset_type TEXT NOT NULL DEFAULT 'solana'")
        
        # Create projects table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                holder_stats JSON,
                collection_stats JSON,
                memecoin_stats JSON,
                chain TEXT,
                token_address TEXT,
                nft_fdv REAL,
                floorPrice REAL
            )
        """)
        
        conn.commit()
        conn.close()

    def create_dataset(self, name, dataset_type='solana', is_memecoin=False):
        if dataset_type not in ['solana', 'ethereum']:
            raise ValueError("dataset_type must be either 'solana' or 'ethereum'")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            print(f"[DEBUG] Executing SQL to create dataset: name={name}, dataset_type={dataset_type}, is_memecoin={is_memecoin}")
            cursor.execute(
                "INSERT INTO datasets (name, dataset_type, is_memecoin) VALUES (?, ?, ?)",
                (name, dataset_type, is_memecoin)
            )
            conn.commit()
            print(f"{dataset_type.capitalize()} dataset '{name}' created.")
        except sqlite3.IntegrityError as e:
            print(f"[DEBUG] IntegrityError: {e}")
            print(f"Dataset '{name}' already exists.")
        except Exception as e:
            print(f"[DEBUG] Exception occurred: {e}")
        finally:
            # Check if the dataset was actually created
            cursor.execute("SELECT * FROM datasets WHERE name = ? AND dataset_type = ?", (name, dataset_type))
            dataset = cursor.fetchone()
            if dataset:
                print(f"[DEBUG] Dataset '{name}' confirmed in database.")
            else:
                print(f"[DEBUG] Failed to confirm dataset '{name}' in database.")
            conn.close()

    def create_project(self, dataset_id, symbol, holder_stats, collection_stats, nft_fdv, floor_price):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO projects (dataset_id, symbol, holder_stats, collection_stats, nft_fdv, floorPrice) VALUES (?, ?, ?, ?, ?, ?)",
                (dataset_id, symbol, json.dumps(holder_stats), json.dumps(collection_stats), nft_fdv, floor_price)
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Error: {e}")
        finally:
            conn.close()

    def get_dataset_type(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT dataset_type FROM datasets WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def add_projects(self, dataset_name, symbols, api):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, dataset_type FROM datasets WHERE name = ?", (dataset_name,))
        dataset = cursor.fetchone()
        
        if not dataset:
            print(f"Dataset '{dataset_name}' not found.")
            conn.close()
            return
            
        dataset_id, dataset_type = dataset
        
        for symbol in symbols:
            # Check if project already exists
            cursor.execute("""
                SELECT symbol FROM projects 
                WHERE dataset_id = ? AND symbol = ?
            """, (dataset_id, symbol))
            
            if cursor.fetchone():
                print(f"Project '{symbol}' already exists in dataset '{dataset_name}'. Skipping...")
                continue
            
            print(f"\nFetching project data for {symbol}...")
            
            try:
                project_data = api.fetch_complete_project_data(symbol)
                
                if project_data:
                    holder_stats = json.dumps(project_data['holder_stats'])
                    collection_stats = json.dumps(project_data['collection_stats'])
                    nft_fdv = project_data.get('nft_fdv', 0)  # Default to 0 if not found
                    floor_price = project_data.get('floorPrice', 0)  # Default to 0 if not found
                    
                    cursor.execute("""
                        INSERT INTO projects (
                            dataset_id, symbol, holder_stats, collection_stats, nft_fdv, floorPrice
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        dataset_id, symbol, 
                        holder_stats,
                        collection_stats,
                        nft_fdv,
                        floor_price
                    ))
                    conn.commit()
                    print(f"\nProject '{symbol}' added to dataset '{dataset_name}'.")
                else:
                    print(f"Failed to fetch stats for project '{symbol}'.")
                
            except Exception as e:
                print(f"Error processing project {symbol}: {str(e)}")
                continue
        
        conn.close()

    def list_datasets(self):
        """List all available datasets."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM datasets")
        datasets = cursor.fetchall()
        conn.close()
        return [dataset[0] for dataset in datasets]

    def view_dataset(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(""" 
            SELECT p.symbol, p.holder_stats, p.collection_stats, p.memecoin_stats, p.chain, p.token_address, p.nft_fdv, p.floorPrice, d.dataset_type 
            FROM projects p 
            JOIN datasets d ON p.dataset_id = d.id 
            WHERE d.name = ? 
        """, (name,))
        projects = cursor.fetchall()
        conn.close()

        dataset = []
        for project in projects:
            try:
                holder_stats = json.loads(project[1])
                collection_stats = json.loads(project[2])
                dataset_type = project[8]
                total_supply = holder_stats.get('totalSupply', 0)
                floor_price = collection_stats.get('floorPrice', 0)
                
                # Convert nft_fdv to float to avoid type issues
                nft_fdv = float(project[6])  # Fetching nft_fdv from the project data
                nft_fdv_calculated = floor_price * total_supply if total_supply > 0 else 0

                # Debugging output to check fetched values
                print(f"[DEBUG] Fetched project: {project[0]}, nft_fdv: {nft_fdv}, calculated nft_fdv: {nft_fdv_calculated}")  # Debugging line

                project_data = {
                    "symbol": project[0],
                    "nft_fdv": nft_fdv if nft_fdv > 0 else nft_fdv_calculated,  # Use calculated if not stored
                    "holder_stats": holder_stats,
                    "collection_stats": collection_stats,
                    "dataset_type": dataset_type,
                    "chain": project[4],
                    "token_address": project[5]
                }
                dataset.append(project_data)
            except Exception as e:
                print(f"[ERROR] Failed to process project data: {e}")
        
        return dataset

    def get_dataset(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.symbol, p.holder_stats, p.collection_stats, d.dataset_type
            FROM projects p
            JOIN datasets d ON p.dataset_id = d.id
            WHERE d.name = ?
        """, (name,))
        projects = cursor.fetchall()
        conn.close()
        
        dataset = []
        for p in projects:
            try:
                symbol, holder_stats_json, collection_stats_json, dataset_type = p
                holder_stats = json.loads(holder_stats_json) if holder_stats_json else {}
                collection_stats = json.loads(collection_stats_json) if collection_stats_json else {}
                
                # Skip projects with missing required data
                if not holder_stats or not collection_stats:
                    print(f"Warning: Skipping project {symbol} due to missing stats data")
                    continue
                
                # Get required values with defaults
                floor_price = collection_stats.get('floorPrice', 0)
                total_supply = holder_stats.get('totalSupply', 0)
                
                # Convert Solana floor price from lamports if needed
                if dataset_type == 'solana' and floor_price > 1000000:  # Likely in lamports
                    floor_price = floor_price
                      # Convert lamports to SOL
                
                # Skip if essential values are missing or zero
                if floor_price <= 0 or total_supply <= 0:
                    print(f"Warning: Skipping project {symbol} due to invalid floor price or supply")
                    continue
                
                project_data = {
                    "symbol": symbol,
                    "holder_stats": holder_stats,
                    "collection_stats": collection_stats,
                    "floor_price": floor_price,  # Store normalized floor price
                    "total_supply": total_supply,
                    "dataset_type": dataset_type,
                    "fdv": floor_price * total_supply  # Calculate FDV without division
                }
                
                dataset.append(project_data)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON data for project {p[0]}: {str(e)}")
                continue
            except Exception as e:
                print(f"Warning: Error processing project {p[0]}: {str(e)}")
                continue
        
        if not dataset:
            print(f"No valid projects found in dataset '{name}'")
            return None
        
        return dataset

    def refresh_dataset(self, name, api):
        print(f"[DEBUG] Starting refresh for dataset: {name}")  # Debugging line
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                SELECT p.id, p.symbol FROM projects p 
                JOIN datasets d ON p.dataset_id = d.id 
                WHERE d.name = ? 
            """, (name,))
            projects = cursor.fetchall()
            
            # Determine the chain type based on the dataset name
            dataset_type = self.get_dataset_type(name)
            print(f"[DEBUG] Dataset type: {dataset_type}")  # Debugging line
            chain = 'solana' if dataset_type == 'solana' else 'ethereum'
            
            for project in projects:
                project_id, symbol = project
                print(f"[DEBUG] Processing project: {symbol}, ID: {project_id}")  # Debugging line
                
                # Fetch complete project data which includes proper floor price conversion
                try:
                    # Adjusted method call to match the correct number of arguments
                    project_data = api.fetch_complete_project_data(symbol)  # Removed chain argument
                    print(f"[DEBUG] Project data fetched for {symbol}: {project_data}")  # Debugging line
                    
                    # Use .get() to safely access the values
                    holder_stats = json.dumps(project_data.get('holder_stats', {}))
                    collection_stats = project_data.get('collection_stats', {})
                    
                    # Calculate the adjusted floor price
                    floor_price = collection_stats.get('floorPrice', 0)  # Get the original floor price
                    if chain == 'solana':
                        floor_price /= 1e9  # Adjust for Solana
                    
                    # Retrieve total supply from holder stats
                    total_supply = project_data.get('holder_stats', {}).get('totalSupply', 0)
                    
                    # Calculate NFT FDV correctly
                    nft_fdv = floor_price * total_supply  # Calculate NFT FDV
                    
                    # Update the collection_stats with the adjusted floor price
                    collection_stats['floorPrice'] = floor_price  # Store the adjusted floor price
                    
                    # Debugging output before the update
                    print(f"[DEBUG] Updating project ID: {project_id}, nft_fdv: {nft_fdv}, floor_price: {floor_price}")  # Debugging line
                    
                    # Update the database with the correct values
                    cursor.execute(""" 
                        UPDATE projects 
                        SET holder_stats = ?, collection_stats = ?, nft_fdv = ?, floorPrice = ? 
                        WHERE id = ? 
                    """, (holder_stats, json.dumps(collection_stats), nft_fdv, floor_price, project_id))  # Update statement
                    
                    print(f"[DEBUG] Successfully updated project: {symbol}")  # Debugging line
                except Exception as e:
                    print(f"[ERROR] Failed to fetch or update project {symbol}: {e}")  # Error handling
            
            conn.commit()
            print(f"Dataset '{name}' refreshed.")

    def delete_project(self, dataset_name, symbols):
        """Delete multiple projects from a dataset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get dataset ID
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        dataset = cursor.fetchone()
        
        if not dataset:
            print(f"Dataset '{dataset_name}' not found.")
            conn.close()
            return
        
        dataset_id = dataset[0]
        deleted_count = 0
        not_found_count = 0
        
        # Delete each project
        for symbol in symbols:
            cursor.execute(
                "DELETE FROM projects WHERE dataset_id = ? AND symbol = ?", 
                (dataset_id, symbol)
            )
            if cursor.rowcount > 0:
                deleted_count += 1
            else:
                not_found_count += 1
                print(f"Project '{symbol}' not found in dataset '{dataset_name}'.")
        
        conn.commit()
        
        if deleted_count > 0:
            print(f"Successfully deleted {deleted_count} project(s) from dataset '{dataset_name}'.")
        if not_found_count > 0:
            print(f"{not_found_count} project(s) were not found in the dataset.")
        
        conn.close()

    def fetch_all_collections(self, api):
        """Fetch all collections from Magic Eden and store them in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create the collections table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS me_collections (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                image TEXT,
                twitter TEXT,
                discord TEXT,
                website TEXT,
                categories TEXT,
                is_badged BOOLEAN,
                has_cnfts BOOLEAN,
                is_ocp BOOLEAN,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        offset = 0
        limit = 500
        total_collections = 0
        
        print("\nFetching collections from Magic Eden...")
        print("This may take several minutes...")
        
        while True:
            collections = api.fetch_collections(offset=offset, limit=limit)
            if not collections:
                break
            
            for collection in collections:
                cursor.execute("""
                    INSERT OR REPLACE INTO me_collections (
                        symbol, name, description, image, twitter, discord, 
                        website, categories, is_badged, has_cnfts, is_ocp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    collection.get('symbol'),
                    collection.get('name'),
                    collection.get('description'),
                    collection.get('image'),
                    collection.get('twitter'),
                    collection.get('discord'),
                    collection.get('website'),
                    json.dumps(collection.get('categories', [])),
                    collection.get('isBadged', False),
                    collection.get('hasCNFTs', False),
                    collection.get('isOcp', False)
                ))
            
            total_collections += len(collections)
            print(f"Processed {total_collections} collections...")
            
            if len(collections) < limit:
                break
            
            offset += limit
            conn.commit()  # Commit after each batch
        
        conn.commit()
        conn.close()
        print(f"\nFinished! Total collections fetched: {total_collections}")

    
    def add_project_direct(self, dataset_name: str, project: Project) -> None:
        """
        Add a project directly to a dataset without making API calls
        """
        if not self.dataset_exists(dataset_name):
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
            
        # Add project to the dataset
        self.datasets[dataset_name][project.symbol] = project
        
        # Save the updated database
        self.save() 

    def duplicate_dataset(self, source_name, target_name):
        """
        Create a copy of an existing dataset with a new name.
        
        Args:
            source_name (str): Name of the dataset to copy from
            target_name (str): Name of the new dataset
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First check if source dataset exists
            cursor.execute("SELECT id, is_memecoin FROM datasets WHERE name = ?", (source_name,))
            source_dataset = cursor.fetchone()
            
            if not source_dataset:
                print(f"Source dataset '{source_name}' not found.")
                return False
            
            source_id, is_memecoin = source_dataset
            
            # Check if target name already exists
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (target_name,))
            if cursor.fetchone():
                print(f"Target dataset name '{target_name}' already exists.")
                return False
            
            # Create new dataset
            cursor.execute(
                "INSERT INTO datasets (name, is_memecoin) VALUES (?, ?)",
                (target_name, is_memecoin)
            )
            new_dataset_id = cursor.lastrowid
            
            # Copy all projects from source to target
            cursor.execute("""
                INSERT INTO projects (
                    dataset_id, symbol, holder_stats, collection_stats, 
                    memecoin_stats, chain, token_address, nft_fdv, floorPrice
                )
                SELECT 
                    ?, symbol, holder_stats, collection_stats, 
                    memecoin_stats, chain, token_address, nft_fdv, floorPrice
                FROM projects 
                WHERE dataset_id = ?
            """, (new_dataset_id, source_id))
            
            conn.commit()
            print(f"Dataset '{source_name}' successfully duplicated to '{target_name}'.")
            return True
            
        except Exception as e:
            print(f"Error duplicating dataset: {str(e)}")
            conn.rollback()
            return False
            
        finally:
            conn.close()

    def list_memecoin_datasets(self):
        """List all available memecoin datasets."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM datasets WHERE is_memecoin = 1")
        datasets = cursor.fetchall()
        conn.close()
        return [dataset[0] for dataset in datasets]

    def check_dataset_exists(self, name, dataset_type):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM datasets WHERE name = ? AND dataset_type = ?", (name, dataset_type))
        dataset = cursor.fetchone()
        conn.close()
        return dataset is not None
