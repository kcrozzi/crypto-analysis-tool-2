import sqlite3
import json
from pathlib import Path

class Database:
    def __init__(self):
        # Create a 'data' directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        self.db_path = "data/crypto_analysis.db"
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                chain_id TEXT NOT NULL,
                token_address TEXT NOT NULL,
                data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """)

        # Create memecoin_datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memecoin_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create memecoin_projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memecoin_projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                chain_id TEXT NOT NULL,
                token_address TEXT NOT NULL,
                data JSON NOT NULL,
                humor_grade TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES memecoin_datasets (id)
            )
        """)

        conn.commit()
        conn.close()

    def dataset_exists(self, dataset_name):
        """Check if a dataset with the given name exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM datasets WHERE name = ?", (dataset_name,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def duplicate_dataset(self, original_dataset_name, new_dataset_name):
        """Duplicate an existing dataset under a new name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if the original dataset exists
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (original_dataset_name,))
            original_dataset = cursor.fetchone()
            
            if not original_dataset:
                print(f"Original dataset '{original_dataset_name}' not found.")
                return False
            
            # Create the new dataset
            cursor.execute("INSERT INTO datasets (name) VALUES (?)", (new_dataset_name,))
            conn.commit()
            new_dataset_id = cursor.lastrowid
            
            # Copy projects from the original dataset to the new dataset
            cursor.execute("SELECT chain_id, token_address, data FROM projects WHERE dataset_id = ?", (original_dataset[0],))
            projects = cursor.fetchall()
            
            for project in projects:
                cursor.execute(""" 
                    INSERT INTO projects (dataset_id, chain_id, token_address, data)
                    VALUES (?, ?, ?, ?)
                """, (new_dataset_id, project[0], project[1], project[2]))
            
            conn.commit()
            print(f"Dataset '{original_dataset_name}' duplicated as '{new_dataset_name}'.")
            return True
            
        except Exception as e:
            print(f"Error duplicating dataset: {e}")
            return False
        finally:
            conn.close()

    def update_project(self, dataset_name, chain_id, token_address, data):
        """Update a project's data in a dataset."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get dataset id
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
            dataset = cursor.fetchone()
            
            if dataset:
                # Convert data dict to JSON string
                data_json = json.dumps(data)
                cursor.execute("""
                    UPDATE projects
                    SET data = ?
                    WHERE dataset_id = ? AND chain_id = ? AND token_address = ?
                """, (data_json, dataset[0], chain_id, token_address))
                conn.commit()
                conn.close()
                return True
            else:
                print(f"[DEBUG] Dataset '{dataset_name}' not found in database")
                conn.close()
                return False
        except Exception as e:
            print(f"[DEBUG] Database error in update_project: {str(e)}")
            return False

    def create_dataset(self, dataset_name):
        """Create a new dataset."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO datasets (name) VALUES (?)", (dataset_name,))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"Dataset '{dataset_name}' already exists.")
            return False
        finally:
            conn.close()

    def delete_dataset(self, dataset_name):
        """Delete a dataset and all its projects."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get dataset id
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        dataset = cursor.fetchone()
        
        if dataset:
            print(f"[DEBUG] Found dataset '{dataset_name}' with ID {dataset[0]}. Proceeding with deletion.")
            # Delete all projects in the dataset
            cursor.execute("DELETE FROM projects WHERE dataset_id = ?", (dataset[0],))
            # Delete the dataset
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset[0],))
            conn.commit()
            print(f"[DEBUG] Dataset '{dataset_name}' and its projects deleted successfully.")
            conn.close()
            return True
        else:
            print(f"[DEBUG] Dataset '{dataset_name}' not found. No deletion performed.")
            conn.close()
            return False

    def add_project(self, dataset_name, chain_id, token_address, data):
        """Add a project to a dataset."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get dataset id
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
            dataset = cursor.fetchone()
            
            if dataset:
                # Convert data dict to JSON string
                data_json = json.dumps(data)
                cursor.execute("""
                    INSERT INTO projects (dataset_id, chain_id, token_address, data)
                    VALUES (?, ?, ?, ?)
                """, (dataset[0], chain_id, token_address, data_json))
                conn.commit()
                conn.close()
                return True
            else:
                print(f"[BUG] Dataset '{dataset_name}' not found in database")
                conn.close()
                return False
        except Exception as e:
            print(f"[DEBUG] Database error in add_project: {str(e)}")
            return False

    def get_dataset(self, dataset_name):
        """Get all projects in a dataset."""
        print(f"\n[DEBUG DB] Getting dataset '{dataset_name}'")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First check if dataset exists
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
            dataset = cursor.fetchone()
            print(f"[DEBUG DB] Dataset found: {dataset}")
            
            if not dataset:
                print(f"[DEBUG DB] Dataset '{dataset_name}' not found")
                return []
            
            cursor.execute("""
                SELECT p.chain_id, p.token_address, p.data
                FROM projects p
                JOIN datasets d ON p.dataset_id = d.id
                WHERE d.name = ?
            """, (dataset_name,))
            
            projects = cursor.fetchall()
            print(f"[DEBUG DB] Raw database results:", projects)
            
            if projects:
                processed_projects = []
                for p in projects:
                    try:
                        print(f"[DEBUG DB] Processing project data:", p)
                        data = json.loads(p[2])  # p[2] is the JSON data string
                        print(f"[DEBUG DB] Parsed JSON data:", data)
                        processed_project = {
                            'chain_id': p[0],
                            'token_address': p[1],
                            'data': data
                        }
                        print(f"[DEBUG DB] Processed project:", processed_project)
                        processed_projects.append(processed_project)
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG DB] JSON decode error: {e}")
                        print(f"[DEBUG DB] Problematic data: {p[2]}")
            
                return processed_projects
            return []
            
        except Exception as e:
            print(f"[DEBUG DB] Database error: {e}")
            return []
        finally:
            conn.close()

    def list_datasets(self):
        """List all available datasets."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, created_at,
                   (SELECT COUNT(*) FROM projects p WHERE p.dataset_id = d.id) as project_count
            FROM datasets d
            ORDER BY created_at DESC
        """)
        
        datasets = cursor.fetchall()
        conn.close()
        
        return datasets 

    def delete_project(self, dataset_name, token_address):
        """Delete a project from a dataset."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First get the dataset id
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
            dataset = cursor.fetchone()
            
            if not dataset:
                print(f"[DEBUG] Dataset '{dataset_name}' not found.")
                return False
            
            # Delete the project
            cursor.execute("""
                DELETE FROM projects 
                WHERE dataset_id = ? AND token_address = ?
            """, (dataset[0], token_address))
            
            conn.commit()
            deleted = cursor.rowcount > 0  # Returns True if a row was deleted
            if deleted:
                print(f"[DEBUG] Project '{token_address}' deleted from dataset '{dataset_name}'.")
            else:
                print(f"[DEBUG] Project '{token_address}' could not be deleted from dataset '{dataset_name}'.")
            return deleted
            
        except Exception as e:
            print(f"[DEBUG] Error deleting project: {e}")
            return False
        finally:
            conn.close()
            
    def create_memecoin_dataset(self, original_dataset_name, new_dataset_name):
        """Create a memecoin dataset from an existing dataset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if the original dataset exists
            cursor.execute("SELECT id FROM datasets WHERE name = ?", (original_dataset_name,))
            original_dataset = cursor.fetchone()
            
            if not original_dataset:
                print(f"Original dataset '{original_dataset_name}' not found.")
                return False
            
            # Create the new memecoin dataset
            cursor.execute("INSERT INTO memecoin_datasets (name) VALUES (?)", (new_dataset_name,))
            conn.commit()
            new_dataset_id = cursor.lastrowid
            
            # Get all projects from original dataset
            cursor.execute("""
                SELECT chain_id, token_address, data
                FROM projects 
                WHERE dataset_id = ?
            """, (original_dataset[0],))
            projects = cursor.fetchall()
            
            # For each project, prompt for humor grade and add to memecoin dataset
            for project in projects:
                chain_id, token_address, data = project
                project_data = json.loads(data)
                project_name = project_data.get('name', 'Unknown')
                
                # Prompt for humor grade
                while True:
                    grade = input(f"\nRate the humor level for {project_name} (A/B/C/D/F): ").upper()
                    if grade in ['A', 'B', 'C', 'D', 'F']:
                        break
                    print("Invalid grade. Please use A, B, C, D, or F.")
                
                # Insert into memecoin_projects
                cursor.execute("""
                    INSERT INTO memecoin_projects (dataset_id, chain_id, token_address, data, humor_grade)
                    VALUES (?, ?, ?, ?, ?)
                """, (new_dataset_id, chain_id, token_address, data, grade))
            
            conn.commit()
            print(f"\nMemecoin dataset '{new_dataset_name}' created successfully!")
            return True
            
        except Exception as e:
            print(f"Error creating memecoin dataset: {e}")
            return False
        finally:
            conn.close()

    def get_memecoin_dataset(self, dataset_name):
        """Get all projects in a memecoin dataset with their humor grades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM memecoin_datasets WHERE name = ?", (dataset_name,))
            dataset = cursor.fetchone()
            
            if not dataset:
                print(f"Memecoin dataset '{dataset_name}' not found")
                return []
            
            cursor.execute("""
                SELECT p.chain_id, p.token_address, p.data, p.humor_grade
                FROM memecoin_projects p
                WHERE p.dataset_id = ?
            """, (dataset[0],))
            
            projects = cursor.fetchall()
            processed_projects = []
            
            for p in projects:
                try:
                    data = json.loads(p[2])
                    processed_project = {
                        'chain_id': p[0],
                        'token_address': p[1],
                        'data': data,
                        'humor_grade': p[3]
                    }
                    processed_projects.append(processed_project)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    
            return processed_projects
            
        except Exception as e:
            print(f"Database error: {e}")
            return []
        finally:
            conn.close()
