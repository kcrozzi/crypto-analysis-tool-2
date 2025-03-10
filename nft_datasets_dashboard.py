from dash import dcc, html, Input, Output, callback_context, State
from nft_data.nft_database import SolanaDatabase
from nft_analysis.solana_regression import SolanaRegressionAnalyzer
from nft_nft.nft_api import SolanaAPI
from nft_nft.ethereum_api import EthereumAPI
import json

# Initialize necessary components
solana_db = SolanaDatabase()
solana_api = SolanaAPI()
ethereum_api = EthereumAPI()
regression_analyzer = SolanaRegressionAnalyzer()

# Define NFT Datasets page layout
nft_datasets_layout = html.Div([
    dcc.Link('Home', href='/', style={'color': '#00FF00', 'font-family': 'Courier New'}),
    html.H2("NFT Datasets"),
    html.Button("Create NFT Dataset", id='create-nft-dataset', n_clicks=0),
    html.Button("Add Project to Dataset", id='add-project-to-dataset', n_clicks=0),
    html.Button("View Dataset", id='view-nft-dataset', n_clicks=0),
    html.Button("Refresh Dataset", id='refresh-nft-dataset', n_clicks=0),
    dcc.Input(id='refresh-dataset-name', type='text', placeholder='Enter dataset name to refresh', style={'display': 'none'}),
    html.Button('Submit Dataset Name for Refresh', id='submit-nft-dataset-refresh', n_clicks=0, style={'display': 'none'}),
    html.Button("Duplicate Dataset", id='duplicate-nft-dataset', n_clicks=0),
    html.Button('Submit Project to Dataset', id='submit-project-to-dataset', n_clicks=0, style={'display': 'none'}),
    dcc.Input(id='nft-dataset-name', type='text', placeholder='Enter dataset name', style={'display': 'none'}),
    dcc.Dropdown(
        id='nft-dataset-type',
        options=[
            {'label': 'Solana', 'value': 'solana'},
            {'label': 'Ethereum', 'value': 'ethereum'}
        ],
        placeholder='Select dataset type',
        style={'display': 'none'}
    ),
    dcc.Input(id='nft-project-identifier', type='text', placeholder='Enter project symbol or address', style={'display': 'none'}),
    dcc.Input(id='nft-new-dataset-name', type='text', placeholder='Enter new dataset name for duplication', style={'display': 'none'}),
    html.Button('Submit Dataset Name for Creation', id='submit-nft-dataset-create', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Dataset Name for Viewing', id='submit-nft-dataset-view', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit New Dataset Name for Duplication', id='submit-nft-dataset-duplicate', n_clicks=0, style={'display': 'none'}),
    html.Button("Delete Project from NFT Dataset", id='nft-delete-project-from-dataset', n_clicks=0),
    dcc.Input(id='nft-delete-project-identifier', type='text', placeholder='Enter project symbol or address to delete', style={'display': 'none'}),
    html.Button('Submit NFT Project Deletion', id='nft-submit-project-deletion', n_clicks=0, style={'display': 'none'}),
    html.Div(id='nft-datasets-output')
])

# Define callbacks for NFT functionalities
def register_nft_callbacks(app):
    print("[DEBUG] Registering NFT callbacks")
    @app.callback(
        [Output('nft-datasets-output', 'children'),
         Output('nft-dataset-name', 'style'),
         Output('nft-dataset-type', 'style'),
         Output('nft-project-identifier', 'style'),
         Output('nft-new-dataset-name', 'style'),
         Output('submit-nft-dataset-create', 'style'),
         Output('submit-project-to-dataset', 'style'),
         Output('submit-nft-dataset-view', 'style'),
         Output('refresh-dataset-name', 'style'),
         Output('submit-nft-dataset-refresh', 'style'),
         Output('submit-nft-dataset-duplicate', 'style'),
         Output('nft-delete-project-identifier', 'style'),
         Output('nft-submit-project-deletion', 'style')],
        [Input('create-nft-dataset', 'n_clicks'),
         Input('add-project-to-dataset', 'n_clicks'),
         Input('view-nft-dataset', 'n_clicks'),
         Input('refresh-nft-dataset', 'n_clicks'),
         Input('duplicate-nft-dataset', 'n_clicks'),
         Input('submit-nft-dataset-create', 'n_clicks'),
         Input('submit-project-to-dataset', 'n_clicks'),
         Input('submit-nft-dataset-view', 'n_clicks'),
         Input('submit-nft-dataset-refresh', 'n_clicks'),
         Input('submit-nft-dataset-duplicate', 'n_clicks'),
         Input('nft-delete-project-from-dataset', 'n_clicks'),
         Input('nft-submit-project-deletion', 'n_clicks')],
        [State('nft-dataset-name', 'value'),
         State('nft-dataset-type', 'value'),
         State('nft-project-identifier', 'value'),
         State('nft-new-dataset-name', 'value'),
         State('refresh-dataset-name', 'value'),
         State('nft-delete-project-identifier', 'value')]
    )
    def handle_nft_operations_and_visibility(n_clicks_create, n_clicks_add, n_clicks_view, n_clicks_refresh, n_clicks_duplicate, n_clicks_submit_create, n_clicks_submit_project, n_clicks_submit_view, n_clicks_submit_refresh, n_clicks_submit_duplicate, n_clicks_delete_project, n_clicks_submit_deletion, dataset_name, dataset_type, project_identifier, new_dataset_name, refresh_dataset_name, delete_project_identifier):
        ctx = callback_context
        if not ctx.triggered:
            return "Select an action to perform.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"[DEBUG] Button clicked: {button_id}")

        if button_id == 'create-nft-dataset':
            return "Please enter a dataset name and select a type.", {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'submit-nft-dataset-create':
            if not dataset_name or not dataset_type:
                return "Please enter a dataset name and select a type.", {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            if dataset_type == 'solana':
                solana_db.create_dataset(dataset_name, dataset_type='solana')
            elif dataset_type == 'ethereum':
                solana_db.create_dataset(dataset_name, dataset_type='ethereum')
            dataset_exists = solana_db.check_dataset_exists(dataset_name, dataset_type)
            if dataset_exists:
                return f"{dataset_type.capitalize()} NFT dataset '{dataset_name}' created successfully.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            else:
                return f"Failed to create {dataset_type} NFT dataset '{dataset_name}'.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'add-project-to-dataset':
            return "Please select dataset type and enter project identifier.", {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'submit-project-to-dataset' and dataset_name and dataset_type and project_identifier:
            if dataset_type == 'ethereum':
                solana_db.add_projects(dataset_name, [project_identifier], ethereum_api)
            else:
                solana_db.add_projects(dataset_name, [project_identifier], solana_api)
            return f"Project '{project_identifier}' added to dataset '{dataset_name}'.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'view-nft-dataset':
            return "Please enter the dataset name and select a type to view.", {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'submit-nft-dataset-view' and dataset_name and dataset_type:
            try:
                dataset_content = solana_db.view_dataset(dataset_name)
                return html.Pre(json.dumps(dataset_content, indent=2)), {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            except Exception as e:
                return f"Error viewing dataset: {str(e)}", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'refresh-nft-dataset':
            return "Please enter the dataset name to refresh.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'submit-nft-dataset-refresh' and refresh_dataset_name:
            dataset_type = solana_db.get_dataset_type(refresh_dataset_name)
            if dataset_type == 'ethereum':
                # Log the request details for Ethereum API
                solana_db.refresh_dataset(refresh_dataset_name, ethereum_api)
            else:
                solana_db.refresh_dataset(refresh_dataset_name, solana_api)
            return f"Dataset '{refresh_dataset_name}' refreshed.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'duplicate-nft-dataset':
            return "Please enter the dataset name to duplicate and the new dataset name.", {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'submit-nft-dataset-duplicate' and dataset_name and new_dataset_name:
            solana_db.duplicate_dataset(dataset_name, new_dataset_name)
            return f"Dataset '{dataset_name}' duplicated to '{new_dataset_name}'.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        elif button_id == 'nft-delete-project-from-dataset':
            return "Please enter the dataset name and project identifier to delete.", {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
        elif button_id == 'nft-submit-project-deletion' and dataset_name and delete_project_identifier:
            solana_db.delete_project(dataset_name, [delete_project_identifier])
            return f"Project '{delete_project_identifier}' deleted from dataset '{dataset_name}'.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        
        return "Action not recognized or missing required inputs.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Add more callbacks as needed for other functionalities
