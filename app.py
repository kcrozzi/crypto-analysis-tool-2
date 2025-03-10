from flask import Flask
from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.express as px
from data.data_manager import DataSetManager
from config import DEXTOOLS_API_KEY, CMC_API_KEY
from analysis.custom_regression import CustomRegressionAnalyzer
from analysis.regression import RegressionAnalyzer
from data.api.coinmarketcap_api import CoinMarketCapAPI
from data.database import Database
import numpy as np
import json
from data.data_loader import DataLoader
import dash
from nft_datasets_dashboard import nft_datasets_layout, register_nft_callbacks
from nft_analysis_dashboard import nft_analysis_layout, register_analysis_callbacks
import dash_bootstrap_components as dbc
#from app_instance import app

# Initialize Flask and Dash
server = Flask(__name__)
app = Dash(__name__, server=server, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True  # Suppress callback exceptions



# Initialize managers
dataset_manager = DataSetManager(api_key=DEXTOOLS_API_KEY)
custom_analyzer = CustomRegressionAnalyzer()
cmc_api = CoinMarketCapAPI(api_key=CMC_API_KEY)
database = Database()
regression_analyzer = RegressionAnalyzer()

# Initialize DataLoader with the DEXTools API key
data_loader = DataLoader(api_key=DEXTOOLS_API_KEY)

# Define your regression coefficients and parameters
poly_coeffs = [1, 0.5, -0.01]  # Example coefficients for polynomial regression
a, b = 1, 0.5  # Example parameters for power regression

# Define the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(style={'textAlign': 'center'}, children=[
        dcc.Link('Crypto Analysis', href='/dashboard/analysis', style={'color': '#00FF00', 'font-family': 'Courier New', 'margin': '10px'}),
        dcc.Link('Crypto Datasets', href='/dashboard/datasets', style={'color': '#00FF00', 'font-family': 'Courier New', 'margin': '10px'}),
        dcc.Link('NFT Analysis', href='/nft/analysis', style={'color': '#00FF00', 'font-family': 'Courier New', 'margin': '10px'}),
        dcc.Link('NFT Datasets', href='/nft/datasets', style={'color': '#00FF00', 'font-family': 'Courier New', 'margin': '10px'}),
    ])
])

# Main menu layout
main_menu_layout = html.Div(
    style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'width': '95%'},
    children=[
        html.H1("Crypto Analysis Dashboard"),
    ]
)

# Analysis page layout
analysis_layout = html.Div([
    dcc.Link('Home', href='/', style={'color': '#00FF00', 'font-family': 'Courier New'}),
    html.H2("Analysis", style={'margin': '20px'}),
    dcc.Dropdown(
        id='analysis-dropdown',
        options=[
            {'label': 'Analyze FDV', 'value': 'analyze_fdv'},
            {'label': 'Analyze with Custom Regression', 'value': 'analyze_custom'},
            {'label': 'Analyze Float Metrics', 'value': 'analyze_float'}
        ],
        placeholder="Select an analysis option",
        style={'width': '50%', 'margin': '20px auto'}
    ),
    html.Button('Submit', id='submit-analysis', n_clicks=0, style={'display': 'block', 'margin': '30px auto'}),
    dcc.Input(id='analysis-dataset-name', type='text', placeholder='Enter dataset name', style={'display': 'none'}),
    html.Button('Submit Dataset Name', id='submit-dataset-name', n_clicks=0, style={'display': 'none'}),
    dcc.Graph(
        id='analysis-plot',
        style={'display': 'none', 'margin': '20px auto', 'width': '350px', 'height': '250px'},  # Further reduced size
        config={'displayModeBar': False},
        figure={
            'layout': {
                'autosize': False,
                'margin': {'l': 100, 'r': 40, 't': 40, 'b': 40},
                'plot_bgcolor': '#46515E',
                'paper_bgcolor': '#46515E',
                'font': {'color': '#00FF00'},
                'width': 350,  # Further reduced width
                'height': 250  # Further reduced height
            }
        }
    ),
    dcc.Dropdown(
        id='sort-order-dropdown',
        options=[
            {'label': 'Descending', 'value': 'desc'},
            {'label': 'Ascending', 'value': 'asc'}
        ],
        value='desc',
        style={'width': '50%', 'margin': '20px auto', 'display': 'none'}
    ),
    html.Div(id='analysis-output', style={'margin': '20px'}),
    dcc.Store(id='analysis-results', data={}),
    html.H3("Custom Regressions", style={'margin': '20px'}),
    html.Button('List Custom Regression Equations', id='list-equations', n_clicks=0, style={'margin': '20px'}),
    html.Button('Add Custom Regression Equation', id='add-equation', n_clicks=0, style={'margin': '20px'})
])

# Datasets page layout
datasets_layout = html.Div([
    dcc.Link('Home', href='/', style={'color': '#00FF00', 'font-family': 'Courier New'}),
    html.H2("Datasets"),
    html.Button('Create Dataset', id='create-dataset', n_clicks=0),
    html.Button('View Dataset', id='view-dataset', n_clicks=0),
    html.Button('Add Project to Dataset', id='add-project', n_clicks=0),
    html.Button('Refresh Dataset', id='refresh-dataset', n_clicks=0),
    html.Button('Delete Project from Dataset', id='delete-project', n_clicks=0),
    html.Button('Delete Dataset', id='delete-dataset', n_clicks=0),
    html.Button('Copy Dataset with FDV Filter', id='copy-dataset-fdv', n_clicks=0),
    html.Button('List Datasets', id='list-datasets', n_clicks=0),
    dcc.Input(id='dataset-name', type='text', placeholder='Enter dataset name', style={'display': 'none'}),
    dcc.Input(id='new-dataset-name', type='text', placeholder='Enter new dataset name', style={'display': 'none'}),
    dcc.Input(id='min-fdv', type='number', placeholder='Enter minimum FDV', style={'display': 'none'}),
    dcc.Input(id='max-fdv', type='number', placeholder='Enter maximum FDV', style={'display': 'none'}),
    dcc.Input(id='chain-id', type='text', placeholder='Enter chain ID', style={'display': 'none'}),
    dcc.Input(id='token-address', type='text', placeholder='Enter token address', style={'display': 'none'}),
    html.Button('Submit Project Details', id='submit-project-details', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Dataset Name for View', id='submit-dataset-name-view', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Dataset Name for Refresh', id='submit-dataset-name-refresh', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Project Deletion', id='submit-project-deletion', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Dataset Deletion', id='submit-dataset-deletion', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Dataset Creation', id='submit-dataset-creation', n_clicks=0, style={'display': 'none'}),
    html.Button('Submit Copy with FDV Filter', id='submit-copy-fdv', n_clicks=0, style={'display': 'none'}),
    html.Div(id='datasets-output')
])


# Update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/dashboard/analysis':
        return analysis_layout
    elif pathname == '/dashboard/datasets':
        return datasets_layout
    elif pathname == '/nft/analysis':
        return nft_analysis_layout
    elif pathname == '/nft/datasets':
        return nft_datasets_layout
    else:
        return main_menu_layout

# Redirect buttons to respective pages
@app.callback(
    Output('url', 'pathname'),
    [Input('crypto-analysis-button', 'n_clicks'),
     Input('crypto-datasets-button', 'n_clicks')]
)
def navigate(n_clicks_analysis, n_clicks_datasets):
    ctx = callback_context
    if not ctx.triggered:
        return '/dashboard/'
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'crypto-analysis-button':
        return '/dashboard/analysis'
    elif button_id == 'crypto-datasets-button':
        return '/dashboard/datasets'
    elif button_id == 'nft-analysis-button':
        return '/nft/analysis'
    elif button_id == 'nft-datasets-button':
        return '/nft/datasets'
    return '/dashboard/'

# Show input fields for dataset operations
@app.callback(
    [Output('dataset-name', 'style'),
     Output('new-dataset-name', 'style'),
     Output('min-fdv', 'style'),
     Output('max-fdv', 'style'),
     Output('chain-id', 'style'),
     Output('token-address', 'style'),
     Output('submit-project-details', 'style'),
     Output('submit-dataset-name-view', 'style'),
     Output('submit-dataset-name-refresh', 'style'),
     Output('submit-project-deletion', 'style'),
     Output('submit-dataset-deletion', 'style'),
     Output('submit-dataset-creation', 'style'),
     Output('submit-copy-fdv', 'style')],
    [Input('create-dataset', 'n_clicks'),
     Input('view-dataset', 'n_clicks'),
     Input('add-project', 'n_clicks'),
     Input('refresh-dataset', 'n_clicks'),
     Input('delete-project', 'n_clicks'),
     Input('delete-dataset', 'n_clicks'),
     Input('copy-dataset-fdv', 'n_clicks')]
)
def show_dataset_inputs(n_clicks_create, n_clicks_view, n_clicks_add, n_clicks_refresh, n_clicks_delete, n_clicks_delete_dataset, n_clicks_copy_fdv):
    ctx = callback_context
    if not ctx.triggered:
        return [{'display': 'none'}] * 13

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'create-dataset':
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}]
    elif button_id == 'add-project':
        return [{'display': 'block'},  # dataset-name
                {'display': 'none'},  # new-dataset-name
                {'display': 'none'},  # min-fdv
                {'display': 'none'},  # max-fdv
                {'display': 'block'},  # chain-id
                {'display': 'block'},  # token-address
                {'display': 'block'},  # submit-project-details
                {'display': 'none'},  # submit-dataset-name-view
                {'display': 'none'},  # submit-dataset-name-refresh
                {'display': 'none'},  # submit-project-deletion
                {'display': 'none'},  # submit-dataset-deletion
                {'display': 'none'},  # submit-dataset-creation
                {'display': 'none'}]  # submit-copy-fdv
    elif button_id == 'view-dataset':
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}]
    elif button_id == 'refresh-dataset':
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}]
    elif button_id == 'delete-project':
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}]
    elif button_id == 'delete-dataset':
        return [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}]
    elif button_id == 'copy-dataset-fdv':
        return [{'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}]
    return [{'display': 'none'}] * 13

# Handle dataset operations and adding a project
@app.callback(
    Output('datasets-output', 'children'),
    [Input('list-datasets', 'n_clicks'),
     Input('submit-dataset-name-view', 'n_clicks'),
     Input('submit-project-details', 'n_clicks'),
     Input('submit-dataset-name-refresh', 'n_clicks'),
     Input('submit-project-deletion', 'n_clicks'),
     Input('submit-dataset-deletion', 'n_clicks'),
     Input('submit-dataset-creation', 'n_clicks'),
     Input('submit-copy-fdv', 'n_clicks')],
    [State('dataset-name', 'value'),
     State('new-dataset-name', 'value'),
     State('min-fdv', 'value'),
     State('max-fdv', 'value'),
     State('chain-id', 'value'),
     State('token-address', 'value')]
)
def handle_dataset_operations(n_clicks_list, n_clicks_view, n_clicks_add, n_clicks_refresh, n_clicks_delete, n_clicks_delete_dataset, n_clicks_create, n_clicks_copy_fdv, dataset_name, new_dataset_name, min_fdv, max_fdv, chain_id, token_address):
    ctx = callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"[DEBUG] Button clicked: {button_id}")

    if button_id == 'list-datasets' and n_clicks_list > 0:
        datasets = dataset_manager.db.list_datasets()
        if datasets:
            return html.Div([
                html.Div(f"Name: {name}, Created: {created_at}, Projects: {project_count}") 
                for name, created_at, project_count in datasets
            ])
        else:
            return "No datasets available."
    elif button_id == 'submit-dataset-name-view' and n_clicks_view > 0 and dataset_name:
        # Logic for viewing a dataset
        projects = dataset_manager.db.get_dataset(dataset_name)
        if projects:
            return html.Div([
                html.Pre(json.dumps(projects, indent=2))
            ])
        else:
            return f"No projects found in dataset '{dataset_name}'."
    elif button_id == 'submit-project-details' and n_clicks_add > 0 and dataset_name and chain_id and token_address:
        # Use the DataSetManager to add the project
        try:
            dataset_manager.add_project_to_dataset(dataset_name, chain_id, token_address)
            return f"Project added to dataset '{dataset_name}'."
        except Exception as e:
            return f"Error adding project: {str(e)}"
    elif button_id == 'submit-dataset-name-refresh' and n_clicks_refresh > 0 and dataset_name:
        # Use the DataSetManager to refresh the dataset
        try:
            success = dataset_manager.refresh_dataset(dataset_name)
            if success:
                return f"Dataset '{dataset_name}' refreshed successfully."
            else:
                return f"Failed to refresh dataset '{dataset_name}'."
        except Exception as e:
            return f"Error refreshing dataset: {str(e)}"
    elif button_id == 'submit-project-deletion' and n_clicks_delete > 0 and dataset_name and token_address:
        # Debugging: Print the values being passed
        print(f"[DEBUG] Attempting to delete project with dataset_name: {dataset_name}, token_address: {token_address}")
        
        # Use the DataSetManager to delete the project
        try:
            result = dataset_manager.delete_project(dataset_name, token_address)
            if result:
                print(f"[DEBUG] Project {token_address} successfully deleted from dataset '{dataset_name}'.")
                return f"Project {token_address} deleted from dataset '{dataset_name}'."
            else:
                print(f"[DEBUG] Project {token_address} not found or could not be deleted from dataset '{dataset_name}'.")
                return f"Project not found or could not be deleted from dataset '{dataset_name}'."
        except Exception as e:
            print(f"[DEBUG] Error deleting project: {str(e)}")
            return f"Error deleting project: {str(e)}"
    elif button_id == 'submit-dataset-deletion' and n_clicks_delete_dataset > 0 and dataset_name:
        try:
            if dataset_manager.db.delete_dataset(dataset_name):
                return f"Dataset '{dataset_name}' deleted successfully."
            else:
                return f"Dataset '{dataset_name}' not found or could not be deleted."
        except Exception as e:
            return f"Error deleting dataset: {str(e)}"
    elif button_id == 'submit-dataset-creation' and n_clicks_create > 0 and dataset_name:
        try:
            if dataset_manager.create_dataset(dataset_name):
                return f"Dataset '{dataset_name}' created successfully."
            else:
                return f"Failed to create dataset '{dataset_name}'. It may already exist."
        except Exception as e:
            return f"Error creating dataset: {str(e)}"
    elif button_id == 'submit-copy-fdv' and n_clicks_copy_fdv > 0:
        # Validate FDV inputs
        if dataset_name and new_dataset_name:
            try:
                # Convert FDV inputs to float
                min_fdv = float(min_fdv) if min_fdv is not None else None
                max_fdv = float(max_fdv) if max_fdv is not None else None

                print(f"[DEBUG] FDV values received: min_fdv={min_fdv}, max_fdv={max_fdv}")

                if min_fdv is not None and max_fdv is not None:
                    print(f"[DEBUG] Initiating copy with FDV filter: {min_fdv} to {max_fdv}")
                    dataset_manager.copy_dataset_with_fdv_filter(dataset_name, new_dataset_name, min_fdv, max_fdv)
                    return f"Dataset '{dataset_name}' copied to '{new_dataset_name}' with FDV filter ({min_fdv} - {max_fdv})."
                else:
                    return "Please provide valid FDV values."
            except ValueError:
                return "FDV values must be numbers."
            except Exception as e:
                return f"Error copying dataset with FDV filter: {str(e)}"
        else:
            return "Please provide valid dataset names."
    return ""

# Redirect dropdown selection to respective analysis
@app.callback(
    [Output('analysis-dataset-name', 'style'),
     Output('submit-dataset-name', 'style')],
    [Input('submit-analysis', 'n_clicks')],
    [State('analysis-dropdown', 'value')]
)
def show_analysis_inputs(n_clicks_submit, analysis_option):
    if n_clicks_submit > 0 and analysis_option:
        return [{'display': 'block'}, {'display': 'block'}]
    return [{'display': 'none'}, {'display': 'none'}]

# Handle analysis operations and sorting
@app.callback(
    [Output('analysis-output', 'children'),
     Output('analysis-plot', 'figure'),
     Output('analysis-plot', 'style'),  # Add style output for plot visibility
     Output('sort-order-dropdown', 'style'),
     Output('analysis-results', 'data')],
    [Input('submit-dataset-name', 'n_clicks'),
     Input('sort-order-dropdown', 'value')],
    [State('analysis-dataset-name', 'value'),
     State('analysis-dropdown', 'value'),  # Add analysis type to state
     State('analysis-results', 'data')]
)
def analyze_and_sort(n_clicks_submit, sort_order, dataset_name, analysis_type, stored_results):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'submit-dataset-name' and n_clicks_submit > 0 and dataset_name:
        # Check which analysis type was selected
        if analysis_type == 'analyze_float':
            # Call the float metrics analysis function
            results = dataset_manager.analyze_float_metrics(dataset_name)
            if not results:
                return "No valid float data found for analysis.", {}, {'display': 'none'}, {'display': 'none'}, {}
            
            # Extract plot data for float metrics
            names = [p['symbol'] for p in results]
            float_ratios = [p['float_ratio'] * 100 for p in results]  # Convert to percentage
            locked_ratios = [p['locked_ratio'] * 100 for p in results]  # Convert to percentage
            burned_ratios = [p['burned_ratio'] * 100 for p in results]  # Convert to percentage
            
            # Create a bar chart for float metrics
            fig = {
                'data': [
                    {
                        'x': names,
                        'y': float_ratios,
                        'type': 'bar',
                        'name': 'Circulating Supply %',
                        'marker': {'color': '#00FF00'}
                    },
                    {
                        'x': names,
                        'y': locked_ratios,
                        'type': 'bar',
                        'name': 'Locked Supply %',
                        'marker': {'color': '#0000FF'}
                    },
                    {
                        'x': names,
                        'y': burned_ratios,
                        'type': 'bar',
                        'name': 'Burned Supply %',
                        'marker': {'color': '#FF0000'}
                    }
                ],
                'layout': {
                    'title': 'Token Supply Distribution',
                    'xaxis': {'title': 'Token'},
                    'yaxis': {'title': 'Percentage of Total Supply', 'range': [0, 100]},
                    'barmode': 'group',
                    'plot_bgcolor': '#46515E',
                    'paper_bgcolor': '#46515E',
                    'font': {'color': '#00FF00'},
                    'margin': {'l': 80, 'r': 80, 't': 80, 'b': 80},
                    'autosize': True,
                    'width': None,
                    'height': None
                }
            }
            
            # Store results for sorting
            return "", fig, {'display': 'block'}, {'display': 'block'}, {'data_points': results, 'type': 'float'}
        else:
            # Existing FDV analysis code
            # Fetch the dataset
            projects = dataset_manager.db.get_dataset(dataset_name)
            if not projects:
                return "Dataset not found or empty.", {}, {'display': 'none'}, {'display': 'none'}, {}

            # Perform regression analysis
            results = regression_analyzer.analyze_dataset(projects, analysis_type='power')
            if not results:
                return "No valid data points for analysis.", {}, {'display': 'none'}, {'display': 'none'}, {}

            # Extract plot data
            fdv = [p['fdv'] for p in results['data_points']]
            fdv_per_holder = [p['fdv_per_holder'] for p in results['data_points']]
            names = [p['name'] for p in results['data_points']]

            # Create a plot based on the results
            fig = {
                'data': [
                    {
                        'x': fdv,
                        'y': fdv_per_holder,
                        'text': names,
                        'type': 'scatter',
                        'mode': 'markers+text',
                        'textposition': 'top center',
                        'name': 'Data Points',
                        'marker': {'color': 'white', 'size': 10}
                    },
                    # Add power regression line
                    {
                        'x': fdv,
                        'y': results['power']['expected_values'],
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Power Regression',
                        'line': {'color': 'black', 'width': 2}
                    }
                ],
                'layout': {
                    'title': 'FDV vs FDV per Holder',
                    'xaxis': {'title': 'FDV', 'type': 'log'},
                    'yaxis': {'title': 'FDV per Holder', 'type': 'log'},
                    'plot_bgcolor': '#46515E',  # Updated plot background
                    'paper_bgcolor': '#46515E',  # Updated paper background
                    'font': {'color': '#00FF00'},
                    'margin': {'l': 80, 'r': 200, 't': 80, 'b': 80},
                    'autosize': True,
                    'width': None,
                    'height': None
                }
            }

            # Store results for sorting
            return "", fig, {'display': 'block'}, {'display': 'block'}, results

    elif triggered_id == 'sort-order-dropdown' and stored_results:
        # Check if we're sorting float metrics or FDV analysis
        if stored_results.get('type') == 'float':
            # Sort float metrics
            sort_key = 'float_ratio'  # Default sort by float ratio
            sorted_data_points = sorted(
                stored_results['data_points'],
                key=lambda p: p.get(sort_key, 0),
                reverse=(sort_order == 'desc')
            )
            
            # Create a grid layout for float metrics
            float_grid = html.Div(
                style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'},
                children=[
                    html.Div([
                        html.H4(f"{point['symbol']}"),
                        html.P(f"Float Ratio: {point['float_ratio']*100:.2f}%"),
                        html.P(f"Locked Ratio: {point['locked_ratio']*100:.2f}%"),
                        html.P(f"Burned Ratio: {point['burned_ratio']*100:.2f}%"),
                        html.P(f"Lock Score: {point['lock_score']:.2f}")
                    ]) for point in sorted_data_points
                ]
            )
            
            return float_grid, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            # Sort FDV deviations (existing code)
            sorted_data_points = sorted(
                stored_results['data_points'],
                key=lambda p: p.get('power_deviation', 0),
                reverse=(sort_order == 'desc')
            )

            # Prepare results and deviations text
            power_results = stored_results['power']
            results_text = (
                f"Power Regression Results:\n"
                f"Equation: y = {power_results['coefficients'][0]:.4f}x^{power_results['coefficients'][1]:.4f}\n"
                f"R² Score: {power_results['r2_score']:.4f}\n"
                f"MAE: {power_results['mae']:.4f}\n"
                f"RMSE: {power_results['rmse']:.4f}\n"
                + "-" * 50
            )

            # Create a grid layout for deviations
            deviations_grid = html.Div(
                style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'},
                children=[
                    html.Div([
                        html.H4(f"{point['name']} ({point['symbol']})"),
                        html.P(f"FDV: ${point['fdv']:,.2f}"),
                        html.P(f"Actual FDV per Holder: ${point['fdv_per_holder']:,.2f}"),
                        html.P(f"Expected FDV per Holder: ${point['power_expected']:,.2f}"),
                        html.P(f"Deviation: {point['power_deviation']:.2f}%")
                    ]) for point in sorted_data_points
                ]
            )

            return deviations_grid, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return "Select an operation and provide necessary inputs.", {'data': [], 'layout': {}}, {'display': 'none'}, {'display': 'none'}, {}

def polynomial_regression(x, coeffs):
    """Calculate polynomial regression value."""
    return sum(c * x**i for i, c in enumerate(coeffs))

def power_regression(x, a, b):
    """Calculate power regression value."""
    return a * x**b

# Register NFT callbacks
register_nft_callbacks(app)

# Register NFT analysis callbacks
register_analysis_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)