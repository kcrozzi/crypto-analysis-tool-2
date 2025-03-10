from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from nft_data.nft_database import SolanaDatabase
from nft_analysis.solana_regression import SolanaRegressionAnalyzer
from nft_nft.nft_api import SolanaAPI
from nft_nft.ethereum_api import EthereumAPI
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import traceback

# Initialize modules
solana_db = SolanaDatabase()
solana_api = SolanaAPI()
ethereum_api = EthereumAPI()
regression_analyzer = SolanaRegressionAnalyzer()

# Define the NFT Analysis page layout
nft_analysis_layout = html.Div([
    dcc.Link('Home', href='/', className='link'),
    html.H2("NFT Analysis", className='header'),
    
    # Dropdown for dataset selection
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': dataset, 'value': dataset} for dataset in solana_db.list_datasets()],
        multi=True,
        placeholder="Select datasets",
        className='dash-dropdown'
    ),
    
    # Checkbox for refreshing datasets
    dbc.Checklist(
        options=[{"label": "Refresh datasets before analysis", "value": "refresh"}],
        value=[],
        id="refresh-checkbox",
        inline=True
    ),
    
    # Input for floor price filtering
    dbc.InputGroup([
        dbc.InputGroupText("Min Floor Price (USD)"),
        dbc.Input(id="min-floor-price", type="number", placeholder="Enter min price"),
    ]),
    dbc.InputGroup([
        dbc.InputGroupText("Max Floor Price (USD)"),
        dbc.Input(id="max-floor-price", type="number", placeholder="Enter max price"),
    ]),

    # Button to perform analysis
    dbc.Button("Perform Analysis", id="perform-analysis-button", color="primary", className="mr-2"),
    
    # Hidden area for weights input, initially not displayed
    html.Div(id='weights-input-area', style={'display': 'none'}, children=[
        html.H4("Set Weights for Analysis", className='sub-header'),
        dbc.InputGroup([
            dbc.InputGroupText("Listings Weight"),
            dbc.Input(id="listings-weight", type="number", value=2, min=0, step=0.1),
        ]),
        dbc.InputGroup([
            dbc.InputGroupText("FDV Weight"),
            dbc.Input(id="fdv-weight", type="number", value=3, min=0, step=0.1),
        ]),
        dbc.InputGroup([
            dbc.InputGroupText("Ownership Weight"),
            dbc.Input(id="ownership-weight", type="number", value=2, min=0, step=0.1),
        ]),
        dbc.Button("Confirm Analysis with These Weights", id="confirm-weights-button", color="success", className="mr-2"),
    ]),

    # Dropdown to select analysis type
    dcc.Dropdown(
        id='analysis-type-dropdown',
        options=[
            {'label': 'FDV/Holder Analysis', 'value': 'fdv_holder'},
            {'label': 'Listings Ratio Analysis', 'value': 'listings_ratio'},
            {'label': 'Ownership Analysis', 'value': 'ownership'},
            {'label': 'Combined Analysis', 'value': 'combined'}
        ],
        placeholder="Select analysis type",
        className='dash-dropdown'
    ),
    
    # Output area for analysis results
    html.Div(id='analysis-results', className='output-area'),

    # Output area for resorted results
    html.Div(id='resorted-results', className='output-area'),
])

# Define callbacks for NFT Analysis functionalities
def register_analysis_callbacks(app):
    print("[DEBUG] Registering Analysis callbacks")
    
    @app.callback(
        Output('weights-input-area', 'style'),
        Input('perform-analysis-button', 'n_clicks'),
        State('analysis-type-dropdown', 'value')
    )
    def show_weights_input(n_clicks, analysis_type):
        if n_clicks and analysis_type == 'combined':
            return {'display': 'block'}  # Show weights input area
        return {'display': 'none'}  # Hide weights input area

    @app.callback(
        Output('analysis-results', 'children'),
        Input('perform-analysis-button', 'n_clicks'),
        State('dataset-dropdown', 'value'),
        State('refresh-checkbox', 'value'),
        State('min-floor-price', 'value'),
        State('max-floor-price', 'value'),
        State('analysis-type-dropdown', 'value'),
        State('listings-weight', 'value'),
        State('fdv-weight', 'value'),
        State('ownership-weight', 'value')
    )
    def perform_analysis(n_clicks, selected_datasets, refresh, min_floor, max_floor, analysis_type, listings_weight, fdv_weight, ownership_weight):
        if not n_clicks or not selected_datasets or not analysis_type:
            return "Please select datasets, analysis type, and click 'Perform Analysis'."
        
        # Ensure weights are provided
        if listings_weight is None or fdv_weight is None or ownership_weight is None:
            return "Please enter weights for Listings, FDV, and Ownership before performing analysis."

        try:
            # Ensure SOL price is fetched correctly
            sol_price = solana_api.get_sol_price()
            if sol_price is None:
                return "Error fetching SOL price."
            print(f"[DEBUG] Fetched SOL price: {sol_price}")

            # Refresh datasets if selected
            if 'refresh' in refresh:
                for dataset_name in selected_datasets:
                    dataset_type = solana_db.get_dataset_type(dataset_name)
                    print(f"[DEBUG] Refreshing dataset: {dataset_name} of type {dataset_type}")
                    if dataset_type == 'solana':
                        solana_db.refresh_dataset(dataset_name, solana_api)
                    else:
                        solana_db.refresh_dataset(dataset_name, ethereum_api)
            
            # Perform analysis
            combined_dataset = []
            listings_data = {}
            processed_symbols = set()  # Track processed symbols

            for dataset_name in selected_datasets:
                dataset = solana_db.get_dataset(dataset_name)
                if dataset:
                    print(f"[DEBUG] Processing dataset: {dataset_name}")
                    for project in dataset:
                        symbol = project["symbol"]
                        if symbol in processed_symbols:
                            continue  # Skip if already processed
                        processed_symbols.add(symbol)
                        print(f"[DEBUG] Processing project: {symbol}")
                        if solana_db.get_dataset_type(dataset_name) == 'ethereum':
                            listings_data[symbol] = int(project['collection_stats'].get('listedCount', 0))
                        else:
                            listings = solana_api.fetch_listings(symbol)
                            listings_data[symbol] = int(listings)
                        combined_dataset.append(project)  # Use append instead of extend
            
            if not combined_dataset:
                return "No valid data found in selected datasets."
            
            # Convert floor prices and perform regression analysis
            for project in combined_dataset:
                dataset_type = project.get('dataset_type')
                collection_stats = project.get('collection_stats', {})
                
                # Debugging prints
                print(f"[DEBUG] Analyzing project: {project.get('symbol')}")
                print(f"[DEBUG] Collection stats: {collection_stats}")

                if dataset_type == 'solana':
                    lamports_floor = collection_stats.get('floorPrice')
                    if lamports_floor is None:
                        print(f"[ERROR] Missing floor price for project {project.get('symbol')}")
                        continue  # Skip this project if floor price is missing
                    sol_floor = lamports_floor / 1e9  # Convert lamports to SOL
                    project['floor_price_sol'] = sol_floor
                    project['floor_price'] = sol_floor * sol_price
                    project['fdv'] = project['floor_price'] * project.get('total_supply', 0)
                    
                    # Debugging FDV calculation
                    print(f"[DEBUG] Floor Price: {project['floor_price']}, Total Supply: {project.get('total_supply', 0)}, FDV: {project['fdv']}")
                elif dataset_type == 'ethereum':
                    project['floor_price'] = collection_stats.get('floorPrice')
                    if project['floor_price'] is None:
                        print(f"[ERROR] Missing floor price for project {project.get('symbol')}")
                        print(f"[DEBUG] Collection stats keys: {list(collection_stats.keys())}")
                        print(f"[DEBUG] Collection stats content: {collection_stats}")
                        continue  # Skip this project if floor price is missing
                    project['fdv'] = project['floor_price'] * project.get('total_supply', 0)
                
                # Ensure floor_price is logged
                print(f"[DEBUG] Project {project.get('symbol')} floor price set to: {project.get('floor_price', 'Not Set')}")
                
                # Calculate FDV per holder
                unique_holders = project.get('holder_stats', {}).get('uniqueHolders', 0)
                if unique_holders > 0:
                    project['fdv_per_holder'] = project['fdv'] / unique_holders
                else:
                    project['fdv_per_holder'] = 0  # Handle division by zero
                
                # Calculate Listings Ratio
                total_supply = project.get('total_supply', 0)
                listings_count = listings_data.get(project['symbol'], 0)
                if total_supply > 0:
                    project['listings_ratio'] = listings_count / total_supply
                else:
                    project['listings_ratio'] = 0  # Handle division by zero
                
                # Calculate Ownership Percentage
                if total_supply > 0:
                    project['ownership_percentage'] = unique_holders / total_supply
                else:
                    project['ownership_percentage'] = 0  # Handle division by zero

                # Debugging for ownership analysis
                print(f"[DEBUG] Ownership analysis for project {project.get('symbol')}:")
                print(f"  Total Supply: {total_supply}")
                print(f"  Unique Holders: {unique_holders}")
                print(f"  Ownership Percentage: {project.get('ownership_percentage', 'Not Set')}")
            
            # Debugging before ownership analysis
            if analysis_type == 'ownership':
                print("[DEBUG] Starting ownership analysis")
                for project in combined_dataset:
                    # Check if floor_price is still present
                    if 'floor_price' not in project:
                        print(f"[ERROR] Floor price missing for project {project.get('symbol')} before ownership analysis")
                    else:
                        print(f"[DEBUG] Project {project.get('symbol')} floor price before ownership analysis: {project.get('floor_price')}")

            # Initialize deviation_info
            deviation_info = html.Div("No deviations calculated.")

            # Perform specific analysis based on user selection
            if analysis_type == 'fdv_holder':
                results = regression_analyzer.analyze_fdv_holder(combined_dataset, solana_api=solana_api)
                deviation_info = regression_analyzer.print_fdv_deviations(results)
            elif analysis_type == 'listings_ratio':
                results = regression_analyzer.analyze_listings_ratio(combined_dataset, listings_data, solana_api=solana_api)
                deviation_info = regression_analyzer.print_listings_deviations(results)
            elif analysis_type == 'ownership':
                results = regression_analyzer.analyze_ownership(combined_dataset, solana_api=solana_api)
                deviation_info = regression_analyzer.print_ownership_deviations(results)
            elif analysis_type == 'combined':  # Handle combined analysis
                results = regression_analyzer.analyze_all(combined_dataset, listings_data, solana_api=solana_api)
                deviation_info = html.Div("Combined analysis results are displayed below.")
                return html.Div([
                    plot_combined_results(results, listings_weight, fdv_weight, ownership_weight),
                    deviation_info
                ])
            else:
                return "Invalid analysis type selected."

            if not results:
                return "Analysis did not return any results."

            # Debugging before filtering results
            print("[DEBUG] Results before filtering:")
            print(results[0])
            for result in results:
                # Check if floor_price is still present
                if 'floor_price' not in result:
                    print(f"[ERROR] Floor price missing for project {result.get('symbol', 'Unknown')} before filtering")
                else:
                    print(f"[DEBUG] Project {result.get('symbol', 'Unknown')} floor price before filtering: {result.get('floor_price')}")

            # Filter results
            if min_floor is not None or max_floor is not None:
                results = [
                    result for result in results
                    if 'floor_price' in result and
                       (min_floor is None or result['floor_price'] >= min_floor) and
                       (max_floor is None or result['floor_price'] <= max_floor)
                ]
                if not results:
                    return "No projects match the specified floor price criteria."
            
            # Convert results to DataFrame for plotting
            df = pd.DataFrame(results)
            
            # Plot results and display deviations
            return html.Div([
                plot_results(results, analysis_type),
                html.Div(f"Analysis complete. {len(results)} projects analyzed."),
                deviation_info
            ])
        
        except Exception as e:
            print(f"[ERROR] Error in perform_analysis: {str(e)}")
            print(f"[DEBUG] Exception details: {traceback.format_exc()}")
            return f"An error occurred during analysis: {str(e)}"


def plot_results(filtered_results, analysis_type):
    # Prepare data for plotting
    floor_prices = [result['floor_price'] for result in filtered_results]  # Convert lamports to SOL
    fdv_per_holders = [result['fdv_per_holder'] for result in filtered_results]
    listings_ratios = [result.get('listings_ratio', 0) for result in filtered_results]
    ownership_percentages = [result.get('ownership_percentage', 0) for result in filtered_results]

    # Ensure floor prices are positive for log transformation
    floor_prices = np.array(floor_prices) + 1e-10

    fig = go.Figure()
    regression_info = html.Div()

    x_range = np.linspace(min(floor_prices), max(floor_prices), 100)

    if analysis_type == 'fdv_holder':
        poly_coeffs = np.polyfit(floor_prices, fdv_per_holders, 2)
        expected_fdv_line = np.polyval(poly_coeffs, x_range)

        fig.add_trace(go.Scatter(
            x=floor_prices,
            y=fdv_per_holders,
            mode='markers',
            name='FDV per Holder'
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=expected_fdv_line,
            mode='lines',
            name='FDV per Holder Regression',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Floor Price vs FDV per Holder',
            xaxis_title='Floor Price',
            yaxis_title='FDV per Holder',
            xaxis_type='log'
        )

        fdv_r2 = np.corrcoef(fdv_per_holders, np.polyval(poly_coeffs, floor_prices))[0, 1] ** 2
        fdv_mae = mean_absolute_error(fdv_per_holders, np.polyval(poly_coeffs, floor_prices))
        fdv_equation = f"y = {poly_coeffs[0]:.4f}x² + {poly_coeffs[1]:.4f}x + {poly_coeffs[2]:.4f}"

        regression_info = html.Div([
            html.H4("Regression Statistics"),
            html.P(f"FDV Regression: R²={fdv_r2:.2f}, MAE={fdv_mae:.2f}, Equation: {fdv_equation}"),
            html.Hr()
        ])

    elif analysis_type == 'listings_ratio':
        # Log-linear regression for Listings Ratio
        log_floor_prices = np.log(floor_prices)
        listings_intercept, listings_slope = np.polyfit(log_floor_prices, listings_ratios, 1)
        expected_listings_line = listings_intercept + listings_slope * np.log(x_range)

        # Plot Floor Price vs Listings Ratio
        fig.add_trace(go.Scatter(
            x=floor_prices,
            y=listings_ratios,
            mode='markers',
            name='Listings Ratio'
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=expected_listings_line,
            mode='lines',
            name='Listings Ratio Regression',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Floor Price vs Listings Ratio',
            xaxis_title='Floor Price',
            yaxis_title='Listings Ratio',
            xaxis_type='log'
        )

        # Calculate regression statistics
        listings_r2 = np.corrcoef(listings_ratios, listings_intercept + listings_slope * log_floor_prices)[0, 1] ** 2
        listings_mae = mean_absolute_error(listings_ratios, listings_intercept + listings_slope * log_floor_prices)
        listings_equation = f"y = {listings_intercept:.4f} + {listings_slope:.4f}log(x)"

        regression_info = html.Div([
            html.H4("Regression Statistics"),
            html.P(f"Listings Regression: R²={listings_r2:.2f}, MAE={listings_mae:.2f}, Equation: {listings_equation}"),
            html.Hr()
        ])

    elif analysis_type == 'ownership':
        # Log-linear regression for Ownership Percentage
        log_floor_prices = np.log(floor_prices)
        ownership_intercept, ownership_slope = np.polyfit(log_floor_prices, ownership_percentages, 1)
        expected_ownership_line = ownership_intercept + ownership_slope * np.log(x_range)

        # Plot Floor Price vs Ownership Ratio
        fig.add_trace(go.Scatter(
            x=floor_prices,
            y=ownership_percentages,
            mode='markers',
            name='Ownership Ratio'
        ))
        fig.add_trace(go.Scatter(
            x=x_range,
            y=expected_ownership_line,
            mode='lines',
            name='Ownership Ratio Regression',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Floor Price vs Ownership Ratio',
            xaxis_title='Floor Price',
            yaxis_title='Ownership Ratio',
            xaxis_type='log'
        )

        # Calculate regression statistics
        ownership_r2 = np.corrcoef(ownership_percentages, ownership_intercept + ownership_slope * log_floor_prices)[0, 1] ** 2
        ownership_mae = mean_absolute_error(ownership_percentages, ownership_intercept + ownership_slope * log_floor_prices)
        ownership_equation = f"y = {ownership_intercept:.4f} + {ownership_slope:.4f}log(x)"

        regression_info = html.Div([
            html.H4("Regression Statistics"),
            html.P(f"Ownership Regression: R²={ownership_r2:.2f}, MAE={ownership_mae:.2f}, Equation: {ownership_equation}"),
            html.Hr()
        ])

    return html.Div([
        dcc.Graph(figure=fig),
        regression_info
    ])

def plot_combined_results(combined_results, listings_weight, fdv_weight, ownership_weight):
    """Plot combined analysis results with separate plots for each metric and include deviations and regression statistics."""
    print("[DEBUG] Starting plot_combined_results function")
    
    # Extract data for plotting
    floor_prices = [result['floor_price'] for result in combined_results]
    fdv_per_holders = [result['fdv_per_holder'] for result in combined_results]
    listings_ratios = [result.get('listings_ratio', 0) for result in combined_results]
    ownership_percentages = [result.get('ownership_percentage', 0) for result in combined_results]
    
    print(f"[DEBUG] Extracted floor_prices: {floor_prices}")
    print(f"[DEBUG] Extracted fdv_per_holders: {fdv_per_holders}")
    print(f"[DEBUG] Extracted listings_ratios: {listings_ratios}")
    print(f"[DEBUG] Extracted ownership_percentages: {ownership_percentages}")

    # Calculate combined deviation for each project using weights
    for result in combined_results:
        fdv_deviation = result.get('fdv_deviation', 0)
        listings_deviation = result.get('listings_deviation', 0)
        ownership_deviation = result.get('ownership_deviation', 0)
        
        # Ensure deviations are scalars
        if isinstance(fdv_deviation, np.ndarray):
            fdv_deviation = fdv_deviation.item()
        if isinstance(listings_deviation, np.ndarray):
            listings_deviation = listings_deviation.item()
        if isinstance(ownership_deviation, np.ndarray):
            ownership_deviation = ownership_deviation.item()
        
        # Calculate the weighted average deviation
        total_weight = fdv_weight + listings_weight + ownership_weight
        combined_deviation = (
            (fdv_deviation * fdv_weight) +
            (listings_deviation * listings_weight) +
            (ownership_deviation * ownership_weight)
        ) / total_weight
        
        # Ensure combined_deviation is a scalar
        if isinstance(combined_deviation, np.ndarray):
            combined_deviation = combined_deviation.item()
        
        result['combined_deviation'] = combined_deviation
        print(f"[DEBUG] Combined deviation for {result.get('symbol', 'Unknown')}: {combined_deviation}")

    # Sort results by combined deviation
    combined_results.sort(key=lambda x: x['combined_deviation'], reverse=True)

    # Plot FDV per Holder
    fig_fdv = go.Figure()
    fig_fdv.add_trace(go.Scatter(
        x=floor_prices,
        y=fdv_per_holders,
        mode='markers',
        name='FDV per Holder'
    ))
    fig_fdv.update_layout(
        title='Floor Price vs FDV per Holder',
        xaxis_title='Floor Price',
        yaxis_title='FDV per Holder',
        xaxis_type='log'
    )
    print("[DEBUG] FDV per Holder plot created")

    # Plot Listings Ratio
    fig_listings = go.Figure()
    fig_listings.add_trace(go.Scatter(
        x=floor_prices,
        y=listings_ratios,
        mode='markers',
        name='Listings Ratio'
    ))
    fig_listings.update_layout(
        title='Floor Price vs Listings Ratio',
        xaxis_title='Floor Price',
        yaxis_title='Listings Ratio',
        xaxis_type='log'
    )
    print("[DEBUG] Listings Ratio plot created")

    # Plot Ownership Percentage
    fig_ownership = go.Figure()
    fig_ownership.add_trace(go.Scatter(
        x=floor_prices,
        y=ownership_percentages,
        mode='markers',
        name='Ownership Percentage'
    ))
    fig_ownership.update_layout(
        title='Floor Price vs Ownership Percentage',
        xaxis_title='Floor Price',
        yaxis_title='Ownership Percentage',
        xaxis_type='log'
    )
    print("[DEBUG] Ownership Percentage plot created")

    # Aggregate regression statistics
    regression_info = html.Div([
        html.H4("Regression Statistics"),
        html.P("FDV Regression:"),
        html.P(f"Equation: y = {combined_results[0]['fdv_analysis']['polynomial']['coefficients'][0]:.4f}x² + {combined_results[0]['fdv_analysis']['polynomial']['coefficients'][1]:.4f}x + {combined_results[0]['fdv_analysis']['polynomial']['coefficients'][2]:.4f}"),
        html.P(f"R² Score: {combined_results[0]['fdv_analysis']['polynomial']['r2_score']:.4f}"),
        html.P(f"MAE: {combined_results[0]['fdv_analysis']['polynomial']['mae']:.4f}"),
        html.P(f"RMSE: {combined_results[0]['fdv_analysis']['polynomial']['rmse']:.4f}"),
        html.Hr(),
        html.P("Listings Regression:"),
        html.P(f"Equation: y = {combined_results[0]['listings_analysis']['log_linear']['coefficients'][0]:.4f} + {combined_results[0]['listings_analysis']['log_linear']['coefficients'][1]:.4f}log(x)"),
        html.P(f"R² Score: {combined_results[0]['listings_analysis']['log_linear']['r2_score']:.4f}"),
        html.P(f"MAE: {combined_results[0]['listings_analysis']['log_linear']['mae']:.4f}"),
        html.P(f"RMSE: {combined_results[0]['listings_analysis']['log_linear']['rmse']:.4f}"),
        html.Hr(),
        html.P("Ownership Regression:"),
        html.P(f"Equation: y = {combined_results[0]['ownership_analysis']['log_linear']['coefficients'][0]:.4f} + {combined_results[0]['ownership_analysis']['log_linear']['coefficients'][1]:.4f}log(x)"),
        html.P(f"R² Score: {combined_results[0]['ownership_analysis']['log_linear']['r2_score']:.4f}"),
        html.P(f"MAE: {combined_results[0]['ownership_analysis']['log_linear']['mae']:.4f}"),
        html.P(f"RMSE: {combined_results[0]['ownership_analysis']['log_linear']['rmse']:.4f}"),
        html.Hr()
    ])

    # Consolidate deviations for each project
    deviation_info = []
    for result in combined_results:
        symbol = result.get('symbol', 'Unknown')
        fdv_dev = result.get('fdv_deviation', 0)
        listings_dev = result.get('listings_deviation', 0)
        ownership_dev = result.get('ownership_deviation', 0)
        combined_dev = result.get('combined_deviation', 0)
        
        # Ensure deviations are scalars
        if isinstance(fdv_dev, np.ndarray):
            fdv_dev = fdv_dev.item()
        if isinstance(listings_dev, np.ndarray):
            listings_dev = listings_dev.item()
        if isinstance(ownership_dev, np.ndarray):
            ownership_dev = ownership_dev.item()
        if isinstance(combined_dev, np.ndarray):
            combined_dev = combined_dev.item()
        
        expected_fdv = result.get('expected_fdv_per_holder', 0)
        expected_listings = result.get('expected_listings_ratio', 0)
        expected_ownership = result.get('expected_ownership_percentage', 0)
        
        # Ensure expected values are scalars
        if isinstance(expected_fdv, np.ndarray):
            expected_fdv = expected_fdv.item()
        if isinstance(expected_listings, np.ndarray):
            expected_listings = expected_listings.item()
        if isinstance(expected_ownership, np.ndarray):
            expected_ownership = expected_ownership.item()
        
        actual_fdv = result.get('fdv_per_holder', 0)
        actual_listings = result.get('listings_ratio', 0)
        actual_ownership = result.get('ownership_percentage', 0)

        deviation_info.append(html.Div([
            html.H5(f"Project: {symbol}"),
            html.P(f"FDV/Holder Deviation: {fdv_dev:.4f} ({fdv_dev*100:.1f}%)"),
            html.P(f"Expected FDV/Holder: ${expected_fdv:,.2f}"),
            html.P(f"Actual FDV/Holder: ${actual_fdv:,.2f}"),
            html.P(f"FDV: ${result['fdv']:.2f}"),  # Display FDV
            html.P(f"Floor Price: ${result['floor_price']:.2f}"),  # Display floor price
            html.P(f"Listings Deviation: {listings_dev:.4f} ({listings_dev*100:.1f}%)"),
            html.P(f"Expected Listings/Supply: {expected_listings:.2f}%"),
            html.P(f"Actual Listings/Supply: {actual_listings:.2f}%"),
            html.P(f"Ownership Deviation: {ownership_dev:.4f} ({ownership_dev*100:.1f}%)"),
            html.P(f"Expected Ownership: {expected_ownership*100:.2f}%"),
            html.P(f"Actual Ownership: {actual_ownership*100:.2f}%"),
            html.P(f"Combined Deviation: {combined_dev:.4f}"),
            html.Hr()
        ]))

    print("[DEBUG] Returning combined HTML Div with plots, regression statistics, and deviations")
    return html.Div([
        dcc.Graph(figure=fig_fdv),
        dcc.Graph(figure=fig_listings),
        dcc.Graph(figure=fig_ownership),
        regression_info,  # Display regression statistics after plots
        html.Div(deviation_info)  # Display consolidated deviations for each project
    ])




