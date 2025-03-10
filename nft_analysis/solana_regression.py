import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import json
import traceback
import plotly.graph_objects as go
import dash.html as html
import dash.dcc as dcc
from plotly.subplots import make_subplots

class SolanaRegressionAnalyzer:
    def __init__(self):
        """Initialize the regression analyzer with polynomial features."""
        self.poly_features = PolynomialFeatures(degree=1, include_bias=True)

    def analyze_dataset(self, dataset):
        """Perform regression analysis on the dataset."""
        print("\n=== Starting Dataset Analysis ===")
        
        data_points = []
        analyses = {}
        
        # Process each project in the dataset
        for project in dataset:
            try:
                # Extract and validate required data
                holder_stats = project.get('holder_stats', {})
                collection_stats = project.get('collection_stats', {})
                
                if not holder_stats or not collection_stats:
                    print(f"Skipping project {project.get('symbol')}: Missing stats")
                    continue
                
                # Extract values with validation
                floor_price = collection_stats.get('floorPrice', 0)
                total_supply = holder_stats.get('totalSupply', 0)
                unique_holders = holder_stats.get('uniqueHolders', 0)
                
                if floor_price <= 0 or total_supply <= 0 or unique_holders <= 0:
                    print(f"Skipping project {project.get('symbol')}: Invalid values")
                    continue
                
                # Calculate metrics
                fdv = floor_price * total_supply
                fdv_per_holder = fdv / unique_holders if unique_holders > 0 else 0
                
                data_point = {
                    'symbol': project.get('symbol'),
                    'floor_price': floor_price,
                    'fdv': fdv,
                    'fdv_per_holder': fdv_per_holder,
                    'holder_stats': holder_stats,
                    'total_supply': total_supply
                }
                
                # Add FDV ratio if available
                if project.get('fdv_ratio') is not None:
                    data_point['fdv_ratio'] = project['fdv_ratio']
                
                data_points.append(data_point)
                
            except Exception as e:
                print(f"Error processing project {project.get('symbol')}: {str(e)}")
                continue
        
        if not data_points:
            print("No valid data points found for analysis")
            return None
        
        # Prepare data for FDV per holder analysis
        X_fdv = np.array([p['floor_price'] for p in data_points]).reshape(-1, 1)
        y_fdv = np.array([p['fdv_per_holder'] for p in data_points])
        
        analyses['fdv_per_holder'] = self._perform_regression(X_fdv, y_fdv)
        print("\nDEBUG - Analysis results after regression:")
        print(f"FDV Analysis: {analyses['fdv_per_holder']}")
        
        # FDV ratio analysis (if available)
        fdv_ratio_points = [p for p in data_points if p.get('fdv_ratio') is not None]
        if fdv_ratio_points:
            X_ratio = np.array([p['floor_price'] for p in fdv_ratio_points]).reshape(-1, 1)
            y_ratio = np.array([p['fdv_ratio'] for p in fdv_ratio_points])
            
            analyses['fdv_ratio'] = self._perform_regression(X_ratio, y_ratio)
        
        return {'data_points': data_points, 'analyses': analyses}

    def _perform_regression(self, X, y, regression_type='power', degree=2):
        """Perform regression analysis with both power and polynomial options."""
        # Ensure X and y are numpy arrays
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Power regression (keep existing code)
        X_log = np.log(X)
        y_log = np.log(y)
        coeffs = np.polyfit(X_log.flatten(), y_log, 1)
        a = np.exp(coeffs[1])
        b = coeffs[0]
        power_expected = a * (X ** b).flatten()
        
        # Polynomial regression
        poly = np.polynomial.Polynomial.fit(X.flatten(), y, degree)
        poly_coeffs = poly.convert().coef  # Ensure coefficients are in the correct order
        poly_expected = poly(X.flatten())
        
        # Reverse the order of coefficients for correct polynomial evaluation
        poly_coeffs = poly_coeffs[::-1]
        
        # Calculate metrics for both regression types
        if regression_type == 'power':
            expected_values = power_expected
            equation_coeffs = [a, b]
        else:  # polynomial
            expected_values = poly_expected
            equation_coeffs = poly_coeffs
        
        # Calculate regression metrics
        r2 = r2_score(y, expected_values)
        mae = mean_absolute_error(y, expected_values)
        rmse = np.sqrt(mean_squared_error(y, expected_values))
        
        # Debugging: Print calculated metrics
        print(f"[DEBUG] Regression metrics - R²: {r2}, MAE: {mae}, RMSE: {rmse}")
        
        return {
            'power': {
                'coefficients': [a, b],
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'expected_values': power_expected.tolist()
            },
            'polynomial': {
                'coefficients': poly_coeffs.tolist(),
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'expected_values': poly_expected.tolist()
            }
        }

    def print_results(self, results):
        if not results:
            print("No results to display")
            return
        
        print("\n=== Deviation Analysis Results ===")
        print(f"Total Projects Analyzed: {len(results['data_points'])}")
        print("-" * 50 + "\n")
        
        for point in results['data_points']:
            symbol = point.get('symbol', 'Unknown')
            floor_price = point.get('floor_price', 0)
            total_supply = point.get('total_supply', 0)
            fdv = point.get('fdv', 0)
            unique_holders = point['holder_stats'].get('uniqueHolders', 0)
            holder_percentage = (unique_holders / total_supply * 100) if total_supply > 0 else 0
            listings = point.get('listings', 0)
            listings_ratio = (listings / total_supply * 100) if total_supply > 0 else 0
            
            print(f"#{symbol}")
            print(f"Floor Price: ${floor_price:,.2f}")
            print(f"Total Supply: {total_supply:,}")
            print(f"FDV: ${fdv:,.2f}")
            print(f"Unique Holders: {unique_holders:,}")
            print(f"Holder Percentage: {holder_percentage:.2f}%")
            print(f"Active Listings: {listings:,}")
            print(f"Listings Ratio: {listings_ratio:.2f}%")
            print("\nDeviation Analysis:")
            
            # Get deviations and expected values directly from the analysis results
            fdv_dev = point.get('fdv_deviation', 0)
            listings_dev = point.get('listings_deviation', 0)
            ownership_dev = point.get('ownership_deviation', 0)
            combined_dev = point.get('combined_deviation', 0)
            
            # Get expected values from the analysis results
            expected_fdv = point.get('expected_fdv_per_holder', 0)
            expected_listings = point.get('expected_listings_ratio', 0)
            expected_ownership = point.get('expected_ownership_ratio', 0)
            
            # Calculate actual values
            fdv_per_holder = fdv / unique_holders if unique_holders > 0 else 0
            
            print(f"Combined Deviation: {combined_dev:.4f} ({combined_dev*100:.1f}%)")
            print(f"→ FDV/Holder Deviation: {fdv_dev:.4f} ({fdv_dev*100:.1f}%)")
            print(f"  Expected FDV/Holder: ${expected_fdv:,.2f}")
            print(f"  Actual FDV/Holder: ${fdv_per_holder:,.2f}")
            print(f"→ Listings Deviation: {listings_dev:.4f} ({listings_dev*100:.1f}%)")
            print(f"  Expected Listings/Supply: {expected_listings:.2f}%")
            print(f"  Actual Listings/Supply: {listings_ratio:.2f}%")
            print(f"→ Ownership Deviation: {ownership_dev:.4f} ({ownership_dev*100:.1f}%)")
            print(f"  Expected Ownership: {expected_ownership*100:.2f}%")
            print(f"  Actual Ownership: {point.get('ownership_percentage', 0)*100:.2f}%")
            print("-" * 30 + "\n")
            
            # Get the stored expected values directly from the point dictionary
            expected_fdv = point.get('expected_fdv_per_holder', 0)
            expected_listings = point.get('expected_listings_ratio', 0)
            expected_ownership = point.get('expected_ownership_ratio', 0)
            
            # Debug print
            print(f"\nDEBUG - Retrieved expected values for {point.get('symbol')}:")
            print(f"FDV: {expected_fdv}")
            print(f"Listings: {expected_listings}")
            print(f"Ownership: {expected_ownership}")

    def plot_results(self, filtered_results, fdv_model, listings_model, ownership_model):
        # Prepare data for plotting
        floor_prices = [result['floor_price'] for result in filtered_results]
        listings_ratios = [result['listings_ratio'] for result in filtered_results]
        fdv_per_holders = [result['fdv_per_holder'] for result in filtered_results]
        ownership_percentages = [result['ownership_percentage'] for result in filtered_results]

        # Create a continuous range for floor prices
        x_range = np.linspace(min(floor_prices), max(floor_prices), 100)

        # Calculate expected values for the regression line
        expected_fdv_line = np.exp(fdv_model.intercept_ + fdv_model.coef_[0] * np.log(x_range))
        expected_listings_line = np.exp(listings_model.intercept_ + listings_model.coef_[0] * np.log(x_range))
        expected_ownership_line = np.exp(ownership_model.intercept_ + ownership_model.coef_[0] * np.log(x_range))

        # Plot Floor Price vs Listings Ratio
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=floor_prices,
            y=listings_ratios,
            mode='markers',
            name='Listings Ratio'
        ))
        fig1.add_trace(go.Scatter(
            x=x_range,
            y=expected_listings_line,
            mode='lines',
            name='Listings Ratio Regression',
            line=dict(color='red')
        ))
        fig1.update_layout(title='Floor Price vs Listings Ratio', xaxis_title='Floor Price', yaxis_title='Listings Ratio')

        # Plot Floor Price vs FDV per Holder
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=floor_prices,
            y=fdv_per_holders,
            mode='markers',
            name='FDV per Holder'
        ))
        fig2.add_trace(go.Scatter(
            x=x_range,
            y=expected_fdv_line,
            mode='lines',
            name='FDV per Holder Regression',
            line=dict(color='red')
        ))
        fig2.update_layout(title='Floor Price vs FDV per Holder', xaxis_title='Floor Price', yaxis_title='FDV per Holder')

        # Plot Floor Price vs Ownership Ratio
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=floor_prices,
            y=ownership_percentages,
            mode='markers',
            name='Ownership Ratio'
        ))
        fig3.add_trace(go.Scatter(
            x=x_range,
            y=expected_ownership_line,
            mode='lines',
            name='Ownership Ratio Regression',
            line=dict(color='red')
        ))
        fig3.update_layout(title='Floor Price vs Ownership Ratio', xaxis_title='Floor Price', yaxis_title='Ownership Ratio')

        # Add regression statistics
        fdv_r2 = fdv_model.score(np.log(np.array(floor_prices).reshape(-1, 1)), np.log(np.array(fdv_per_holders)))
        listings_r2 = listings_model.score(np.log(np.array(floor_prices).reshape(-1, 1)), np.log(np.array(listings_ratios)))
        ownership_r2 = ownership_model.score(np.log(np.array(floor_prices).reshape(-1, 1)), np.log(np.array(ownership_percentages)))

        regression_info = html.Div([
            html.H4("Regression Statistics"),
            html.P(f"FDV Regression: R²={fdv_r2:.2f}"),
            html.P(f"Listings Regression: R²={listings_r2:.2f}"),
            html.P(f"Ownership Regression: R²={ownership_r2:.2f}"),
            html.Hr()
        ])

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
            regression_info
        ])

    def analyze_listings(self, dataset, listings_data):
        """Analyze listings/supply ratio against FDV."""
        data_points = []
        
        print("\nProcessing listings data:")
        for project in dataset:
            # Only analyze if we have the required NFT data
            if (project["holder_stats"] and project["collection_stats"] and 
                "totalSupply" in project["holder_stats"] and 
                "floorPrice" in project["collection_stats"]):
                
                # Get project data
                symbol = project["symbol"]
                total_supply = project["holder_stats"]["totalSupply"]
                floor_price = project["collection_stats"]["floorPrice"]
                
                # Get listings count (now it's directly a number)
                listings_count = listings_data.get(symbol, 0)
                print(f"\nProject: {symbol}")
                print(f"Listings count: {listings_count}")
                print(f"Total supply: {total_supply}")
                
                if listings_count > 0 and total_supply > 0:
                    listings_ratio = listings_count / total_supply
                    data_points.append({
                        "symbol": symbol,
                        "floor_price": floor_price,
                        "listings_ratio": listings_ratio,
                        "total_supply": total_supply,
                        "listings_count": listings_count
                    })
                    print(f"Added to analysis with {listings_count} listings")
                else:
                    print(f"Skipped: listings_count={listings_count}, total_supply={total_supply}")
        
        if not data_points:
            print("No valid data points found for listings analysis.")
            return None
        
        # Prepare data for regression
        X = np.array([p['floor_price'] for p in data_points]).reshape(-1, 1)
        y = np.array([p['listings_ratio'] for p in data_points])
        
        # Use log-linear regression instead of power/polynomial
        analysis = self._perform_log_linear_regression(X.flatten(), y)
        return {'data_points': data_points, 'analysis': analysis}

    def plot_listings_analysis(self, results):
        """Plot FDV vs Listings Ratio with log-linear regression line."""
        if not results:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract data points
        X = np.array([p['floor_price'] for p in results['data_points']]).reshape(-1, 1)
        y = np.array([p['listings_ratio'] for p in results['data_points']])
        
        # Scatter plot of actual data points
        plt.scatter(X, y, color='blue', label='Data Points')
        
        # Add labels to each point
        for i, point in enumerate(results['data_points']):
            plt.annotate(point['symbol'], (X[i], y[i]), fontsize=8, ha='right')
        
        # Plot regression line
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_log = np.log(X_range + 1e-10)  # Add small constant to avoid log(0)
        
        # Calculate predicted values using log-linear coefficients
        intercept, slope = results['analysis']['coefficients']
        y_pred = intercept + slope * X_log.flatten()
        plt.plot(X_range, y_pred, color='red', label='Log-Linear Regression')
        
        plt.title('Floor Price vs Listings Ratio')
        plt.xlabel('Floor Price')
        plt.ylabel('Listings/Supply Ratio')
        plt.xscale('log')  # X-axis is logarithmic
        plt.legend()
        plt.grid(True)
        
        # Adjust y-axis limits to reduce empty space
        y_min = min(y) * 0.9  # Add 10% padding below
        y_max = max(y) * 1.1  # Add 10% padding above
        plt.ylim(y_min, y_max)
        
        plt.show()

    def print_listings_results(self, results):
        """Print regression results specifically for listings analysis."""
        if not results:
            return

        print("\nSolana NFT Listings Analysis Results:")
        print("-" * 50)

        analysis = results['analysis']
        
        if 'polynomial' in analysis:
            coef = analysis['polynomial']['coefficients']
            print("\nPolynomial Regression:")
            print(f"Equation: y = {coef[1]:.4e}x + {coef[0]:.4e}")
            print(f"R² Score: {analysis['polynomial']['r2_score']:.4f}")
            print(f"MAE: {analysis['polynomial']['mae']:.4f}")

        if 'power' in analysis:
            a, b = analysis['power']['coefficients']
            print("\nPower Regression:")
            print(f"Equation: y = {a:.4f}x^{b:.4f}")
            print(f"R² Score: {analysis['power']['r2_score']:.4f}")
            print(f"MAE: {analysis['power']['mae']:.4f}")

        print("\nDetailed Results:")
        for point in results['data_points']:
            print(f"\nProject Symbol: {point['symbol']}")
            print(f"Floor Price: ${point['floor_price']:,.2f}")
            print(f"FDV: ${point['fdv']:,.2f}")
            print(f"Total Supply: {point['total_supply']:,}")
            print(f"Owner Count: {point.get('holder_stats', {}).get('uniqueHolders', 'N/A'):,}")
            print(f"FDV per Holder: ${point['fdv'] / point.get('holder_stats', {}).get('uniqueHolders', 0):,.2f}")
            print(f"Active Listings: {point['listings_count']:,}")
            print(f"Listings Percentage: {(point['listings_ratio'] * 100):.2f}%")
            print("-" * 30)

    def analyze_fdv_holder(self, dataset, solana_api=None):
        """Perform FDV/Holder analysis."""
        print("\n=== FDV/Holder Analysis ===")
        regression_data = self._prepare_regression_data(dataset, solana_api)
        if not regression_data:
            return []

        X = np.array([[p['floor_price']] for p in regression_data])
        y_fdv = np.array([p['fdv_per_holder'] for p in regression_data])

        # Add small constant to avoid log(0)
        X = X + 1e-10
        y_fdv = y_fdv + 1e-10

        # Fit FDV model using polynomial regression
        polynomial_analysis = self._perform_regression(X, y_fdv, regression_type='polynomial', degree=2)

        # Debugging: Print polynomial analysis results
        print(f"[DEBUG] Polynomial analysis results: {polynomial_analysis}")

        # Calculate expected values and deviations
        results = self._calculate_fdv_deviations(regression_data, polynomial_analysis)

        # Include polynomial analysis in results
        for result in results:
            result['fdv_analysis'] = {'polynomial': polynomial_analysis['polynomial']}

        # Debugging: Print the results to verify structure
        print("[DEBUG] FDV Analysis Results:", results)

        return results

    def _calculate_fdv_deviations(self, regression_data, fdv_model):
        """Calculate deviations for FDV/Holder analysis using polynomial regression."""
        results = []
        for data in regression_data:
            try:
                x_val = np.array([[data['floor_price'] + 1e-10]])  # Add small constant to avoid log(0)
                
                # Use polynomial coefficients to calculate expected FDV per holder
                poly_coeffs = fdv_model['polynomial']['coefficients']
                expected_fdv_per_holder = np.polyval(poly_coeffs, x_val.flatten())
                
                actual_fdv_per_holder = data['fdv_per_holder']
                
                # Check for valid expected FDV per holder
                if expected_fdv_per_holder <= 0:
                    print(f"Invalid expected FDV per holder for {data['symbol']}: {expected_fdv_per_holder}")
                    continue

                fdv_deviation = (actual_fdv_per_holder - expected_fdv_per_holder) / expected_fdv_per_holder if expected_fdv_per_holder > 0 else 0
                fdv_deviation *= -1  # Flip the sign

                results.append({
                    **data,
                    'expected_fdv_per_holder': expected_fdv_per_holder,
                    'fdv_deviation': fdv_deviation
                })

            except Exception as e:
                print(f"Error calculating deviation for {data['symbol']}: {str(e)}")
                continue

        return results

    def print_fdv_deviations(self, results):
        """Return deviations for FDV/Holder analysis as HTML."""
        if not results:
            return html.Div("No results to display")

        deviation_info = []
        for result in results:
            symbol = result.get('symbol', 'Unknown')
            fdv_deviation = result.get('fdv_deviation', 0)
            expected_fdv_per_holder = result.get('expected_fdv_per_holder', 0)
            actual_fdv_per_holder = result.get('fdv_per_holder', 0)
            fdv = result.get('fdv', 0)  # Get FDV
            floor_price = result.get('floor_price', 0)  # Get floor price

            # Ensure fdv_deviation is a scalar
            if isinstance(fdv_deviation, np.ndarray):
                fdv_deviation = fdv_deviation.item()  # Convert to scalar if it's an array

            # Ensure expected_fdv_per_holder is a scalar
            if isinstance(expected_fdv_per_holder, np.ndarray):
                expected_fdv_per_holder = expected_fdv_per_holder.item()  # Convert to scalar if it's an array

            deviation_info.append(html.Div([
                html.H5(f"Project: {symbol}"),
                html.P(f"FDV/Holder Deviation: {fdv_deviation:.4f} ({fdv_deviation*100:.1f}%)"),
                html.P(f"Expected FDV/Holder: ${expected_fdv_per_holder:,.2f}"),
                html.P(f"Actual FDV/Holder: ${actual_fdv_per_holder:,.2f}"),
                html.P(f"FDV: ${fdv:,.2f}"),  # Display FDV
                html.P(f"Floor Price: ${floor_price:,.2f}"),  # Display floor price
                html.Hr()
            ]))

        return html.Div(deviation_info)

    def analyze_listings_ratio(self, dataset, listings_data, solana_api=None):
        """Perform Listings Ratio analysis."""
        print("\n=== Listings Ratio Analysis ===")
        regression_data = self._prepare_regression_data(dataset, solana_api, listings_data)
        if not regression_data:
            return []

        X = np.array([[p['floor_price']] for p in regression_data])
        y_listings = np.array([p['listings_ratio'] for p in regression_data])

        # Add small constant to avoid log(0)
        X = X + 1e-10
        y_listings = y_listings + 1e-10

        # Perform log-linear regression
        log_linear_analysis = self._perform_log_linear_regression(X.flatten(), y_listings)

        # Debugging: Print log-linear analysis results
        print(f"[DEBUG] Log-Linear analysis results: {log_linear_analysis}")

        # Calculate expected values and deviations
        results = []
        for data in regression_data:
            try:
                x_val = np.array([[data['floor_price'] + 1e-10]])
                # Use the predicted value directly without exponential transformation
                expected_listings_ratio = log_linear_analysis['expected_values'][regression_data.index(data)]
                actual_listings_ratio = data['listings_ratio']
                listings_deviation = (actual_listings_ratio - expected_listings_ratio) / expected_listings_ratio if expected_listings_ratio > 0 else 0
                listings_deviation *= -1  # Flip the sign

                results.append({
                    **data,
                    'expected_listings_ratio': expected_listings_ratio,  # Use predicted value directly
                    'listings_deviation': listings_deviation,  # Use the float value directly
                    'listings_analysis': {'log_linear': log_linear_analysis}
                })

            except Exception as e:
                print(f"Error calculating deviation for {data['symbol']}: {str(e)}")
                continue

        # Debugging: Print the results to verify structure
        print("[DEBUG] Listings Analysis Results:", results)

        return results

    def analyze_ownership(self, dataset, solana_api=None):
        """Perform Ownership analysis."""
        print("\n=== Ownership Analysis ===")
        regression_data = self._prepare_regression_data(dataset, solana_api)
        if not regression_data:
            return []

        X = np.array([[p['floor_price']] for p in regression_data])
        y_ownership = np.array([p['ownership_percentage'] for p in regression_data])

        # Add small constant to avoid log(0)
        X = X + 1e-10
        y_ownership = y_ownership + 1e-10

        # Fit Ownership model
        ownership_model = LinearRegression()
        ownership_model.fit(np.log(X), np.log(y_ownership))

        # Calculate expected values and deviations
        results = []
        for data in regression_data:
            try:
                x_val = np.array([[data['floor_price'] + 1e-10]])
                expected_ownership_percentage = np.exp(ownership_model.predict(np.log(x_val))[0])
                actual_ownership_percentage = data['ownership_percentage']
                ownership_deviation = (actual_ownership_percentage - expected_ownership_percentage) / expected_ownership_percentage if expected_ownership_percentage > 0 else 0

                results.append({
                    **data,
                    'expected_ownership_percentage': expected_ownership_percentage,
                    'ownership_deviation': ownership_deviation
                })

            except Exception as e:
                print(f"Error calculating deviation for {data['symbol']}: {str(e)}")
                continue

        # Perform log-linear regression
        log_floor_prices = np.log([p['floor_price'] for p in regression_data])
        ownership_percentages = [p['ownership_percentage'] for p in regression_data]

        model = LinearRegression()
        model.fit(log_floor_prices.reshape(-1, 1), ownership_percentages)

        # Calculate predicted values
        y_pred = model.predict(log_floor_prices.reshape(-1, 1))

        # Calculate regression metrics
        r2 = r2_score(ownership_percentages, y_pred)
        mae = mean_absolute_error(ownership_percentages, y_pred)
        rmse = np.sqrt(mean_squared_error(ownership_percentages, y_pred))

        # Store regression results
        ownership_analysis = {
            'coefficients': [model.intercept_, model.coef_[0]],  # [intercept, slope]
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'expected_values': y_pred.tolist()
        }

        # Include ownership analysis in results
        for result in results:
            result['ownership_analysis'] = {'log_linear': ownership_analysis}

        # Debugging: Print the results to verify structure
        print("[DEBUG] Ownership Analysis Results:", results)

        return results

    def _prepare_regression_data(self, dataset, solana_api=None, listings_data=None):
        """Prepare data for regression analysis."""
        sol_price = solana_api.get_sol_price() if solana_api else None
        regression_data = []

        for project in dataset:
            try:
                symbol = project['symbol']
                project_type = project.get('dataset_type', 'solana')
                floor_price = float(project['collection_stats'].get('floorPrice', 0))  # Convert lamports to SOL
                
                if project_type == 'solana' and sol_price is not None:
                    floor_price *= sol_price  # Convert to USD if necessary

                listings_count = int(listings_data.get(symbol, 0)) if listings_data else 0
                total_supply = int(project['holder_stats'].get('totalSupply', 0))
                unique_holders = int(project['holder_stats'].get('uniqueHolders', 0))

                if floor_price <= 0 or total_supply <= 0 or unique_holders <= 0:
                    continue

                fdv = floor_price * total_supply
                fdv_per_holder = fdv / unique_holders
                listings_ratio = listings_count / total_supply
                ownership_percentage = unique_holders / total_supply

                regression_data.append({
                    'symbol': symbol,
                    'floor_price': floor_price,
                    'total_supply': total_supply,
                    'unique_holders': unique_holders,
                    'listings_count': listings_count,
                    'fdv': fdv,
                    'fdv_per_holder': fdv_per_holder,
                    'listings_ratio': listings_ratio,
                    'ownership_percentage': ownership_percentage,
                    'dataset_type': project_type
                })

            except Exception as e:
                print(f"Error preparing data for {project.get('symbol', 'Unknown')}: {str(e)}")
                continue

        return regression_data

    def _calculate_deviations(self, regression_data, model, metric):
        """Calculate deviations for a given metric using the regression model."""
        results = []
        for data in regression_data:
            try:
                x_val = np.array([[data['floor_price'] + 1e-10]])
                expected_value = np.exp(model.predict(np.log(x_val))[0])
                actual_value = data[metric]
                deviation = (actual_value - expected_value) / expected_value if expected_value > 0 else 0

                results.append({
                    **data,
                    f'expected_{metric}': expected_value,
                    f'{metric}_deviation': deviation
                })

            except Exception as e:
                continue

        return results

    def print_listings_deviations(self, results):
        """Return deviations for Listings Ratio analysis as HTML."""
        if not results:
            return html.Div("No results to display")

        deviation_info = []
        for result in results:
            symbol = result.get('symbol', 'Unknown')
            listings_deviation = result.get('listings_deviation', 0)
            expected_listings_ratio = result.get('expected_listings_ratio', 0)
            actual_listings_ratio = result.get('listings_ratio', 0)

            deviation_info.append(html.Div([
                html.H5(f"Project: {symbol}"),
                html.P(f"Listings Deviation: {listings_deviation:.4f} ({listings_deviation*100:.1f}%)"),
                html.P(f"Expected Listings Ratio: {expected_listings_ratio:.4f}"),
                html.P(f"Actual Listings Ratio: {actual_listings_ratio:.4f}"),
                html.Hr()
            ]))

        return html.Div(deviation_info)

    def print_ownership_deviations(self, results):
        """Return deviations for Ownership analysis as HTML."""
        if not results:
            return html.Div("No results to display")

        deviation_info = []
        for result in results:
            symbol = result.get('symbol', 'Unknown')
            ownership_deviation = result.get('ownership_deviation', 0)
            expected_ownership_percentage = result.get('expected_ownership_percentage', 0)
            actual_ownership_percentage = result.get('ownership_percentage', 0)

            deviation_info.append(html.Div([
                html.H5(f"Project: {symbol}"),
                html.P(f"Ownership Deviation: {ownership_deviation:.4f} ({ownership_deviation*100:.1f}%)"),
                html.P(f"Expected Ownership Percentage: {expected_ownership_percentage:.4f}"),
                html.P(f"Actual Ownership Percentage: {actual_ownership_percentage:.4f}"),
                html.Hr()
            ]))

        return html.Div(deviation_info)

    def _calculate_power_deviation(self, point, regression_results, analysis_type='fdv'):
        """Calculate relative power deviation and expected value for a given data point."""
        try:
            # Get coefficients from regression results
            a = regression_results['power']['coefficients'][0]  # 0.071571
            b = regression_results['power']['coefficients'][1]  # 0.157552
            
            # Get the correct x value based on analysis type
            if analysis_type == 'fdv':
                x_value = point.get('fdv', 0)
                actual_value = point.get('fdv_per_holder', 0)
            elif analysis_type == 'listings':
                x_value = point.get('floor_price', 0)
                actual_value = point.get('listings_ratio', 0) * 100  # Convert to percentage
            else:  # ownership
                x_value = point.get('fdv', 0)
                actual_value = point.get('ownership_percentage', 0) * 100  # Convert to percentage
            
            # Calculate expected value using power regression formula: y = ax^b
            if x_value > 0:
                expected_value = float(a * (x_value ** b))
                
                # Store expected value in point with proper key
                if analysis_type == 'fdv':
                    point['expected_fdv_per_holder'] = expected_value
                elif analysis_type == 'listings':
                    point['expected_listings_ratio'] = expected_value / 100  # Convert back from percentage
                else:
                    point['expected_ownership_ratio'] = expected_value / 100  # Convert back from percentage
                
                # Calculate relative deviation
                relative_deviation = (actual_value - expected_value) / expected_value if expected_value != 0 else 1
            else:
                expected_value = 0
                relative_deviation = 1
                
            return relative_deviation, expected_value
            
        except Exception as e:
            print(f"ERROR in _calculate_power_deviation: {str(e)}")
            print(f"Full error details: {traceback.format_exc()}")
            return 0, 0

    def _analyze_ownership(self, dataset):
        """Analyze ownership percentage trends using power regression."""
        print("\n=== OWNERSHIP ANALYSIS DETAILED DEBUG ===")
        
        # Step 1: Data Validation and Deduplication
        seen_symbols = set()
        valid_data = []
        invalid_data = []
        
        print("\nValidating input data...")
        for project in dataset:
            symbol = project.get('symbol', 'UNKNOWN')
            
            # Debug print project data
            print(f"\nProcessing project: {symbol}")
            print(f"Raw project data: {json.dumps(project, indent=2)}")
            
            # Skip duplicates
            if symbol in seen_symbols:
                print(f"Skipping duplicate project: {symbol}")
                continue
            
            seen_symbols.add(symbol)
            
            # Extract and validate required fields
            try:
                holder_count = project['holder_stats'].get('uniqueHolders', 0)
                total_supply = project.get('total_supply', 0)
                floor_price = project.get('floor_price', 0)
                fdv = project.get('fdv', 0)
                
                print(f"Extracted values:")
                print(f"  holder_count: {holder_count}")
                print(f"  total_supply: {total_supply}")
                print(f"  floor_price: {floor_price}")
                print(f"  fdv: {fdv}")
                
                # Validate data
                if holder_count <= 0:
                    invalid_data.append((symbol, "Zero or negative holder count"))
                    continue
                if total_supply <= 0:
                    invalid_data.append((symbol, "Zero or negative total supply"))
                    continue
                if floor_price <= 0:
                    invalid_data.append((symbol, "Zero or negative floor price"))
                    continue
                if fdv <= 0:
                    invalid_data.append((symbol, "Zero or negative FDV"))
                    continue
                    
                ownership_percentage = holder_count / total_supply
                
                valid_data.append({
                    'symbol': symbol,
                    'fdv': fdv,
                    'ownership_percentage': ownership_percentage,
                    'raw_data': project  # Keep original data for reference
                })
                
                print(f"Successfully validated. Ownership percentage: {ownership_percentage:.4f}")
                
            except Exception as e:
                print(f"Error processing project {symbol}: {str(e)}")
                invalid_data.append((symbol, str(e)))
                continue
        
        # Print validation summary
        print("\n=== Data Validation Summary ===")
        print(f"Total projects processed: {len(dataset)}")
        print(f"Valid projects: {len(valid_data)}")
        print(f"Invalid projects: {len(invalid_data)}")
        
        if invalid_data:
            print("\nInvalid Project Details:")
            for symbol, reason in invalid_data:
                print(f"- {symbol}: {reason}")
        
        if len(valid_data) < 2:
            print("\nError: Not enough valid data points for regression analysis")
            return None
        
        # Prepare regression data
        X = np.array([p['fdv'] for p in valid_data]).reshape(-1, 1)
        y = np.array([p['ownership_percentage'] for p in valid_data])
        
        print("\n=== Regression Input Data ===")
        print("FDV values:", X.flatten())
        print("Ownership percentages:", y)
        
        try:
            # Perform power regression with error handling
            log_X = np.log(X)
            log_y = np.log(y)
            
            print("\nLog-transformed data:")
            print("log(FDV):", log_X.flatten())
            print("log(ownership):", log_y)
            
            power_model = LinearRegression()
            power_model.fit(log_X, log_y)
            
            a = np.exp(power_model.intercept_)
            b = power_model.coef_[0]
            
            power_expected = a * np.power(X.flatten(), b)
            r2 = r2_score(y, power_expected)
            mae = mean_absolute_error(y, power_expected)
            
            print("\n=== Regression Results ===")
            print(f"Coefficient a: {a:.6f}")
            print(f"Exponent b: {b:.6f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
            
            return {
                'power': {
                    'coefficients': [a, b],
                    'r2_score': r2,
                    'mae': mae,
                    'expected_values': power_expected,
                    'valid_data': valid_data  # Include validated data for reference
                }
            }
            
        except Exception as e:
            print(f"\nError during regression analysis: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            return None

    def analyze_memecoin_dataset(self, dataset):
        """Perform specialized analysis for memecoin datasets."""
        print("\n=== Starting Memecoin Dataset Analysis ===")
        
        data_points = []
        analyses = {}
        
        for project in dataset:
            try:
                # Extract basic NFT stats
                holder_stats = project.get('holder_stats', {})
                collection_stats = project.get('collection_stats', {})
                memecoin_stats = project.get('memecoin_stats', {})
                
                if not all([holder_stats, collection_stats, memecoin_stats]):
                    print(f"Skipping project {project.get('symbol')}: Missing stats")
                    continue
                
                # Extract key metrics
                nft_floor_price = collection_stats.get('floorPrice', 0)
                total_supply = holder_stats.get('totalSupply', 0)
                unique_holders = holder_stats.get('uniqueHolders', 0)
                token_fdv = memecoin_stats.get('data', {}).get('fdv', 0)
                
                if any(v <= 0 for v in [nft_floor_price, total_supply, unique_holders, token_fdv]):
                    print(f"Skipping project {project.get('symbol')}: Invalid values")
                    continue
                
                # Calculate advanced metrics
                nft_fdv = nft_floor_price * total_supply
                fdv_ratio = token_fdv / nft_fdv if nft_fdv > 0 else 0
                token_holder_value = token_fdv / unique_holders if unique_holders > 0 else 0
                
                data_point = {
                    'symbol': project.get('symbol'),
                    'nft_floor_price': nft_floor_price,
                    'token_fdv': token_fdv,
                    'nft_fdv': nft_fdv,
                    'fdv_ratio': fdv_ratio,
                    'token_holder_value': token_holder_value,
                    'holder_stats': holder_stats,
                    'total_supply': total_supply
                }
                
                data_points.append(data_point)
                
            except Exception as e:
                print(f"Error processing project {project.get('symbol')}: {str(e)}")
                continue
        
        if not data_points:
            print("No valid data points found for analysis")
            return None
        
        # Perform regression analysis on FDV ratios
        X = np.array([p['nft_floor_price'] for p in data_points]).reshape(-1, 1)
        y = np.array([p['fdv_ratio'] for p in data_points])
        
        analyses['fdv_ratio'] = self._perform_regression(X, y)
        
        # Analyze token holder value correlation
        X_holder = np.array([p['nft_floor_price'] for p in data_points]).reshape(-1, 1)
        y_holder = np.array([p['token_holder_value'] for p in data_points])
        
        analyses['token_holder_value'] = self._perform_regression(X_holder, y_holder)
        
        return {
            'data_points': data_points,
            'analyses': analyses,
            'metrics': {
                'avg_fdv_ratio': np.mean([p['fdv_ratio'] for p in data_points]),
                'median_fdv_ratio': np.median([p['fdv_ratio'] for p in data_points]),
                'avg_token_holder_value': np.mean([p['token_holder_value'] for p in data_points]),
                'median_token_holder_value': np.median([p['token_holder_value'] for p in data_points])
            }
        }

    def _perform_log_linear_regression(self, X, y):
        """Perform log-linear regression where X is log-transformed but y remains untransformed."""
        # Add small constant to avoid log(0)
        X = X + 1e-10
        
        # Transform X using natural log
        X_log = np.log(X).reshape(-1, 1)
        
        # Fit linear regression on log-transformed X
        model = LinearRegression()
        model.fit(X_log, y)
        
        # Calculate predicted values
        y_pred = model.predict(X_log)
        
        # Calculate regression metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Debugging: Print calculated metrics
        print(f"[DEBUG] Log-Linear Regression metrics - R²: {r2}, MAE: {mae}, RMSE: {rmse}")
        
        return {
            'coefficients': [model.intercept_, model.coef_[0]],  # [intercept, slope]
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'expected_values': y_pred.tolist()
        }

    def analyze_all(self, dataset, listings_data, solana_api=None):
        """Perform all types of analysis and combine results."""
        print("\n=== Combined NFT Analysis ===")
        
        # Perform FDV/Holder analysis
        fdv_results = self.analyze_fdv_holder(dataset, solana_api)
        print("[DEBUG] FDV/Holder analysis complete.")
        
        # Perform Listings Ratio analysis
        listings_results = self.analyze_listings_ratio(dataset, listings_data, solana_api)
        print("[DEBUG] Listings Ratio analysis complete.")
        
        # Perform Ownership analysis
        ownership_results = self.analyze_ownership(dataset, solana_api)
        print("[DEBUG] Ownership analysis complete.")
        
        # Combine results
        combined_results = []
        for fdv_result in fdv_results:
            symbol = fdv_result['symbol']
            combined_result = fdv_result.copy()
            
            # Find corresponding results in other analyses
            listings_result = next((res for res in listings_results if res['symbol'] == symbol), None)
            ownership_result = next((res for res in ownership_results if res['symbol'] == symbol), None)
            
            if listings_result:
                combined_result.update({
                    'listings_ratio': listings_result.get('listings_ratio', 0),
                    'listings_deviation': listings_result.get('listings_deviation', 0),
                    'expected_listings_ratio': listings_result.get('expected_listings_ratio', 0),
                    'listings_analysis': listings_result.get('listings_analysis', {})  # Ensure correct key
                })
            
            if ownership_result:
                combined_result.update({
                    'ownership_percentage': ownership_result.get('ownership_percentage', 0),
                    'ownership_deviation': ownership_result.get('ownership_deviation', 0),
                    'expected_ownership_percentage': ownership_result.get('expected_ownership_percentage', 0),
                    'ownership_analysis': ownership_result.get('ownership_analysis', {})  # Ensure correct key
                })
            
            # Include FDV analysis
            combined_result['fdv_analysis'] = fdv_result.get('fdv_analysis', {})  # Ensure correct key
            
            combined_results.append(combined_result)
        
        # Debugging: Print combined results to verify structure
        print("[DEBUG] Combined Analysis Results:", combined_results)
        
        return combined_results

def plot_combined_results(combined_results, listings_weight=2, fdv_weight=3, ownership_weight=2):
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
        
        # Calculate the weighted average deviation
        total_weight = fdv_weight + listings_weight + ownership_weight
        combined_deviation = (
            (fdv_deviation * fdv_weight) +
            (listings_deviation * listings_weight) +
            (ownership_deviation * ownership_weight)
        ) / total_weight
        
        result['combined_deviation'] = combined_deviation
        print(f"[DEBUG] Combined deviation for {result.get('symbol', 'Unknown')}: {combined_deviation}")

    # Sort results by combined deviation
    combined_results.sort(key=lambda x: x['combined_deviation'], reverse=True)

    # Create a summary of combined deviations
    combined_deviation_info = html.Div([
        html.H4("Combined Deviation Summary"),
        html.Ul([
            html.Li(f"{result.get('symbol', 'Unknown')}: {result['combined_deviation']:.4f}")
            for result in combined_results
        ]),
        html.Hr()
    ])

    # Create subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=("FDV per Holder", "Listings Ratio", "Ownership Percentage"))

    # Plot FDV per Holder
    fig.add_trace(go.Scatter(
        x=floor_prices,
        y=fdv_per_holders,
        mode='markers',
        name='FDV per Holder'
    ), row=1, col=1)

    # Plot Listings Ratio
    fig.add_trace(go.Scatter(
        x=floor_prices,
        y=listings_ratios,
        mode='markers',
        name='Listings Ratio'
    ), row=2, col=1)

    # Plot Ownership Percentage
    fig.add_trace(go.Scatter(
        x=floor_prices,
        y=ownership_percentages,
        mode='markers',
        name='Ownership Percentage'
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Combined Analysis Results',
        xaxis=dict(type='log', title='Floor Price'),
        yaxis=dict(title='FDV per Holder'),
        xaxis2=dict(type='log', title='Floor Price'),
        yaxis2=dict(title='Listings Ratio'),
        xaxis3=dict(type='log', title='Floor Price'),
        yaxis3=dict(title='Ownership Percentage'),
        height=900
    )

    # Generate deviation information
    fdv_deviation_info = regression_analyzer.print_fdv_deviations(combined_results)
    listings_deviation_info = regression_analyzer.print_listings_deviations(combined_results)
    ownership_deviation_info = regression_analyzer.print_ownership_deviations(combined_results)
    print("[DEBUG] Deviation information generated")

    return html.Div([
        combined_deviation_info,  # Display combined deviation summary at the top
        dcc.Graph(figure=fig),
        fdv_deviation_info,
        listings_deviation_info,
        ownership_deviation_info
    ])