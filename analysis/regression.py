import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your environment
import matplotlib.pyplot as plt

class RegressionAnalyzer:
    def __init__(self):
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.linear_model = LinearRegression()
        # Define internal filters
        self.filters = {
            '3-week': -100000,   # Example: Only include tokens with a 3-week price increase > 5%
            '6-week': -8000,      # Example: Only include tokens with a 6-week price decrease < -10%
            '3-month': -100,        # Example: Only include tokens with a 3-month price increase > 15%
            'fdv': 100000000000       # Example: Only include tokens with an FDV > 1,000,000
        }

    def analyze_dataset(self, dataset, analysis_type='all'):
        """Perform regression analysis on a dataset."""
        try:
            # Extract data points
            data_points = []
            for project in dataset:
                data = project.get('data', {})
                fdv = data.get('fdv')
                fdv_per_holder = data.get('fdv_per_holder')

                data_points.append({
                    'name': data.get('name', 'Unknown'),
                    'symbol': data.get('symbol', 'Unknown'),
                    'token_address': project['token_address'],
                    'fdv': fdv,
                    'fdv_per_holder': fdv_per_holder,
                })

            # Filter out data points with None values for fdv or fdv_per_holder and non-positive fdv
            data_points = [p for p in data_points if p['fdv'] is not None and p['fdv_per_holder'] is not None and p['fdv'] > 0 and p['fdv_per_holder'] > 0]

            if not data_points:
                print("No valid data points found after filtering.")
                return None

            X = np.array([p['fdv'] for p in data_points]).reshape(-1, 1)
            y = np.array([p['fdv_per_holder'] for p in data_points])

            # Check for NaN values in X and y
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print("NaN values found in input data.")
                return None

            results = {'data_points': data_points}

            # Linear Regression
            if analysis_type in ['linear', 'all']:
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                linear_expected = linear_model.predict(X)

                results['linear'] = {
                    'coefficients': [linear_model.intercept_, linear_model.coef_[0]],
                    'r2_score': r2_score(y, linear_expected),
                    'mae': mean_absolute_error(y, linear_expected),
                    'expected_values': linear_expected
                }

            # Logarithmic Regression
            if analysis_type in ['logarithmic', 'all']:
                try:
                    log_X = np.log(X)
                    log_model = LinearRegression()
                    log_model.fit(log_X, y)
                    log_expected = log_model.predict(log_X)

                    results['logarithmic'] = {
                        'coefficients': [log_model.intercept_, log_model.coef_[0]],
                        'r2_score': r2_score(y, log_expected),
                        'mae': mean_absolute_error(y, log_expected),
                        'expected_values': log_expected
                    }
                except Exception as e:
                    print(f"Warning: Logarithmic regression failed: {str(e)}")

            # Polynomial Regression
            if analysis_type in ['polynomial', 'all']:
                X_poly = self.poly_features.fit_transform(X)
                self.linear_model.fit(X_poly, y)
                poly_expected = self.linear_model.predict(X_poly)

                results['polynomial'] = {
                    'coefficients': [self.linear_model.intercept_] + list(self.linear_model.coef_),
                    'r2_score': r2_score(y, poly_expected),
                    'mae': mean_absolute_error(y, poly_expected),
                    'expected_values': poly_expected
                }

            # Power Regression (y = ax^b)
            if analysis_type in ['power', 'all']:
                try:
                    # Ensure no zero or negative values before log transformation
                    if np.any(X <= 0) or np.any(y <= 0):
                        print("Warning: Non-positive values found, skipping power regression.")
                    else:
                        log_X = np.log(X)
                        log_y = np.log(y)
                        power_model = LinearRegression()
                        power_model.fit(log_X, log_y)
                        a = np.exp(power_model.intercept_)
                        b = power_model.coef_[0]
                        power_expected = a * np.power(X.flatten(), b)

                        # Compute R² in log-space
                        log_power_expected = power_model.predict(log_X)
                        r2_log_space = r2_score(log_y, log_power_expected)

                        # Debugging: Print intermediate values
                        print(f"Power Regression: a = {a}, b = {b}")
                        print(f"Predicted values: {power_expected}")

                        results['power'] = {
                            'coefficients': [a, b],
                            'r2_score': r2_log_space,  # Use log-space R²
                            'mae': mean_absolute_error(y, power_expected),
                            'rmse': np.sqrt(mean_squared_error(y, power_expected)),
                            'expected_values': power_expected
                        }
                except Exception as e:
                    print(f"Warning: Power regression failed: {str(e)}")

            # Ridge Regression
            if analysis_type in ['ridge', 'all']:
                ridge_model = Ridge()
                ridge_model.fit(X, y)
                ridge_expected = ridge_model.predict(X)

                results['ridge'] = {
                    'coefficients': [ridge_model.intercept_, ridge_model.coef_[0]],
                    'r2_score': r2_score(y, ridge_expected),
                    'mae': mean_absolute_error(y, ridge_expected),
                    'expected_values': ridge_expected
                }

            # Add predictions to data points
            for i, point in enumerate(data_points):
                if 'linear' in results:
                    point['linear_expected'] = results['linear']['expected_values'][i]
                    point['linear_deviation'] = ((point['fdv_per_holder'] - point['linear_expected']) / point['linear_expected']) * 100
                if 'polynomial' in results:
                    point['poly_expected'] = results['polynomial']['expected_values'][i]
                    point['poly_deviation'] = ((point['fdv_per_holder'] - point['poly_expected']) / point['poly_expected']) * 100
                if 'power' in results:
                    point['power_expected'] = results['power']['expected_values'][i]
                    point['power_deviation'] = ((point['fdv_per_holder'] - point['power_expected']) / point['power_expected']) * 100
                if 'ridge' in results:
                    point['ridge_expected'] = results['ridge']['expected_values'][i]
                    point['ridge_deviation'] = ((point['fdv_per_holder'] - point['ridge_expected']) / point['ridge_expected']) * 100
                if 'logarithmic' in results:
                    point['log_expected'] = results['logarithmic']['expected_values'][i]
                    point['log_deviation'] = ((point['fdv_per_holder'] - point['log_expected']) / point['log_expected']) * 100

            return results

        except Exception as e:
            print(f"Error in regression analysis: {str(e)}")
            return None

    def print_regression_only(self, results):
        """Return regression equations and R² values as a string."""
        output = "\nRegression Analysis Results:\n" + "-" * 50

        if 'linear' in results:
            coef = results['linear']['coefficients']
            output += f"\nLinear Regression:\n"
            output += f"Equation: y = {coef[1]:.4e}x + {coef[0]:.4e}\n"
            output += f"R² Score: {results['linear']['r2_score']:.4f}\n"
            output += f"MAE: {results['linear']['mae']:.4f}\n"

        if 'polynomial' in results:
            coef = results['polynomial']['coefficients']
            output += f"\nPolynomial Regression:\n"
            output += f"Equation: y = {coef[2]:.4e}x² + {coef[1]:.4e}x + {coef[0]:.4e}\n"
            output += f"R² Score: {results['polynomial']['r2_score']:.4f}\n"
            output += f"MAE: {results['polynomial']['mae']:.4f}\n"

        if 'power' in results:
            a, b = results['power']['coefficients']
            output += f"\nPower Regression:\n"
            output += f"Equation: y = {a:.4f}x^{b:.4f}\n"
            output += f"R² Score: {results['power']['r2_score']:.4f}\n"
            output += f"MAE: {results['power']['mae']:.4f}\n"

        if 'ridge' in results:
            coef = results['ridge']['coefficients']
            output += f"\nRidge Regression:\n"
            output += f"Equation: y = {coef[1]:.4e}x + {coef[0]:.4e}\n"
            output += f"R² Score: {results['ridge']['r2_score']:.4f}\n"
            output += f"MAE: {results['ridge']['mae']:.4f}\n"

        if 'logarithmic' in results:
            coef = results['logarithmic']['coefficients']
            output += f"\nLogarithmic Regression:\n"
            output += f"Equation: y = {coef[1]:.4e} * log(x) + {coef[0]:.4e}\n"
            output += f"R² Score: {results['logarithmic']['r2_score']:.4f}\n"
            output += f"MAE: {results['logarithmic']['mae']:.4f}\n"

        return output

    def plot_regression_results(self, results):
        """Plot FDV vs FDV per Holder with regression lines and visualize residuals."""
        data_points = results['data_points']
        
        X = np.array([p['fdv'] for p in data_points]).reshape(-1, 1)
        y = np.array([p['fdv_per_holder'] for p in data_points])

        plt.figure(figsize=(30, 18))

        # Generate a range of FDV values for smooth plotting
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

        # Scatter plot of actual data points
        plt.scatter(X, y, color='purple', label='Data Points')

        # Add labels to each point
        for i, point in enumerate(data_points):
            plt.annotate(point['name'], (X[i], y[i]), fontsize=8, ha='right')

        # Polynomial Regression Line
        X_poly_range = self.poly_features.fit_transform(X_range)
        poly_expected = self.linear_model.predict(X_poly_range)
        plt.plot(X_range, poly_expected, color='red', label='Polynomial Regression', linewidth=2)

        # Power Regression Line - FIXED
        log_X = np.log(X)  # Take log of original data points
        log_y = np.log(y)  # Take log of original y values
        power_model = LinearRegression()
        power_model.fit(log_X, log_y)
        a = np.exp(power_model.intercept_)
        b = power_model.coef_[0]
        power_expected = a * np.power(X_range.flatten(), b)
        plt.plot(X_range, power_expected, color='green', label='Power Regression', linewidth=2)

        # Set plot labels and scales
        plt.title('FDV vs FDV per Holder')
        plt.xlabel('FDV (Floor Price * Remaining Supply)')
        plt.ylabel('FDV per Holder')
        plt.xscale('log')
        plt.yscale('log')

        # Add grid lines for both major and minor ticks
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.minorticks_on()  # Enable minor ticks

        # Calculate and set x-axis ticks
        min_exp = int(np.floor(np.log10(X.min())))
        max_exp = int(np.ceil(np.log10(X.max())))
        ticks = [10**i for i in range(min_exp, max_exp + 1)]
        tick_labels = [f'{int(tick):,}' for tick in ticks]
        plt.xticks(ticks, tick_labels)
        plt.xlim(10**min_exp * 0.8, 10**max_exp * 1.2)

        # Set y-axis ticks
        plt.yticks([100, 1000, 10000, 100000], ['100', '1,000', '10,000', '100,000'])

        plt.legend()
        plt.grid()
        plt.show()

    def print_full_analysis(self, results):
        """Print full analysis including individual token comparisons."""
        final_results = results['data_points'].copy()

        # Sort results
        def sortfunc(point):
            return point['fdv']
        final_results.sort(reverse=True, key=sortfunc)

        # Print individual token results
        print("\nToken Analysis:")
        for point in final_results:
            print(f"\nToken: {point['name']} ({point['symbol']})")
            print(f"\nFDV: {point['fdv']}")
            print(f"Address: {point['token_address']}")
            print(f"Actual FDV per Holder: ${point['fdv_per_holder']:,.2f}")

            # Print price changes
            print("\nPrice Changes:")
            for period in ["24-HR", "48-HR", "72-HR", "96-HR", "1-week", "2-week", "3-week", "4-week", "5-week", "6-week", "3-month", "6-month"]:
                change = point.get(period)
                if change is not None:
                    print(f"  {period}: {change:.2f}%")

            # Print regression results
            if 'linear_expected' in point:
                print(f"Linear Expected: ${point['linear_expected']:,.2f}")
                print(f"Linear Deviation: {point['linear_deviation']:,.2f}%")
            if 'poly_expected' in point:
                print(f"Polynomial Expected: ${point['poly_expected']:,.2f}")
                print(f"Polynomial Deviation: {point['poly_deviation']:,.2f}%")
            if 'power_expected' in point:
                print(f"Power Expected: ${point['power_expected']:,.2f}")
                print(f"Power Deviation: {point['power_deviation']:,.2f}%")
            if 'ridge_expected' in point:
                print(f"Ridge Expected: ${point['ridge_expected']:,.2f}")
                print(f"Ridge Deviation: {point['ridge_deviation']:,.2f}%")
            print("-" * 30)

        # Print regression equations at the end
        print(self.print_regression_only(results))

        # Generate the plot after printing all results
        print("\nGenerating plot...")
        self.plot_regression_results(results)