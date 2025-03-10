class CustomRegressionAnalyzer:
    def __init__(self):
        self.custom_equations = []
        self.results = []
        
    def add_custom_equation(self, equation_type, coefficients):
        """Add a custom regression equation."""
        if equation_type not in ['power', 'polynomial']:
            print("Invalid equation type. Must be 'power' or 'polynomial'.")
            return False

        if equation_type == 'power' and len(coefficients) != 2:
            print("Power equation requires exactly 2 coefficients (a, b).")
            return False

        if equation_type == 'polynomial' and len(coefficients) != 3:
            print("Polynomial equation requires exactly 3 coefficients (a, b, c).")
            return False

        self.custom_equations.append({
            'type': equation_type,
            'coefficients': coefficients
        })
        print(f"Custom {equation_type} equation added with coefficients: {coefficients}")
        return True

    def list_custom_equations(self):
        """List all stored custom equations."""
        if not self.custom_equations:
            print("No custom equations stored.")
            return

        print("\nStored Custom Equations:")
        for idx, eq in enumerate(self.custom_equations, 1):
            print(f"{idx}. Type: {eq['type']}, Coefficients: {eq['coefficients']}")

    def get_custom_equations(self):
        """Return the list of custom equations."""
        return self.custom_equations
    
    def select_custom_equation(self):
        """Allow user to select a custom equation."""
        if not self.custom_equations:
            print("No custom equations stored.")
            return None

        print("\nSelect a custom equation:")
        for idx, eq in enumerate(self.custom_equations, 1):
            print(f"{idx}. Type: {eq['type']}, Coefficients: {eq['coefficients']}")

        choice = input("Enter the number of the equation to use: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(self.custom_equations):
                return self.custom_equations[index]
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None