import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

def simple_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Sine Wave', color='blue')
    plt.title('Simple Sine Wave Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()  # This should display the plot

if __name__ == "__main__":
    simple_plot()