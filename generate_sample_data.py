import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def generate_battery_discharge_data(temperature, num_points=2000):
    """
    Generate realistic primary battery discharge data
    
    Parameters:
    - temperature: Test temperature in Celsius
    - num_points: Number of data points to generate
    """
    # Time: 0 to 3000 seconds (10ms intervals would be 300k points, 
    # but we'll sample 2000 points for a manageable file size)
    time = np.linspace(0, 3000, num_points)
    
    # Temperature effects on battery performance
    temp_factor = {
        25: 1.0,      # Room temperature - best performance
        0: 0.85,      # Cold - reduced capacity
        -20: 0.65     # Very cold - significantly reduced capacity
    }.get(temperature, 1.0)
    
    # Voltage decay (starts at ~1.5V for alkaline battery, drops to ~0.9V)
    initial_voltage = 1.5
    final_voltage = 0.9
    
    # Voltage curve with temperature-dependent decay rate
    decay_rate = 0.0002 * (1.0 / temp_factor)
    voltage = initial_voltage - (initial_voltage - final_voltage) * (1 - np.exp(-decay_rate * time))
    
    # Add some realistic noise
    voltage += np.random.normal(0, 0.005, num_points)
    
    # Current (constant discharge at 100mA with small variations)
    current = 0.1 + np.random.normal(0, 0.002, num_points)
    current = np.maximum(current, 0)  # No negative current
    
    # Capacity (mAh) - accumulates over time, affected by temperature
    # Capacity = Current (A) * Time (hours) * 1000 (to mAh)
    time_hours = time / 3600
    capacity = np.cumsum(current * np.diff(time_hours, prepend=0)) * 1000 * temp_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time (s)': time,
        'Voltage (V)': voltage,
        'Current (A)': current,
        'Capacity (mAh)': capacity
    })
    
    return df

# Generate data for three temperatures
print("Generating sample battery discharge data...")

temperatures = [25, 0, -20]
datasets = {}

for temp in temperatures:
    print(f"  Creating data for {temp}°C...")
    datasets[f'Battery_Test_{temp}C'] = generate_battery_discharge_data(temp)

# Create Excel file with multiple sheets
print("\nCreating Excel file...")
with pd.ExcelWriter('sample_battery_data.xlsx', engine='openpyxl') as writer:
    for sheet_name, df in datasets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n✅ Sample Excel file created: sample_battery_data.xlsx")
print(f"   - 3 sheets (25°C, 0°C, -20°C)")
print(f"   - {len(datasets['Battery_Test_25C'])} data points per test")
print(f"   - Time range: 0-3000 seconds")
print(f"   - Columns: Time (s), Voltage (V), Current (A), Capacity (mAh)")
print("\nYou can now upload this file to the app to test the analysis features!")
