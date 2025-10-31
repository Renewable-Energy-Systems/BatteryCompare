import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def generate_battery_discharge_data(battery_code, temperature, build_id, duration_seconds, downsample=10):
    """
    Generate realistic primary battery discharge data
    
    Parameters:
    - battery_code: Battery identifier
    - temperature: Test temperature in Celsius
    - build_id: Build identifier
    - duration_seconds: Test duration in seconds
    - downsample: Keep every Nth point (for faster file generation)
    """
    # 10ms intervals = 0.01 seconds
    interval_seconds = 0.01 * downsample
    num_points = int(duration_seconds / interval_seconds)
    
    # Time in seconds
    time = np.arange(0, duration_seconds, interval_seconds)[:num_points]
    
    # Temperature effects
    temp_factor = 1.0 - (25 - temperature) * 0.01 if temperature < 25 else 1.0
    temp_factor = max(temp_factor, 0.5)
    
    # Voltage decay
    initial_voltage = 1.5
    final_voltage = 0.9
    decay_rate = 0.0002 * (1.0 / max(temp_factor, 0.1))
    voltage = initial_voltage - (initial_voltage - final_voltage) * (1 - np.exp(-decay_rate * time))
    voltage += np.random.normal(0, 0.005, num_points)
    voltage = np.maximum(voltage, 0.5)
    
    # Current
    current = 0.1 + np.random.normal(0, 0.002, num_points)
    current = np.maximum(current, 0)
    
    # Capacity
    time_hours = time / 3600
    capacity = np.cumsum(current * np.diff(time_hours, prepend=0)) * 1000 * temp_factor
    
    df = pd.DataFrame({
        'Time (s)': time,
        'Voltage (V)': voltage,
        'Current (A)': current,
        'Capacity (mAh)': capacity
    })
    
    metadata = {
        'battery_code': battery_code,
        'temperature': f"{temperature}°C",
        'build_id': build_id
    }
    
    return df, metadata

print("Generating quick sample battery data...")

test_scenarios = [
    {'battery_code': 'ALK-AA-001', 'temperature': 25, 'build_id': 'Build_A', 'duration': 3000},
    {'battery_code': 'ALK-AA-002', 'temperature': 25, 'build_id': 'Build_B', 'duration': 3000},
    {'battery_code': 'ALK-AA-003', 'temperature': 0, 'build_id': 'Build_C', 'duration': 600},
]

wb = Workbook()
wb.remove(wb.active)

for scenario in test_scenarios:
    df, metadata = generate_battery_discharge_data(
        scenario['battery_code'],
        scenario['temperature'],
        scenario['build_id'],
        scenario['duration'],
        downsample=30  # Keep every 30th point for smaller file
    )
    
    sheet_name = f"{scenario['build_id']}_{scenario['temperature']}C"[:31]
    ws = wb.create_sheet(title=sheet_name)
    
    # Metadata rows
    ws.append(['Battery Code', metadata['battery_code']])
    ws.append(['Temperature', metadata['temperature']])
    ws.append(['Build ID', metadata['build_id']])
    ws.append([])
    
    # Data
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)
    
    print(f"✓ {sheet_name}: {len(df):,} points")

filename = 'sample_battery_data.xlsx'
wb.save(filename)
print(f"\n✅ Created: {filename}")
