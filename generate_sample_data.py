import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def generate_battery_discharge_data(battery_code, temperature, build_id, duration_seconds):
    """
    Generate realistic primary battery discharge data
    
    Parameters:
    - battery_code: Battery identifier (e.g., "ALK-AA-001")
    - temperature: Test temperature in Celsius
    - build_id: Build identifier (e.g., "Build_A", "Build_B")
    - duration_seconds: Test duration in seconds
    """
    # 10ms intervals = 0.01 seconds
    interval_seconds = 0.01
    num_points = int(duration_seconds / interval_seconds)
    
    # Time in seconds (10ms intervals)
    time = np.arange(0, duration_seconds, interval_seconds)[:num_points]
    
    # Temperature effects on battery performance
    temp_factor = {
        25: 1.0,      # Room temperature - best performance
        0: 0.85,      # Cold - reduced capacity
        -20: 0.65     # Very cold - significantly reduced capacity
    }.get(temperature, 1.0 - (25 - temperature) * 0.01)
    
    # Voltage decay (starts at ~1.5V for alkaline battery, drops to ~0.9V)
    initial_voltage = 1.5
    final_voltage = 0.9
    
    # Voltage curve with temperature-dependent decay rate
    decay_rate = 0.0002 * (1.0 / max(temp_factor, 0.1))
    voltage = initial_voltage - (initial_voltage - final_voltage) * (1 - np.exp(-decay_rate * time))
    
    # Add realistic noise
    voltage += np.random.normal(0, 0.005, num_points)
    voltage = np.maximum(voltage, 0.5)  # Floor at 0.5V
    
    # Current (constant discharge at 100mA with small variations)
    current = 0.1 + np.random.normal(0, 0.002, num_points)
    current = np.maximum(current, 0)  # No negative current
    
    # Capacity (mAh) - accumulates over time, affected by temperature
    time_hours = time / 3600
    capacity = np.cumsum(current * np.diff(time_hours, prepend=0)) * 1000 * temp_factor
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time (s)': time,
        'Voltage (V)': voltage,
        'Current (A)': current,
        'Capacity (mAh)': capacity
    })
    
    # Create metadata
    metadata = {
        'battery_code': battery_code,
        'temperature': f"{temperature}°C",
        'build_id': build_id
    }
    
    return df, metadata

# Generate test scenarios
print("Generating sample battery discharge data with metadata...")
print("=" * 60)

test_scenarios = [
    # Different batteries at same temperature (25°C)
    {'battery_code': 'ALK-AA-001', 'temperature': 25, 'build_id': 'Build_A', 'duration': 3000},
    {'battery_code': 'ALK-AA-002', 'temperature': 25, 'build_id': 'Build_B', 'duration': 3000},
    
    # Same battery at different temperatures
    {'battery_code': 'ALK-AA-003', 'temperature': 25, 'build_id': 'Build_C', 'duration': 3000},
    {'battery_code': 'ALK-AA-003', 'temperature': 0, 'build_id': 'Build_C', 'duration': 600},
    {'battery_code': 'ALK-AA-003', 'temperature': -20, 'build_id': 'Build_C', 'duration': 100},
    
    # Different durations
    {'battery_code': 'ALK-AA-004', 'temperature': 25, 'build_id': 'Build_D', 'duration': 100},
    {'battery_code': 'ALK-AA-005', 'temperature': 0, 'build_id': 'Build_E', 'duration': 600},
]

# Create workbook
wb = Workbook()
wb.remove(wb.active)  # Remove default sheet

for scenario in test_scenarios:
    df, metadata = generate_battery_discharge_data(
        scenario['battery_code'],
        scenario['temperature'],
        scenario['build_id'],
        scenario['duration']
    )
    
    # Create sheet name
    sheet_name = f"{scenario['build_id']}_{scenario['temperature']}C"[:31]  # Excel limit
    ws = wb.create_sheet(title=sheet_name)
    
    # Write metadata rows
    ws.append(['Battery Code', metadata['battery_code']])
    ws.append(['Temperature', metadata['temperature']])
    ws.append(['Build ID', metadata['build_id']])
    ws.append([])  # Blank row
    
    # Write data
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=5):
        ws.append(row)
    
    print(f"✓ Created sheet: {sheet_name}")
    print(f"  - Battery: {metadata['battery_code']}")
    print(f"  - Temperature: {metadata['temperature']}")
    print(f"  - Build ID: {metadata['build_id']}")
    print(f"  - Duration: {scenario['duration']}s ({len(df):,} data points @ 10ms intervals)")
    print()

# Save workbook
filename = 'sample_battery_data_with_metadata.xlsx'
wb.save(filename)

print("=" * 60)
print(f"✅ Sample Excel file created: {filename}")
print(f"\nFile Structure:")
print("  - Each sheet has metadata at the top (rows 1-3)")
print("  - Data table starts at row 5 with headers")
print("  - 10ms sampling intervals (0.01 seconds)")
print("  - Variable test durations: 100s, 600s, 3000s")
print("\nTest Scenarios Included:")
print("  • Multiple batteries at same temperature (compare builds)")
print("  • Same battery at different temperatures (temperature analysis)")
print("  • Different test durations (100s, 600s, 3000s)")
print("\nYou can now upload this file to the app to test all features!")
