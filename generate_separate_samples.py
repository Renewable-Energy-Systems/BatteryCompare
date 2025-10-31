import pandas as pd
import numpy as np

def generate_battery_discharge_data(battery_code, temperature, build_id, duration_seconds, downsample=30):
    """Generate realistic primary battery discharge data"""
    interval_seconds = 0.01 * downsample
    num_points = int(duration_seconds / interval_seconds)
    
    time = np.arange(0, duration_seconds, interval_seconds)[:num_points]
    
    temp_factor = 1.0 - (25 - temperature) * 0.01 if temperature < 25 else 1.0
    temp_factor = max(temp_factor, 0.5)
    
    initial_voltage = 1.5
    final_voltage = 0.9
    decay_rate = 0.0002 * (1.0 / max(temp_factor, 0.1))
    voltage = initial_voltage - (initial_voltage - final_voltage) * (1 - np.exp(-decay_rate * time))
    voltage += np.random.normal(0, 0.005, num_points)
    voltage = np.maximum(voltage, 0.5)
    
    current = 0.1 + np.random.normal(0, 0.002, num_points)
    current = np.maximum(current, 0)
    
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

def save_with_metadata(df, metadata, filename):
    """Save DataFrame with metadata rows at the top"""
    # Create metadata rows
    metadata_rows = [
        ['Battery Code', metadata['battery_code']],
        ['Temperature', metadata['temperature']],
        ['Build ID', metadata['build_id']],
        []  # Blank row
    ]
    
    # Write to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Create a temporary df for metadata
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=0)
        
        # Write actual data starting after metadata
        df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=len(metadata_rows))
    
    return filename

print("Generating separate sample files...")

# Create 3 separate files
files = []

# Build A - 25°C
df1, meta1 = generate_battery_discharge_data('ALK-AA-001', 25, 'Build_A', 3000, downsample=30)
file1 = save_with_metadata(df1, meta1, 'sample_build_A_25C.xlsx')
files.append((file1, meta1, len(df1)))
print(f"✓ Created: {file1} ({len(df1):,} points)")

# Build B - 25°C (different battery, same temp)
df2, meta2 = generate_battery_discharge_data('ALK-AA-002', 25, 'Build_B', 3000, downsample=30)
file2 = save_with_metadata(df2, meta2, 'sample_build_B_25C.xlsx')
files.append((file2, meta2, len(df2)))
print(f"✓ Created: {file2} ({len(df2):,} points)")

# Build C - 0°C (different battery and temp)
df3, meta3 = generate_battery_discharge_data('ALK-AA-003', 0, 'Build_C', 600, downsample=30)
file3 = save_with_metadata(df3, meta3, 'sample_build_C_0C.xlsx')
files.append((file3, meta3, len(df3)))
print(f"✓ Created: {file3} ({len(df3):,} points)")

print(f"\n✅ Created 3 separate sample files")
print("\nUpload Instructions:")
print("  1. Upload sample_build_A_25C.xlsx to Build 1")
print("  2. Upload sample_build_B_25C.xlsx to Build 2")
print("  3. Upload sample_build_C_0C.xlsx to Build 3")
print("\nThis will test:")
print("  - Different batteries at same temperature (A & B at 25°C)")
print("  - Different temperatures (Build C at 0°C)")
