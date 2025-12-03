import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import io
from datetime import datetime
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
import json
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

DATABASE_URL = os.environ.get('DATABASE_URL')

Base = declarative_base()

class SavedComparison(Base):
    __tablename__ = 'saved_comparisons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_builds = Column(Integer)
    build_names = Column(Text)
    data_json = Column(Text)
    metrics_json = Column(Text)
    battery_type = Column(String(100), nullable=True)  # New: battery type (General, Madhava, etc.)
    extended_metadata_json = Column(Text, nullable=True)  # New: extended build metadata (weights, calorific value, etc.)
    password_hash = Column(String(128), nullable=True)  # Optional password protection
    standard_params_json = Column(Text, nullable=True)  # Standard benchmark parameters (min voltage, std activation time, etc.)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    
    # Run migrations to add new columns if they don't exist
    try:
        inspector = inspect(engine)
        if 'saved_comparisons' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('saved_comparisons')]
            
            # Add password_hash column
            if 'password_hash' not in columns:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE saved_comparisons ADD COLUMN password_hash VARCHAR(128)"))
                    conn.commit()
            
            # Add standard_params_json column
            if 'standard_params_json' not in columns:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE saved_comparisons ADD COLUMN standard_params_json TEXT"))
                    conn.commit()
    except Exception as e:
        # Migration error - log but continue to allow app to run
        pass
else:
    engine = None
    Session = None

# Initialize password hasher for secure password storage
ph = PasswordHasher()

st.set_page_config(page_title="Battery Discharge Analysis", layout="wide", initial_sidebar_state="expanded")

st.title("🔋 Battery Discharge Data Analysis")
st.markdown("Compare discharge curves and performance metrics across different builds")

def safe_scalar(value):
    """Convert pandas Series to scalar, or return value as-is if already scalar"""
    if isinstance(value, pd.Series):
        return value.iloc[0] if len(value) > 0 else None
    return value

def extract_metadata_from_file(uploaded_file):
    """
    Extract ALL metadata from Excel header: basic info, standard params, and extended build data.
    Returns: (metadata_dict, standard_params_dict, extended_metadata_dict, data_start_row)
    """
    metadata = {
        'battery_code': None,
        'temperature': None,
        'build_id': None
    }
    
    standard_params = {
        'min_activation_voltage': None,
        'std_max_oc_voltage': None,
        'std_activation_time_ms': None,
        'std_duration_sec': None
    }
    
    extended_metadata = {
        'anode_weight_per_cell': None,
        'cathode_weight_per_cell': None,
        'electrolyte_weight': None,
        'heat_pellet_weight': None,
        'calorific_value_per_gram': None,
        'cells_in_series': None,
        'stacks_in_parallel': None
    }
    
    try:
        if uploaded_file.name.endswith('.csv'):
            temp_df = pd.read_csv(uploaded_file, nrows=20, header=None)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            temp_df = pd.read_excel(uploaded_file, nrows=20, header=None)
        else:
            return metadata, standard_params, extended_metadata, 0
        
        uploaded_file.seek(0)
        
        data_start_row = 0
        found_header = False
        
        for idx, row in temp_df.iterrows():
            row_strings = [str(val).lower() if pd.notna(val) else "" for val in row]
            
            # Check if this is the data header row
            header_found_in_row = any('time' in s or 'voltage' in s or 'current' in s for s in row_strings)
            
            if header_found_in_row:
                data_start_row = idx
                found_header = True
                break
            
            # Parse metadata from any column pair
            for col_idx in range(len(row) - 1):
                label = str(row[col_idx]).lower() if pd.notna(row[col_idx]) else ""
                value = row[col_idx + 1]
                
                if not label or not pd.notna(value):
                    continue
                
                # Basic metadata
                if 'battery' in label and 'code' in label:
                    metadata['battery_code'] = str(value)
                elif 'temperature' in label:
                    metadata['temperature'] = str(value)
                elif 'build' in label and 'number' in label:
                    metadata['build_id'] = str(value)
                
                # Standard benchmark parameters
                elif 'min' in label and 'voltage' in label and 'activ' in label:
                    try:
                        standard_params['min_activation_voltage'] = float(value)
                    except:
                        pass
                elif 'std' in label and 'max' in label and ('open' in label or 'circuit' in label):
                    try:
                        standard_params['std_max_oc_voltage'] = float(value)
                    except:
                        pass
                elif 'std' in label and 'activ' in label and 'time' in label:
                    try:
                        standard_params['std_activation_time_ms'] = float(value)
                    except:
                        pass
                elif 'target' in label and 'duration' in label:
                    try:
                        standard_params['std_duration_sec'] = float(value)
                    except:
                        pass
                
                # Extended build metadata
                elif 'anode' in label and 'weight' in label:
                    try:
                        extended_metadata['anode_weight_per_cell'] = float(value)
                    except:
                        pass
                elif 'cathode' in label and 'weight' in label:
                    try:
                        extended_metadata['cathode_weight_per_cell'] = float(value)
                    except:
                        pass
                elif 'electrolyte' in label and 'weight' in label:
                    try:
                        extended_metadata['electrolyte_weight'] = float(value)
                    except:
                        pass
                elif 'heat' in label and 'pellet' in label and 'weight' in label:
                    try:
                        extended_metadata['heat_pellet_weight'] = float(value)
                    except:
                        pass
                elif 'calorific' in label and 'value' in label:
                    try:
                        extended_metadata['calorific_value_per_gram'] = float(value)
                    except:
                        pass
                elif 'cells' in label and 'series' in label:
                    try:
                        extended_metadata['cells_in_series'] = int(float(value))
                    except:
                        pass
                elif 'stacks' in label and 'parallel' in label:
                    try:
                        extended_metadata['stacks_in_parallel'] = int(float(value))
                    except:
                        pass
        
        if not found_header:
            data_start_row = 0
        
        return metadata, standard_params, extended_metadata, data_start_row
    except:
        return metadata, standard_params, extended_metadata, 0

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file with complete metadata extraction"""
    try:
        metadata, standard_params, extended_metadata, skip_rows = extract_metadata_from_file(uploaded_file)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skip_rows)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None, None, None, None
        
        # Drop columns that are completely empty (all NaN)
        df = df.dropna(axis=1, how='all')
        
        # Drop rows that are completely empty
        df = df.dropna(axis=0, how='all')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # CRITICAL FIX: Convert numeric columns to proper dtypes
        # This prevents "str vs int" comparison errors in calculations
        for col in df.columns:
            # Try to convert each column to numeric, keeping non-numeric as-is
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df, metadata, standard_params, extended_metadata
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None, None, None

def load_multi_build_file(uploaded_file):
    """
    Load a single Excel file containing multiple builds arranged horizontally.
    
    Expected format:
    - Builds are arranged side by side (horizontally) with empty columns as separators
    - Each build has metadata in rows 1-10 (Battery Code, Temperature, Build Number, weights, etc.)
    - Data columns (Time, Discharge Current, Voltage) start at row 11
    - Build Number row identifies each build section
    
    Returns:
    - List of tuples: [(df, metadata, standard_params, extended_metadata), ...]
    """
    try:
        if not uploaded_file.name.endswith(('.xlsx', '.xls')):
            st.error("Multi-build format requires Excel files (.xlsx or .xls)")
            return []
        
        # Read the entire sheet without headers
        raw_df = pd.read_excel(uploaded_file, header=None)
        
        # Find build sections by looking for "Build Number" labels
        build_sections = []
        
        # Scan the first 15 rows for "Build Number" labels to find column positions
        for row_idx in range(min(15, len(raw_df))):
            row = raw_df.iloc[row_idx]
            for col_idx, val in enumerate(row):
                if pd.notna(val) and 'build' in str(val).lower() and 'number' in str(val).lower():
                    # Found a Build Number label, the next column has the build ID
                    if col_idx + 1 < len(row):
                        build_id = row.iloc[col_idx + 1]
                        if pd.notna(build_id):
                            # Find the start column for this build (look for metadata in nearby columns)
                            start_col = col_idx
                            build_sections.append({
                                'start_col': start_col,
                                'build_id': str(build_id),
                                'row_idx': row_idx
                            })
        
        if not build_sections:
            st.warning("Could not detect build sections in file. Looking for 'Build Number' labels.")
            return []
        
        # Sort by column position
        build_sections.sort(key=lambda x: x['start_col'])
        
        # Now extract data for each build
        results = []
        
        for i, section in enumerate(build_sections):
            start_col = section['start_col']
            build_id = section['build_id']
            
            # Determine end column (either next build's start or end of data)
            if i + 1 < len(build_sections):
                end_col = build_sections[i + 1]['start_col']
            else:
                end_col = len(raw_df.columns)
            
            # Extract metadata for this build from the header rows
            metadata = {
                'battery_code': None,
                'temperature': None,
                'build_id': build_id
            }
            
            standard_params = {
                'min_activation_voltage': None,
                'std_max_oc_voltage': None,
                'std_activation_time_ms': None,
                'std_duration_sec': None
            }
            
            extended_metadata = {
                'anode_weight_per_cell': None,
                'cathode_weight_per_cell': None,
                'electrolyte_weight': None,
                'heat_pellet_weight': None,
                'calorific_value_per_gram': None,
                'cells_in_series': None,
                'stacks_in_parallel': None
            }
            
            # Find the data header row (Time, Voltage, Current)
            data_start_row = 0
            for row_idx in range(min(20, len(raw_df))):
                row = raw_df.iloc[row_idx, start_col:end_col]
                row_strings = [str(val).lower() if pd.notna(val) else "" for val in row]
                
                if any('time' in s for s in row_strings):
                    data_start_row = row_idx
                    break
            
            # Parse metadata from header rows (before data_start_row)
            for row_idx in range(data_start_row):
                row = raw_df.iloc[row_idx, start_col:end_col]
                
                for col_offset in range(len(row) - 1):
                    label = str(row.iloc[col_offset]).lower() if pd.notna(row.iloc[col_offset]) else ""
                    value = row.iloc[col_offset + 1] if col_offset + 1 < len(row) else None
                    
                    if not label or not pd.notna(value):
                        continue
                    
                    # Basic metadata
                    if 'battery' in label and 'code' in label:
                        metadata['battery_code'] = str(value)
                    elif 'temperature' in label:
                        metadata['temperature'] = str(value)
                    
                    # Extended build metadata
                    elif 'anode' in label and 'weight' in label:
                        try:
                            extended_metadata['anode_weight_per_cell'] = float(value)
                        except:
                            pass
                    elif 'cathode' in label and 'weight' in label:
                        try:
                            extended_metadata['cathode_weight_per_cell'] = float(value)
                        except:
                            pass
                    elif 'electrolyte' in label and 'weight' in label:
                        try:
                            extended_metadata['electrolyte_weight'] = float(value)
                        except:
                            pass
                    elif 'heat' in label and 'pellet' in label and 'weight' in label:
                        try:
                            extended_metadata['heat_pellet_weight'] = float(value)
                        except:
                            pass
                    elif 'calorific' in label and 'value' in label:
                        try:
                            extended_metadata['calorific_value_per_gram'] = float(value)
                        except:
                            pass
                    elif 'cells' in label and 'series' in label:
                        try:
                            extended_metadata['cells_in_series'] = int(float(value))
                        except:
                            pass
                    elif 'stacks' in label and 'parallel' in label:
                        try:
                            extended_metadata['stacks_in_parallel'] = int(float(value))
                        except:
                            pass
            
            # Extract data columns for this build
            # Find the actual data columns (Time, Current, Voltage)
            header_row = raw_df.iloc[data_start_row, start_col:end_col]
            data_cols = []
            col_names = []
            
            for col_offset, val in enumerate(header_row):
                if pd.notna(val) and str(val).strip():
                    actual_col = start_col + col_offset
                    data_cols.append(actual_col)
                    col_names.append(str(val).strip())
            
            if len(data_cols) >= 2:  # Need at least Time and Voltage
                # Extract the data rows
                build_data = raw_df.iloc[data_start_row + 1:, data_cols].copy()
                build_data.columns = col_names
                
                # Drop empty rows
                build_data = build_data.dropna(how='all')
                build_data = build_data.reset_index(drop=True)
                
                # Convert numeric columns
                for col in build_data.columns:
                    build_data[col] = pd.to_numeric(build_data[col], errors='ignore')
                
                # Only add if we have actual data
                if len(build_data) > 0:
                    results.append((build_data, metadata, standard_params, extended_metadata))
        
        return results
    
    except Exception as e:
        st.error(f"Error loading multi-build file: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

def detect_columns(df):
    """Auto-detect relevant columns in the dataset"""
    time_col = None
    voltage_col = None
    current_col = None
    capacity_col = None
    
    for col in df.columns:
        col_lower = str(col).lower()
        
        # Time column
        if 'time' in col_lower and time_col is None:
            time_col = col
        
        # Voltage column
        elif ('voltage' in col_lower or 'volt' in col_lower) and voltage_col is None:
            voltage_col = col
        
        # Current column (handle both "discharge" and "dicharge" typo)
        elif ('current' in col_lower or 'amp' in col_lower or 'dicharge' in col_lower or 'discharge' in col_lower) and current_col is None:
            current_col = col
        
        # Capacity column
        elif ('capacity' in col_lower or 'cap' in col_lower) and capacity_col is None:
            capacity_col = col
    
    return time_col, voltage_col, current_col, capacity_col

def detect_time_unit_and_convert(time_series):
    """
    Detect time unit (milliseconds or seconds) and convert to minutes.
    Defaults to seconds for safety.
    
    Detection Logic:
    - max > 10000: milliseconds (e.g., 3000s = 3,000,000ms)
    - avg_interval > 1 AND max > 100: milliseconds (e.g., 10ms intervals in ms data)
    - otherwise: seconds (safe default)
    
    Note: If your data is already in minutes, please convert to seconds before uploading.
    The application is optimized for high-frequency test data in seconds or milliseconds.
    """
    try:
        numeric_series = pd.to_numeric(time_series, errors='coerce')
        if numeric_series.isna().all():
            return time_series
        
        max_time = numeric_series.max()
        min_time = numeric_series.min()
        time_range = max_time - min_time
        num_points = len(numeric_series)
        
        if time_range == 0:
            return numeric_series
        
        avg_interval = time_range / max(num_points - 1, 1)
        
        if max_time > 10000:
            conversion_factor = 1.0 / 60000.0
            unit = "milliseconds"
        elif avg_interval > 1.0 and max_time > 100:
            conversion_factor = 1.0 / 60000.0
            unit = "milliseconds"
        else:
            conversion_factor = 1.0 / 60.0
            unit = "seconds"
        
        converted = numeric_series * conversion_factor
        
        if unit == "milliseconds":
            st.info(f"⏱️ Time column auto-detected as **{unit}** and converted to minutes for calculations.")
        
        return converted
    except:
        return time_series

def extract_temperature_from_name(name):
    """
    Extract temperature from build name (e.g., '25C', '-20C', '0C')
    Returns temperature as a float or None if not found
    """
    import re
    match = re.search(r'(-?\d+)\s*°?C', name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def downsample_for_plotting(df, max_points=10000):
    """
    Downsample large datasets for efficient plotting while preserving data characteristics.
    Uses intelligent sampling to keep start, end, and evenly distributed points.
    Guarantees output has at most max_points rows.
    """
    if len(df) <= max_points:
        return df
    
    step = max(1, len(df) // max_points)
    if step * max_points < len(df):
        step += 1
    
    indices = list(range(0, len(df), step))
    
    if len(indices) > max_points:
        indices = indices[:max_points-1]
    
    if indices[-1] != len(df) - 1:
        indices.append(len(df) - 1)
    
    return df.iloc[indices].copy()

def calculate_metrics(df, time_col, voltage_col, current_col=None, min_activation_voltage=1.0, extended_metadata=None, target_duration_sec=None):
    """Calculate key battery metrics including new performance metrics
    
    Args:
        df: DataFrame with battery discharge data
        time_col: Name of time column
        voltage_col: Name of voltage column
        current_col: Name of current column
        min_activation_voltage: Minimum voltage threshold for activation
        extended_metadata: Dict with extended build metadata (weights, calorific value, etc.)
        target_duration_sec: Optional target duration in seconds for voltage lookup
    """
    metrics = {}
    
    if voltage_col and voltage_col in df.columns:
        metrics['Max Voltage (V)'] = float(df[voltage_col].max())
        # Min Voltage and Voltage Range removed per user request
        
        # Calculate max open circuit voltage and max on-load voltage WITH metadata
        if current_col and current_col in df.columns:
            # Open circuit: when current is 0 or very close to 0 (< 0.01A)
            open_circuit_mask = df[current_col].abs() < 0.01
            on_load_mask = df[current_col].abs() >= 0.01
            
            if open_circuit_mask.any():
                oc_idx = df.loc[open_circuit_mask, voltage_col].idxmax()
                metrics['Max Open Circuit Voltage (V)'] = float(df.loc[oc_idx, voltage_col])
                # Add timestamp when max OC voltage occurred
                if time_col and time_col in df.columns:
                    metrics['Max OC Voltage Time (s)'] = None  # Will be filled later when time is converted
            else:
                metrics['Max Open Circuit Voltage (V)'] = None
                metrics['Max OC Voltage Time (s)'] = None
            
            if on_load_mask.any():
                onload_idx = df.loc[on_load_mask, voltage_col].idxmax()
                metrics['Max On-Load Voltage (V)'] = float(df.loc[onload_idx, voltage_col])
                # Add timestamp when max on-load voltage occurred
                if time_col and time_col in df.columns:
                    metrics['Max On-Load Time (s)'] = None  # Will be filled later when time is converted
            else:
                metrics['Max On-Load Voltage (V)'] = None
                metrics['Max On-Load Time (s)'] = None
        else:
            # If no current column, assume all measurements are on-load
            onload_idx = df[voltage_col].idxmax()
            metrics['Max On-Load Voltage (V)'] = float(df.loc[onload_idx, voltage_col])
            metrics['Max On-Load Time (s)'] = None
            metrics['Max Open Circuit Voltage (V)'] = None
            metrics['Max OC Voltage Time (s)'] = None
    
    if time_col and time_col in df.columns:
        time_series = df[time_col]
        
        if not pd.api.types.is_numeric_dtype(time_series) and not pd.api.types.is_datetime64_any_dtype(time_series) and not pd.api.types.is_timedelta64_dtype(time_series):
            try:
                time_series = pd.to_numeric(time_series, errors='raise')
                time_series = detect_time_unit_and_convert(time_series)
            except:
                try:
                    time_series = pd.to_timedelta(time_series)
                except:
                    try:
                        time_series = pd.to_datetime(time_series)
                    except:
                        return metrics
        elif pd.api.types.is_numeric_dtype(time_series):
            time_series = detect_time_unit_and_convert(time_series)
        
        time_max = time_series.max()
        time_min = time_series.min()
        time_range_raw = time_max - time_min
        
        if pd.api.types.is_timedelta64_dtype(time_range_raw):
            time_range_minutes = time_range_raw.total_seconds() / 60
        elif isinstance(time_range_raw, pd.Timedelta):
            time_range_minutes = time_range_raw.total_seconds() / 60
        else:
            try:
                time_range_minutes = float(time_range_raw)
            except:
                time_range_minutes = 0
        
        # Calculate activation time and duration
        # Activation Time (Sec): The time when battery FIRST reaches >= min_activation_voltage
        # Duration (Sec): Time from activation to LAST occurrence of voltage >= min_activation_voltage
        if voltage_col and voltage_col in df.columns:
            # Create mask for when voltage is >= minimum activation voltage
            above_threshold_mask = df[voltage_col] >= min_activation_voltage
            
            if above_threshold_mask.any():
                # Find the FIRST time when voltage >= min_activation_voltage
                first_activation_idx = above_threshold_mask.idxmax()
                
                # Find the LAST time when voltage >= min_activation_voltage
                # Get indices where voltage is above threshold
                above_threshold_indices = df[above_threshold_mask].index
                last_occurrence_idx = above_threshold_indices[-1]
                
                # Get time series as seconds (not minutes)
                if pd.api.types.is_timedelta64_dtype(time_series):
                    time_in_seconds = time_series.dt.total_seconds()
                elif isinstance(time_series.iloc[0], pd.Timedelta):
                    time_in_seconds = time_series.apply(lambda x: x.total_seconds())
                else:
                    # Already in minutes from detect_time_unit_and_convert, convert to seconds
                    time_in_seconds = time_series * 60
                
                # Activation Time: time at first occurrence
                activation_time_sec = time_in_seconds.iloc[first_activation_idx] - time_in_seconds.iloc[0]
                
                # Duration: Time from first activation to LAST occurrence of cutoff voltage
                duration_sec = time_in_seconds.iloc[last_occurrence_idx] - time_in_seconds.iloc[first_activation_idx]
                
                metrics['Activation Time (Sec)'] = float(activation_time_sec)
                metrics['Duration (Sec)'] = float(duration_sec)
                
                # Fill in timestamps for Max OC and On-Load voltages
                if current_col and current_col in df.columns:
                    open_circuit_mask = df[current_col].abs() < 0.01
                    on_load_mask = df[current_col].abs() >= 0.01
                    
                    if open_circuit_mask.any():
                        oc_idx = df.loc[open_circuit_mask, voltage_col].idxmax()
                        metrics['Max OC Voltage Time (s)'] = float(time_in_seconds.loc[oc_idx] - time_in_seconds.iloc[0])
                    
                    if on_load_mask.any():
                        onload_idx = df.loc[on_load_mask, voltage_col].idxmax()
                        metrics['Max On-Load Time (s)'] = float(time_in_seconds.loc[onload_idx] - time_in_seconds.iloc[0])
                
                # Calculate time-weighted average voltage during discharge period
                # Per user requirement: weighted average from activation till reaching min voltage on discharge
                # taking into consideration the duration at which a certain voltage is continued
                discharge_mask = (df.index >= first_activation_idx) & (df.index <= last_occurrence_idx)
                discharge_voltages = df.loc[discharge_mask, voltage_col]
                discharge_times = time_in_seconds.loc[discharge_mask]
                
                if len(discharge_voltages) > 1:
                    # Calculate time differences
                    time_diffs = discharge_times.diff().fillna(0)
                    # Time-weighted average: Σ(V[i] × Δt[i]) / Σ(Δt[i])
                    total_time = time_diffs.sum()
                    if total_time > 0:
                        weighted_avg_voltage = (discharge_voltages * time_diffs).sum() / total_time
                        metrics['Weighted Average Voltage (V)'] = float(weighted_avg_voltage)
                    else:
                        metrics['Weighted Average Voltage (V)'] = float(discharge_voltages.mean())
                else:
                    metrics['Weighted Average Voltage (V)'] = float(discharge_voltages.mean()) if len(discharge_voltages) > 0 else None
                
                # Calculate voltage at targeted duration (if specified)
                if target_duration_sec is not None and target_duration_sec > 0:
                    # Find voltage at target_duration_sec from activation start
                    target_time_absolute = time_in_seconds.iloc[first_activation_idx] + target_duration_sec
                    
                    if target_time_absolute <= time_in_seconds.iloc[last_occurrence_idx]:
                        # Linear interpolation to find voltage at target duration
                        from scipy import interpolate
                        f = interpolate.interp1d(discharge_times.values, discharge_voltages.values, 
                                                kind='linear', fill_value='extrapolate')
                        voltage_at_target = float(f(target_time_absolute))
                        metrics['Voltage at Targeted Duration (V)'] = voltage_at_target
                    else:
                        # Target duration exceeds actual duration
                        metrics['Voltage at Targeted Duration (V)'] = None
                else:
                    metrics['Voltage at Targeted Duration (V)'] = None
            else:
                # Voltage never reaches the threshold
                metrics['Activation Time (Sec)'] = None
                metrics['Duration (Sec)'] = None
                metrics['Weighted Average Voltage (V)'] = None
                metrics['Voltage at Targeted Duration (V)'] = None
        
        if voltage_col and voltage_col in df.columns and time_range_minutes > 0:
            voltage_drop = df[voltage_col].iloc[0] - df[voltage_col].iloc[-1]
            metrics['Discharge Rate (V/sec)'] = float(voltage_drop / (time_range_minutes * 60))
    
    if voltage_col and current_col and voltage_col in df.columns and current_col in df.columns:
        power = df[voltage_col] * df[current_col].abs()
        metrics['Average Power (W)'] = float(power.mean())
        metrics['Max Power (W)'] = float(power.max())
        
        # Total Energy (Wh) removed per user request (calculation was incorrect)
    
    # Calculate extended performance metrics if metadata is provided
    if extended_metadata and current_col and current_col in df.columns and time_col and time_col in df.columns and voltage_col and voltage_col in df.columns:
        # Check if time_series was successfully created and converted earlier
        if 'time_series' in locals() and time_series is not None:
            # Reuse the already-converted time_series from above
            # Handle both Timedelta and numeric (minutes) cases
            if pd.api.types.is_timedelta64_dtype(time_series):
                # time_series is Timedelta, convert directly to seconds
                time_in_seconds = time_series.dt.total_seconds()
            elif isinstance(time_series.iloc[0] if len(time_series) > 0 else None, pd.Timedelta):
                # Individual Timedelta objects
                time_in_seconds = time_series.apply(lambda x: x.total_seconds())
            else:
                # time_series is numeric in minutes, convert to seconds
                time_in_seconds = time_series * 60
            
            if len(df) > 1:
                # Create discharge mask to only include data from activation to last cutoff
                above_threshold_mask = df[voltage_col] >= min_activation_voltage
                
                if above_threshold_mask.any():
                    # Use the same indices calculated for activation/duration
                    first_activation_idx = above_threshold_mask.idxmax()
                    above_threshold_indices = df[above_threshold_mask].index
                    last_occurrence_idx = above_threshold_indices[-1]
                    
                    # Filter to discharge period only
                    discharge_mask = (df.index >= first_activation_idx) & (df.index <= last_occurrence_idx)
                    
                    # Calculate ampere-seconds only during discharge period
                    time_diff_seconds = pd.Series(time_in_seconds).diff().fillna(0)
                    current_abs = df[current_col].abs()
                    
                    # Create a mask that excludes intervals starting before activation
                    # (both current and previous sample must be in discharge period)
                    # Shift discharge_mask by 1 position (prepend False, drop last)
                    discharge_mask_prev = np.concatenate([[False], discharge_mask[:-1]])
                    valid_discharge_intervals = discharge_mask & discharge_mask_prev
                    
                    # Total ampere-seconds = sum of (current * time_diff) during discharge only
                    total_ampere_seconds = (current_abs[valid_discharge_intervals] * time_diff_seconds[valid_discharge_intervals]).sum()
                    metrics['Total Ampere-Seconds (A·s)'] = float(total_ampere_seconds)
                    
                    # Get per-cell weights and stacks in parallel
                    # Correct formula: Total A·s / No. of Stacks in parallel / Weight of single pellet
                    anode_per_cell = extended_metadata.get('anode_weight_per_cell', 0)
                    cathode_per_cell = extended_metadata.get('cathode_weight_per_cell', 0)
                    stacks_in_parallel = extended_metadata.get('stacks_in_parallel', 1)
                    
                    # Ampere-seconds per gram of anode pellet
                    if anode_per_cell > 0 and stacks_in_parallel > 0:
                        metrics['A·s per gram Anode'] = float(total_ampere_seconds / stacks_in_parallel / anode_per_cell)
                    else:
                        metrics['A·s per gram Anode'] = None
                    
                    # Ampere-seconds per gram of cathode pellet
                    if cathode_per_cell > 0 and stacks_in_parallel > 0:
                        metrics['A·s per gram Cathode'] = float(total_ampere_seconds / stacks_in_parallel / cathode_per_cell)
                    else:
                        metrics['A·s per gram Cathode'] = None
                else:
                    # No discharge period found
                    metrics['Total Ampere-Seconds (A·s)'] = None
                    metrics['A·s per gram Anode'] = None
                    metrics['A·s per gram Cathode'] = None
    
    return metrics

def export_to_excel(dataframes, build_names, metrics_df):
    """Export all data and metrics to Excel file"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, name in zip(dataframes, build_names):
            sheet_name = name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name='Metrics_Comparison')
    
    return output.getvalue()

def export_to_csv(dataframes, build_names, metrics_df):
    """Export metrics comparison to CSV"""
    if metrics_df is not None:
        return metrics_df.to_csv(index=True).encode('utf-8')
    return None

def generate_detailed_report(metrics_df, build_names, metadata_list, 
                             min_activation_voltage,
                             use_standards=False, std_max_onload_voltage=None, 
                             std_max_oc_voltage=None, std_activation_time=None, 
                             std_duration=None, std_activation_time_ms=None,
                             std_duration_sec=None):
    """Generate a comprehensive text report with all metrics and comparisons"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 80)
    report.append("BATTERY DISCHARGE ANALYSIS - DETAILED REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {timestamp}")
    report.append(f"Number of Builds Analyzed: {len(build_names)}")
    report.append("")
    
    # Test Configuration
    report.append("-" * 80)
    report.append("TEST CONFIGURATION")
    report.append("-" * 80)
    report.append(f"Min. Voltage for Activation: {min_activation_voltage} V")
    report.append("Note: Activation Time (Sec) = Time when battery FIRST reaches >= min voltage")
    report.append("      Duration (Sec) = Time from activation to LAST occurrence of cutoff voltage")
    report.append("")
    
    # Build Information
    report.append("-" * 80)
    report.append("BUILD INFORMATION")
    report.append("-" * 80)
    for i, (name, metadata) in enumerate(zip(build_names, metadata_list if metadata_list else [{}]*len(build_names))):
        report.append(f"\nBuild {i+1}: {name}")
        if metadata and any(metadata.values()):
            if metadata.get('battery_code'):
                report.append(f"  Battery Code: {metadata['battery_code']}")
            if metadata.get('temperature'):
                report.append(f"  Temperature: {metadata['temperature']}")
            if metadata.get('build_id'):
                report.append(f"  Build ID: {metadata['build_id']}")
    report.append("")
    
    # Performance Metrics
    report.append("-" * 80)
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    
    if metrics_df is not None and not metrics_df.empty:
        for build_name in metrics_df.index:
            report.append(f"\n{build_name}:")
            report.append("-" * 40)
            for col in metrics_df.columns:
                value = metrics_df.loc[build_name, col]
                if pd.notna(value):
                    report.append(f"  {col}: {value:.4f}")
                else:
                    report.append(f"  {col}: N/A")
    report.append("")
    
    # Statistical Summary
    report.append("-" * 80)
    report.append("STATISTICAL SUMMARY")
    report.append("-" * 80)
    
    if metrics_df is not None:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                report.append(f"\n{col}:")
                report.append(f"  Mean: {metrics_df[col].mean():.4f}")
                report.append(f"  Std Dev: {metrics_df[col].std():.4f}")
                report.append(f"  Min: {metrics_df[col].min():.4f}")
                report.append(f"  Max: {metrics_df[col].max():.4f}")
                report.append(f"  Range: {metrics_df[col].max() - metrics_df[col].min():.4f}")
    report.append("")
    
    # Standard Performance Comparison
    if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time_ms, std_duration_sec]):
        report.append("-" * 80)
        report.append("STANDARD PERFORMANCE BENCHMARKS COMPARISON")
        report.append("-" * 80)
        report.append("\nStandard Values:")
        if std_max_onload_voltage:
            report.append(f"  Max On-Load Voltage: {std_max_onload_voltage} V")
        if std_max_oc_voltage:
            report.append(f"  Max Open Circuit Voltage: {std_max_oc_voltage} V")
        if std_activation_time_ms:
            std_activation_sec = std_activation_time_ms / 1000.0
            report.append(f"  Max Activation Time: {std_activation_time_ms} ms ({std_activation_sec:.2f} s)")
        if std_duration_sec:
            report.append(f"  Min Duration: {std_duration_sec} s")
        report.append("")
        
        for build_name in metrics_df.index:
            report.append(f"\n{build_name} Performance:")
            report.append("-" * 40)
            
            # Check each metric
            if std_max_onload_voltage and 'Max On-Load Voltage (V)' in metrics_df.columns:
                actual = safe_scalar(metrics_df.loc[build_name, 'Max On-Load Voltage (V)'])
                if pd.notna(actual):
                    diff = actual - std_max_onload_voltage
                    report.append(f"  Max On-Load Voltage: {actual:.4f} V (Std: {std_max_onload_voltage} V, Diff: {diff:+.4f} V)")
            
            if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
                actual = safe_scalar(metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)'])
                if pd.notna(actual):
                    diff = actual - std_max_oc_voltage
                    report.append(f"  Max Open Circuit Voltage: {actual:.4f} V (Std: {std_max_oc_voltage} V, Diff: {diff:+.4f} V)")
            
            if std_activation_time_ms and 'Activation Time (Sec)' in metrics_df.columns:
                actual_sec = safe_scalar(metrics_df.loc[build_name, 'Activation Time (Sec)'])
                if pd.notna(actual_sec):
                    std_sec = std_activation_time_ms / 1000.0  # Convert ms to seconds
                    diff = actual_sec - std_sec
                    report.append(f"  Activation Time: {actual_sec:.2f} s (Std: {std_activation_time_ms} ms = {std_sec:.2f} s, Diff: {diff:+.2f} s)")
            
            if std_duration_sec and 'Duration (Sec)' in metrics_df.columns:
                actual_sec = safe_scalar(metrics_df.loc[build_name, 'Duration (Sec)'])
                if pd.notna(actual_sec):
                    diff = actual_sec - std_duration_sec
                    report.append(f"  Duration: {actual_sec:.2f} s (Target: {std_duration_sec} s, Diff: {diff:+.2f} s)")
        
        report.append("")
    
    # Build-to-Build Comparison
    if len(metrics_df) >= 2:
        report.append("-" * 80)
        report.append("BUILD-TO-BUILD COMPARISON")
        report.append("-" * 80)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        
        for i in range(len(metrics_df) - 1):
            build1 = metrics_df.index[i]
            build2 = metrics_df.index[i + 1]
            
            report.append(f"\n{build1} vs {build2}:")
            report.append("-" * 40)
            
            for col in numeric_cols:
                val1 = safe_scalar(metrics_df.loc[build1, col])
                val2 = safe_scalar(metrics_df.loc[build2, col])
                if pd.notna(val1) and pd.notna(val2):
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else 0
                    report.append(f"  {col}:")
                    report.append(f"    {build1}: {val1:.4f}")
                    report.append(f"    {build2}: {val2:.4f}")
                    report.append(f"    Difference: {diff:+.4f} ({pct_change:+.2f}%)")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def generate_pdf_report(metrics_df, build_names, metadata_list, 
                        min_activation_voltage,
                        use_standards=False, std_max_onload_voltage=None, 
                        std_max_oc_voltage=None, std_activation_time=None, 
                        std_duration=None, std_activation_time_ms=None,
                        std_duration_sec=None,
                        extended_metadata_list=None,
                        analytics_list=None,
                        correlations=None,
                        duration_correlations=None):
    """Generate a comprehensive PDF report with all metrics, analytics, and correlations"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=rl_colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=rl_colors.HexColor('#2ca02c'),
        spaceAfter=6,
        spaceBefore=12
    )
    # Header style for table headers with word wrapping
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica-Bold',
        textColor=rl_colors.whitesmoke,
        alignment=TA_CENTER,
        wordWrap='CJK'
    )
    # Cell style for table content with word wrapping
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica',
        alignment=TA_CENTER,
        wordWrap='CJK',
        leading=10
    )
    table_cell_left_style = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica',
        alignment=TA_LEFT,
        wordWrap='CJK',
        leading=10
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    story.append(Paragraph("BATTERY DISCHARGE ANALYSIS", title_style))
    story.append(Paragraph("Detailed Performance Report", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    info_data = [
        [Paragraph('Generated:', table_cell_left_style), Paragraph(timestamp, table_cell_style)],
        [Paragraph('Number of Builds:', table_cell_left_style), Paragraph(str(len(build_names)), table_cell_style)],
        [Paragraph('Min. Voltage for Activation:', table_cell_left_style), Paragraph(f"{min_activation_voltage} V", table_cell_style)]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Build Information", heading_style))
    build_data = [[Paragraph('Build', table_header_style), Paragraph('Battery Code', table_header_style), 
                   Paragraph('Temperature', table_header_style), Paragraph('Build ID', table_header_style)]]
    for i, (name, metadata) in enumerate(zip(build_names, metadata_list if metadata_list else [{}]*len(build_names))):
        build_data.append([
            Paragraph(str(name), table_cell_left_style),
            Paragraph(str(metadata.get('battery_code', 'N/A') if metadata else 'N/A'), table_cell_style),
            Paragraph(str(metadata.get('temperature', 'N/A') if metadata else 'N/A'), table_cell_style),
            Paragraph(str(metadata.get('build_id', 'N/A') if metadata else 'N/A'), table_cell_style)
        ])
    
    build_table = Table(build_data, colWidths=[2*inch, 1.5*inch, 1.2*inch, 1.3*inch])
    build_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(build_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Performance Metrics", heading_style))
    
    if metrics_df is not None and not metrics_df.empty:
        key_metrics = ['Max On-Load Voltage (V)', 'Max Open Circuit Voltage (V)', 
                      'Activation Time (Sec)', 'Duration (Sec)']
        
        # Wrap headers in Paragraph objects for word wrapping
        header_row = [Paragraph('Build', table_header_style)]
        for m in key_metrics:
            if m in metrics_df.columns:
                header_row.append(Paragraph(m, table_header_style))
        metrics_data = [header_row]
        
        for build_name in metrics_df.index:
            row = [Paragraph(str(build_name), table_cell_left_style)]
            for metric in key_metrics:
                if metric in metrics_df.columns:
                    value = metrics_df.loc[build_name, metric]
                    if pd.notna(value):
                        row.append(Paragraph(f"{value:.3f}", table_cell_style))
                    else:
                        row.append(Paragraph("N/A", table_cell_style))
            metrics_data.append(row)
        
        col_widths = [2*inch] + [1.5*inch] * (len(metrics_data[0]) - 1)
        metrics_table = Table(metrics_data, colWidths=col_widths)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
    
    if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time_ms, std_duration_sec]):
        story.append(PageBreak())
        story.append(Paragraph("Standard Performance Benchmarks", heading_style))
        
        std_data = [[Paragraph('Metric', table_header_style), Paragraph('Standard Value', table_header_style)]]
        if std_max_onload_voltage:
            std_data.append([Paragraph('Max On-Load Voltage', table_cell_left_style), Paragraph(f"{std_max_onload_voltage} V", table_cell_style)])
        if std_max_oc_voltage:
            std_data.append([Paragraph('Max Open Circuit Voltage', table_cell_left_style), Paragraph(f"{std_max_oc_voltage} V", table_cell_style)])
        if std_activation_time_ms:
            std_activation_sec = std_activation_time_ms / 1000.0
            std_data.append([Paragraph('Max Activation Time', table_cell_left_style), Paragraph(f"{std_activation_time_ms} ms ({std_activation_sec:.2f} s)", table_cell_style)])
        if std_duration_sec:
            std_data.append([Paragraph('Target Min Duration', table_cell_left_style), Paragraph(f"{std_duration_sec} s", table_cell_style)])
        
        std_table = Table(std_data, colWidths=[3*inch, 3*inch])
        std_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightyellow),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(std_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Performance Comparison vs Standards", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        build_list = list(metrics_df.index)
        header_row = [Paragraph('Metric', table_header_style), Paragraph('Standard', table_header_style)]
        for build_name in build_list:
            header_row.append(Paragraph(str(build_name), table_header_style))
        
        comp_data = [header_row]
        
        if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
            row = [Paragraph('Max OC V', table_cell_left_style), Paragraph(f"{std_max_oc_voltage} V", table_cell_style)]
            for build_name in build_list:
                actual = safe_scalar(metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)'])
                row.append(Paragraph(f"{actual:.3f} V" if pd.notna(actual) else "N/A", table_cell_style))
            comp_data.append(row)
        
        if std_activation_time and 'Activation Time (Sec)' in metrics_df.columns:
            std_val = f"{std_activation_time_ms} ms" if std_activation_time_ms else f"{std_activation_time} min"
            row = [Paragraph('Activation Time', table_cell_left_style), Paragraph(std_val, table_cell_style)]
            for build_name in build_list:
                actual = safe_scalar(metrics_df.loc[build_name, 'Activation Time (Sec)'])
                row.append(Paragraph(f"{actual:.3f} sec" if pd.notna(actual) else "N/A", table_cell_style))
            comp_data.append(row)
        
        if std_duration and 'Duration (Sec)' in metrics_df.columns:
            std_val = f"{std_duration_sec} sec" if std_duration_sec else f"{std_duration} min"
            row = [Paragraph('Duration', table_cell_left_style), Paragraph(std_val, table_cell_style)]
            for build_name in build_list:
                actual = safe_scalar(metrics_df.loc[build_name, 'Duration (Sec)'])
                row.append(Paragraph(f"{actual:.3f} sec" if pd.notna(actual) else "N/A", table_cell_style))
            comp_data.append(row)
        
        num_builds = len(build_list)
        col_widths = [1.5*inch, 1.2*inch] + [1.2*inch] * num_builds
        if sum(col_widths) > 7*inch:
            col_widths = [1.2*inch, 1*inch] + [0.9*inch] * num_builds
        
        comp_table = Table(comp_data, colWidths=col_widths)
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(comp_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Extended Build Metadata Section - Combined table for all builds
    if extended_metadata_list and any(meta for meta in extended_metadata_list):
        story.append(PageBreak())
        story.append(Paragraph("Extended Build Metadata", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        header_row = [Paragraph('Metric', table_header_style)]
        for build_name in build_names:
            header_row.append(Paragraph(str(build_name), table_header_style))
        
        ext_data = [header_row]
        
        metric_keys = [
            ('anode_weight_per_cell', 'Anode Weight (g)', lambda v: f"{v:.2f}"),
            ('cathode_weight_per_cell', 'Cathode Weight (g)', lambda v: f"{v:.2f}"),
            ('heat_pellet_weight', 'Heat Pellet (g)', lambda v: f"{v:.2f}"),
            ('electrolyte_weight', 'Electrolyte (g)', lambda v: f"{v:.2f}"),
            ('cells_in_series', 'Cells in Series', lambda v: f"{int(v)}"),
            ('stacks_in_parallel', 'Stacks in Parallel', lambda v: f"{int(v)}"),
            ('calorific_value_per_gram', 'Calorific Value (cal/g)', lambda v: f"{v:.0f}"),
            ('total_anode_weight', 'Total Anode (g)', lambda v: f"{v:.2f}"),
            ('total_cathode_weight', 'Total Cathode (g)', lambda v: f"{v:.2f}"),
            ('total_stack_weight', 'Total Stack (g)', lambda v: f"{v:.2f}"),
            ('total_calorific_value', 'Total Calorific (kJ)', lambda v: f"{v:.2f}"),
        ]
        
        for key, label, formatter in metric_keys:
            has_any_value = any(
                ext_meta and ext_meta.get(key) 
                for ext_meta in extended_metadata_list
            )
            if has_any_value:
                row = [Paragraph(label, table_cell_left_style)]
                for ext_meta in extended_metadata_list:
                    val = ext_meta.get(key) if ext_meta else None
                    row.append(Paragraph(formatter(val) if val else "-", table_cell_style))
                ext_data.append(row)
        
        if len(ext_data) > 1:
            num_builds = len(build_names)
            col_widths = [2*inch] + [1.2*inch] * num_builds
            if sum(col_widths) > 7*inch:
                col_widths = [1.5*inch] + [0.9*inch] * num_builds
            
            ext_table = Table(ext_data, colWidths=col_widths)
            ext_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2ca02c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.HexColor('#e6ffe6')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(ext_table)
            story.append(Spacer(1, 0.2*inch))
    
    # Advanced Performance Metrics Section
    if metrics_df is not None and not metrics_df.empty:
        advanced_metrics = ['Total Ampere-Seconds (A·s)', 'A·s per gram Anode', 
                          'A·s per gram Cathode']
        
        if any(m in metrics_df.columns for m in advanced_metrics):
            story.append(PageBreak())
            story.append(Paragraph("Advanced Performance Metrics", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Wrap headers in Paragraph objects for word wrapping
            header_row = [Paragraph('Build', table_header_style)]
            for m in advanced_metrics:
                if m in metrics_df.columns:
                    header_row.append(Paragraph(m, table_header_style))
            adv_data = [header_row]
            
            for build_name in metrics_df.index:
                row = [Paragraph(str(build_name), table_cell_left_style)]
                for metric in advanced_metrics:
                    if metric in metrics_df.columns:
                        value = metrics_df.loc[build_name, metric]
                        if pd.notna(value):
                            row.append(Paragraph(f"{value:.4f}", table_cell_style))
                        else:
                            row.append(Paragraph("N/A", table_cell_style))
                adv_data.append(row)
            
            col_widths = [1.5*inch] + [1.3*inch] * (len(adv_data[0]) - 1)
            adv_table = Table(adv_data, colWidths=col_widths)
            adv_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#9467bd')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lavender),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(adv_table)
            story.append(Spacer(1, 0.2*inch))
    
    # Discharge Curve Analysis Section - Combined table for all builds
    if analytics_list and any(analytics for analytics in analytics_list):
        story.append(PageBreak())
        story.append(Paragraph("Discharge Curve Analysis", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        header_row = [Paragraph('Metric', table_header_style)]
        for build_name in build_names:
            header_row.append(Paragraph(str(build_name), table_header_style))
        
        curve_data = [header_row]
        
        curve_metrics = [
            ('ΔV/ΔT Mean (V/s)', 'Mean ΔV/ΔT (V/s)', lambda v: f"{v:.6f}"),
            ('ΔV/ΔT Std Dev (V/s)', 'Std Dev ΔV/ΔT (V/s)', lambda v: f"{v:.6f}"),
            ('ΔV/ΔT Max (V/s)', 'Max |ΔV/ΔT| (V/s)', lambda v: f"{v:.6f}"),
            ('Discharge Slope Variability (%)', 'Slope Variability (%)', lambda v: f"{v:.2f}"),
            ('Curve Stability Assessment', 'Stability', lambda v: str(v)),
        ]
        
        for key, label, formatter in curve_metrics:
            has_any_value = any(
                analytics and analytics.get(key) is not None 
                for analytics in analytics_list
            )
            if has_any_value:
                row = [Paragraph(label, table_cell_left_style)]
                for analytics in analytics_list:
                    val = analytics.get(key) if analytics else None
                    row.append(Paragraph(formatter(val) if val is not None else "-", table_cell_style))
                curve_data.append(row)
        
        if len(curve_data) > 1:
            num_builds = len(build_names)
            col_widths = [2*inch] + [1.2*inch] * num_builds
            if sum(col_widths) > 7*inch:
                col_widths = [1.5*inch] + [0.9*inch] * num_builds
            
            curve_table = Table(curve_data, colWidths=col_widths)
            curve_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#17becf')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.HexColor('#e6f7ff')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(curve_table)
            story.append(Spacer(1, 0.2*inch))
    
    # Correlation Analysis Section
    if correlations and len(correlations) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Correlation Analysis", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        corr_data = [[Paragraph('Relationship', table_header_style), Paragraph('Correlation', table_header_style), 
                      Paragraph('P-Value', table_header_style), Paragraph('Strength', table_header_style), 
                      Paragraph('Significance', table_header_style)]]
        
        for corr_name, corr_info in correlations.items():
            corr_data.append([
                Paragraph(str(corr_name), table_cell_left_style),
                Paragraph(f"{corr_info['correlation']:.3f}", table_cell_style),
                Paragraph(f"{corr_info['p_value']:.4f}", table_cell_style),
                Paragraph(f"{corr_info['strength']} ({corr_info['direction']})", table_cell_style),
                Paragraph(corr_info['significance'], table_cell_style)
            ])
        
        corr_table = Table(corr_data, colWidths=[2*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1.2*inch])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#d62728')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightpink),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(corr_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("<b>Interpretation Guide:</b>", styles['Normal']))
        story.append(Spacer(1, 0.05*inch))
        interp_text = """
        • Correlation coefficient ranges from -1 to +1<br/>
        • Strong correlation: |r| > 0.7, Moderate: 0.4 < |r| ≤ 0.7, Weak: |r| ≤ 0.4<br/>
        • Positive: variables increase together; Negative: one increases as other decreases<br/>
        • Significant (p<0.05): relationship is statistically reliable
        """
        story.append(Paragraph(interp_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Duration vs Battery Construction Correlations Section
    if duration_correlations and len(duration_correlations) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Duration vs Battery Construction Correlations", heading_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<i>Analyzing how battery construction parameters affect discharge duration</i>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Display values table if present
        if 'values_table' in duration_correlations:
            story.append(Paragraph("<b>Actual Values Used in Analysis:</b>", styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
            
            values_table_data = [[Paragraph('Build', table_header_style), Paragraph('Duration (s)', table_header_style), 
                                  Paragraph('Total Anode (g)', table_header_style), Paragraph('Total Cathode (g)', table_header_style), 
                                  Paragraph('Total Calorific (kJ)', table_header_style)]]
            for idx, row_dict in enumerate(duration_correlations['values_table']):
                values_table_data.append([
                    Paragraph(f"Build {idx+1}", table_cell_left_style),
                    Paragraph(f"{row_dict.get('Duration (s)', 'N/A')}", table_cell_style),
                    Paragraph(f"{row_dict.get('Total Anode Weight (g)', 'N/A')}", table_cell_style),
                    Paragraph(f"{row_dict.get('Total Cathode Weight (g)', 'N/A')}", table_cell_style),
                    Paragraph(f"{row_dict.get('Total Calorific Value (kJ)', 'N/A')}", table_cell_style)
                ])
            
            values_table = Table(values_table_data, colWidths=[1*inch, 1.2*inch, 1.3*inch, 1.3*inch, 1.4*inch])
            values_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2ca02c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightgreen),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(values_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Duration correlation results table
        dur_corr_data = [[Paragraph('Relationship', table_header_style), Paragraph('Correlation', table_header_style), 
                          Paragraph('P-Value', table_header_style), Paragraph('Strength', table_header_style), 
                          Paragraph('Significance', table_header_style)]]
        
        for corr_name, corr_info in duration_correlations.items():
            if corr_name != 'values_table':
                dur_corr_data.append([
                    Paragraph(str(corr_name), table_cell_left_style),
                    Paragraph(f"{corr_info['correlation']:.3f}", table_cell_style),
                    Paragraph(f"{corr_info['p_value']:.4f}", table_cell_style),
                    Paragraph(f"{corr_info['strength']} ({corr_info['direction']})", table_cell_style),
                    Paragraph(corr_info['significance'], table_cell_style)
                ])
        
        if len(dur_corr_data) > 1:
            dur_corr_table = Table(dur_corr_data, colWidths=[2*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1.2*inch])
            dur_corr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#ff7f0e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.HexColor('#ffe4cc')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(dur_corr_table)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph("<b>Interpretation:</b>", styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
            dur_interp_text = """
            • Positive correlation: Increasing the construction parameter increases discharge duration<br/>
            • Negative correlation: Increasing the construction parameter decreases discharge duration<br/>
            • Strong correlation suggests the parameter significantly influences battery performance
            """
            story.append(Paragraph(dur_interp_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def calculate_advanced_analytics(df, time_col, voltage_col, current_col=None):
    """Calculate advanced battery analytics"""
    analytics = {}
    
    if not voltage_col or voltage_col not in df.columns:
        return analytics
    
    if time_col and time_col in df.columns and len(df) > 10:
        voltage_series = df[voltage_col].values
        time_series_raw = df[time_col].values
        
        try:
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                time_series = (pd.to_datetime(df[time_col]) - pd.to_datetime(df[time_col].iloc[0])).dt.total_seconds() / 60
                time_series = time_series.values
            elif pd.api.types.is_timedelta64_dtype(df[time_col]):
                time_series = df[time_col].dt.total_seconds().values / 60
            elif pd.api.types.is_numeric_dtype(df[time_col]):
                time_series_converted = detect_time_unit_and_convert(df[time_col])
                time_series = time_series_converted.values
            else:
                try:
                    time_series_converted = detect_time_unit_and_convert(pd.to_numeric(df[time_col], errors='raise'))
                    time_series = time_series_converted.values
                except:
                    try:
                        time_parsed = pd.to_timedelta(df[time_col])
                        time_series = time_parsed.dt.total_seconds().values / 60
                    except:
                        try:
                            time_parsed = pd.to_datetime(df[time_col])
                            time_series = (time_parsed - time_parsed.iloc[0]).dt.total_seconds() / 60
                            time_series = time_series.values
                        except:
                            return analytics
        except Exception as e:
            return analytics
        
        initial_voltage = voltage_series[0]
        final_voltage = voltage_series[-1]
        total_time = time_series[-1] - time_series[0]
        
        analytics['Initial Voltage (V)'] = initial_voltage
        analytics['Final Voltage (V)'] = final_voltage
        analytics['Total Voltage Drop (V)'] = initial_voltage - final_voltage
        
        if total_time > 0:
            analytics['Average Degradation Rate (mV/min)'] = (initial_voltage - final_voltage) * 1000 / total_time
            
            voltage_at_80_pct = initial_voltage * 0.8
            indices_below_80 = np.where(voltage_series <= voltage_at_80_pct)[0]
            if len(indices_below_80) > 0:
                time_to_80_pct = time_series[indices_below_80[0]] - time_series[0]
                analytics['Time to 80% Voltage (min)'] = time_to_80_pct
            
            voltage_at_50_pct = initial_voltage * 0.5
            indices_below_50 = np.where(voltage_series <= voltage_at_50_pct)[0]
            if len(indices_below_50) > 0:
                time_to_50_pct = time_series[indices_below_50[0]] - time_series[0]
                analytics['Time to 50% Voltage (min)'] = time_to_50_pct
        
        time_diffs = np.diff(time_series)
        voltage_diffs = np.diff(voltage_series)
        
        valid_indices = time_diffs > 0
        if np.any(valid_indices):
            instantaneous_rates = np.zeros_like(voltage_diffs)
            instantaneous_rates[valid_indices] = voltage_diffs[valid_indices] / time_diffs[valid_indices]
            
            valid_rates = instantaneous_rates[valid_indices]
            if len(valid_rates) > 0:
                analytics['Max Discharge Rate (V/min)'] = np.max(np.abs(valid_rates))
                analytics['Min Discharge Rate (V/min)'] = np.min(np.abs(valid_rates))
        
        mid_idx = len(voltage_series) // 2
        analytics['Mid-Point Voltage (V)'] = voltage_series[mid_idx]
        
        voltage_range = initial_voltage - final_voltage
        if total_time > 0:
            if voltage_range > 0:
                eol_voltage = initial_voltage * 0.7
                
                if final_voltage <= eol_voltage:
                    analytics['Battery Status'] = 'Discharged (Below 70% Initial Voltage)'
                else:
                    analytics['Battery Status'] = 'Active Discharge'
            else:
                analytics['Battery Status'] = 'Stable (No Degradation Detected)'
            
            if initial_voltage > 0:
                voltage_retention = (final_voltage / initial_voltage) * 100
                analytics['Voltage Retention (%)'] = voltage_retention
        
        if current_col and current_col in df.columns:
            current_series = df[current_col].values
            power_series = voltage_series * np.abs(current_series)
            
            analytics['Peak Power (W)'] = np.max(power_series)
            analytics['Average Power (W)'] = np.mean(power_series)
            
            if len(time_diffs) > 0 and np.any(valid_indices):
                valid_time_diffs = time_diffs[valid_indices]
                valid_power = power_series[:-1][valid_indices]
                
                energy_wh = np.sum(valid_power * valid_time_diffs) / 60
                analytics['Total Energy Discharged (Wh)'] = energy_wh
                
                if total_time > 0:
                    analytics['Average Discharge Power (W)'] = energy_wh * 60 / total_time
                
                theoretical_energy = initial_voltage * np.mean(np.abs(current_series)) * total_time / 60
                if theoretical_energy > 0:
                    analytics['Energy Efficiency (%)'] = (energy_wh / theoretical_energy) * 100
                
                valid_current = current_series[:-1][valid_indices]
                coulombic_efficiency = np.sum(np.abs(valid_current) * valid_time_diffs) / 60
                analytics['Coulombic Efficiency (Ah)'] = coulombic_efficiency
        
        # Advanced discharge curve analysis: ΔV/ΔT at 5-second intervals
        # Convert time to seconds for 5-second interval analysis
        time_in_seconds = time_series * 60  # Convert from minutes to seconds
        
        # Create 5-second interval bins
        interval_seconds = 5
        max_time_sec = time_in_seconds[-1]
        
        # Check if max_time_sec is valid (not NaN or infinite)
        if not np.isfinite(max_time_sec) or max_time_sec <= 0:
            return analytics
        
        num_intervals = int(max_time_sec / interval_seconds)
        
        if num_intervals > 1:
            delta_v_delta_t_values = []
            interval_times = []
            
            for i in range(num_intervals):
                start_time = i * interval_seconds
                end_time = (i + 1) * interval_seconds
                
                # Find indices in this time interval
                mask = (time_in_seconds >= start_time) & (time_in_seconds < end_time)
                
                if np.sum(mask) >= 2:
                    time_subset = time_in_seconds[mask]
                    voltage_subset = voltage_series[mask]
                    
                    # Calculate ΔV/ΔT for this interval (in V/s)
                    delta_v = voltage_subset[-1] - voltage_subset[0]
                    delta_t = time_subset[-1] - time_subset[0]
                    
                    if delta_t > 0:
                        slope = delta_v / delta_t
                        delta_v_delta_t_values.append(slope)
                        interval_times.append((start_time + end_time) / 2)
            
            if len(delta_v_delta_t_values) > 0:
                # Store the slopes array for further analysis
                analytics['ΔV/ΔT Values (V/s)'] = delta_v_delta_t_values
                analytics['ΔV/ΔT Mean (V/s)'] = float(np.mean(delta_v_delta_t_values))
                analytics['ΔV/ΔT Std Dev (V/s)'] = float(np.std(delta_v_delta_t_values))
                analytics['ΔV/ΔT Max (V/s)'] = float(np.max(np.abs(delta_v_delta_t_values)))
                analytics['ΔV/ΔT Min (V/s)'] = float(np.min(np.abs(delta_v_delta_t_values)))
                
                # Curve stability analysis
                # Calculate coefficient of variation (CV) to assess stability
                mean_slope = np.mean(np.abs(delta_v_delta_t_values))
                std_slope = np.std(delta_v_delta_t_values)
                
                if mean_slope > 0:
                    cv = (std_slope / mean_slope) * 100
                    analytics['Discharge Slope Variability (%)'] = float(cv)
                    
                    # Generate stability commentary
                    if cv < 10:
                        stability_commentary = "Excellent - Very stable discharge curve with minimal variation"
                    elif cv < 25:
                        stability_commentary = "Good - Stable discharge curve with acceptable variation"
                    elif cv < 50:
                        stability_commentary = "Moderate - Some variability in discharge rate"
                    elif cv < 100:
                        stability_commentary = "Poor - Significant variability in discharge rate"
                    else:
                        stability_commentary = "Very Poor - Highly unstable discharge curve"
                    
                    analytics['Curve Stability Assessment'] = stability_commentary
                
                # Check for sudden changes (anomalies in slope)
                slope_changes = np.diff(delta_v_delta_t_values)
                if len(slope_changes) > 0:
                    max_slope_change = np.max(np.abs(slope_changes))
                    analytics['Max Slope Change (V/s²)'] = float(max_slope_change)
                    
                    # Detect if there are abrupt changes
                    threshold = 3 * np.std(slope_changes)
                    abrupt_changes = np.sum(np.abs(slope_changes) > threshold)
                    analytics['Num Abrupt Slope Changes'] = int(abrupt_changes)
                
                # Find longest plateau where ΔV/ΔT is constant (±5% deviation)
                # Use two-pointer approach to find longest contiguous interval
                if len(delta_v_delta_t_values) >= 2:
                    max_plateau_length = 0
                    max_plateau_start_idx = 0
                    max_plateau_end_idx = 0
                    
                    # Absolute tolerance floor for near-zero slopes (V/s)
                    abs_tolerance = 0.001  # 1 mV/s
                    
                    # Try each starting point
                    for start_idx in range(len(delta_v_delta_t_values)):
                        # Find the longest interval starting from this point
                        for end_idx in range(start_idx + 1, len(delta_v_delta_t_values) + 1):
                            segment = delta_v_delta_t_values[start_idx:end_idx]
                            segment_mean_abs = np.mean(np.abs(segment))
                            
                            # Use max of 5% relative tolerance and absolute tolerance
                            tolerance = max(0.05 * segment_mean_abs, abs_tolerance)
                            
                            # Check if all values in segment are within tolerance of segment mean
                            deviations_abs = np.abs(np.abs(segment) - segment_mean_abs)
                            if np.all(deviations_abs <= tolerance):
                                # This segment is valid, check if it's the longest
                                if len(segment) > max_plateau_length:
                                    max_plateau_length = len(segment)
                                    max_plateau_start_idx = start_idx
                                    max_plateau_end_idx = end_idx - 1
                            else:
                                # Segment failed, no point checking longer segments from this start
                                break
                    
                    if max_plateau_length >= 2:
                        # Convert indices to time in seconds
                        plateau_start_time = interval_times[max_plateau_start_idx]
                        plateau_end_time = interval_times[max_plateau_end_idx]
                        
                        analytics['Constant ΔV/ΔT Region Start (s)'] = float(plateau_start_time)
                        analytics['Constant ΔV/ΔT Region End (s)'] = float(plateau_end_time)
                        analytics['Constant ΔV/ΔT Duration (s)'] = float(plateau_end_time - plateau_start_time)
                        analytics['Constant ΔV/ΔT Region'] = f"from {plateau_start_time:.1f}s to {plateau_end_time:.1f}s ({plateau_end_time - plateau_start_time:.1f}s duration)"
                    else:
                        analytics['Constant ΔV/ΔT Region'] = "No constant region found (ΔV/ΔT varies by >5%)"
    
    return analytics

def calculate_duration_correlations(all_metrics_list, all_extended_metadata_list):
    """
    Calculate correlations between actual duration and total weights/calorific value.
    
    Args:
        all_metrics_list: List of metric dictionaries for each build
        all_extended_metadata_list: List of extended metadata dictionaries for each build
    
    Returns:
        Dictionary with correlation results and actual values used
    """
    correlations = {}
    
    if len(all_metrics_list) < 2:
        return correlations
    
    # Extract data
    durations = []
    total_anode_weights = []
    total_cathode_weights = []
    total_calorific_values = []
    build_info = []  # Store actual values for display
    
    for metrics, ext_meta in zip(all_metrics_list, all_extended_metadata_list):
        duration = metrics.get('Duration (Sec)')
        if duration is None or not ext_meta:
            continue
        
        # Get pre-calculated total weights from extended metadata
        # These are already calculated correctly with stacks_in_parallel multiplier
        total_anode = ext_meta.get('total_anode_weight', 0)
        total_cathode = ext_meta.get('total_cathode_weight', 0)
        
        # Get pre-calculated total calorific value (already converted from cal to kJ)
        # Formula: total_heat_pellet_weight × calorific_value_cal_per_g × 0.004184
        total_calorific = ext_meta.get('total_calorific_value', 0)
        
        # Store data
        durations.append(duration)
        total_anode_weights.append(total_anode)
        total_cathode_weights.append(total_cathode)
        total_calorific_values.append(total_calorific)
        build_info.append({
            'Duration (s)': duration,
            'Total Anode Weight (g)': total_anode,
            'Total Cathode Weight (g)': total_cathode,
            'Total Calorific Value (kJ)': total_calorific
        })
    
    # Calculate correlations if we have enough valid data
    if len(durations) >= 2:
        correlations['values_table'] = build_info
        
        # Duration vs Total Anode Weight
        if len(total_anode_weights) >= 2 and all(w > 0 for w in total_anode_weights):
            try:
                corr, p_val = stats.pearsonr(durations, total_anode_weights)
                correlations['Duration vs Total Anode Weight'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak',
                    'direction': 'Positive' if corr > 0 else 'Negative',
                    'significance': 'Significant (p<0.05)' if p_val < 0.05 else 'Not significant'
                }
            except:
                pass
        
        # Duration vs Total Cathode Weight
        if len(total_cathode_weights) >= 2 and all(w > 0 for w in total_cathode_weights):
            try:
                corr, p_val = stats.pearsonr(durations, total_cathode_weights)
                correlations['Duration vs Total Cathode Weight'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak',
                    'direction': 'Positive' if corr > 0 else 'Negative',
                    'significance': 'Significant (p<0.05)' if p_val < 0.05 else 'Not significant'
                }
            except:
                pass
        
        # Duration vs Total Calorific Value
        if len(total_calorific_values) >= 2 and all(c > 0 for c in total_calorific_values):
            try:
                corr, p_val = stats.pearsonr(durations, total_calorific_values)
                correlations['Duration vs Total Calorific Value'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.4 else 'Weak',
                    'direction': 'Positive' if corr > 0 else 'Negative',
                    'significance': 'Significant (p<0.05)' if p_val < 0.05 else 'Not significant'
                }
            except:
                pass
    
    return correlations

def calculate_correlation_analysis(all_metrics_list, all_analytics_list):
    """
    Calculate correlation analysis between performance metrics and discharge curve characteristics.
    
    Args:
        all_metrics_list: List of metric dictionaries for each build
        all_analytics_list: List of analytics dictionaries for each build
    
    Returns:
        Dictionary with correlation results and insights
    """
    correlations = {}
    
    if len(all_metrics_list) < 2:
        return correlations  # Need at least 2 builds for correlation
    
    # Extract data for correlation analysis
    ampere_secs_anode = []
    ampere_secs_cathode = []
    mean_slopes = []
    slope_variabilities = []
    
    for metrics, analytics in zip(all_metrics_list, all_analytics_list):
        # Ampere-seconds per gram metrics
        if metrics.get('A·s per gram Anode') is not None:
            ampere_secs_anode.append(metrics['A·s per gram Anode'])
        if metrics.get('A·s per gram Cathode') is not None:
            ampere_secs_cathode.append(metrics['A·s per gram Cathode'])
        
        # Discharge curve characteristics
        if analytics.get('ΔV/ΔT Mean (V/s)') is not None:
            mean_slopes.append(abs(analytics['ΔV/ΔT Mean (V/s)']))
        if analytics.get('Discharge Slope Variability (%)') is not None:
            slope_variabilities.append(analytics['Discharge Slope Variability (%)'])
    
    # Calculate correlations if we have enough data
    if len(ampere_secs_anode) >= 2 and len(mean_slopes) >= 2 and len(ampere_secs_anode) == len(mean_slopes):
        try:
            corr_coef, p_value = stats.pearsonr(ampere_secs_anode, mean_slopes)
            correlations['A·s/g Anode vs Mean Slope'] = {
                'correlation': float(corr_coef),
                'p_value': float(p_value),
                'strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.4 else 'Weak',
                'direction': 'Positive' if corr_coef > 0 else 'Negative',
                'significance': 'Significant (p<0.05)' if p_value < 0.05 else 'Not significant'
            }
        except:
            pass
    
    if len(ampere_secs_cathode) >= 2 and len(mean_slopes) >= 2 and len(ampere_secs_cathode) == len(mean_slopes):
        try:
            corr_coef, p_value = stats.pearsonr(ampere_secs_cathode, mean_slopes)
            correlations['A·s/g Cathode vs Mean Slope'] = {
                'correlation': float(corr_coef),
                'p_value': float(p_value),
                'strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.4 else 'Weak',
                'direction': 'Positive' if corr_coef > 0 else 'Negative',
                'significance': 'Significant (p<0.05)' if p_value < 0.05 else 'Not significant'
            }
        except:
            pass
    
    # Add energy density correlations if data available
    if len(ampere_secs_anode) >= 2 and len(ampere_secs_cathode) >= 2 and len(ampere_secs_anode) == len(ampere_secs_cathode):
        try:
            corr_coef, p_value = stats.pearsonr(ampere_secs_anode, ampere_secs_cathode)
            correlations['Anode vs Cathode Performance'] = {
                'correlation': float(corr_coef),
                'p_value': float(p_value),
                'strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.4 else 'Weak',
                'direction': 'Positive' if corr_coef > 0 else 'Negative',
                'significance': 'Significant (p<0.05)' if p_value < 0.05 else 'Not significant'
            }
        except:
            pass
    
    return correlations

def detect_anomalies(df, voltage_col, time_col=None):
    """Detect anomalies in discharge data using statistical methods"""
    anomalies = []
    
    if not voltage_col or voltage_col not in df.columns:
        return anomalies, []
    
    voltage_series = df[voltage_col].values
    
    if len(voltage_series) < 10:
        return anomalies, []
    
    voltage_diffs = np.diff(voltage_series)
    
    sudden_drops = []
    threshold = 3 * np.std(voltage_diffs)
    mean_diff = np.mean(voltage_diffs)
    
    for i, diff in enumerate(voltage_diffs):
        if diff < mean_diff - threshold:
            sudden_drops.append(i + 1)
            anomalies.append(f"Sudden voltage drop at index {i+1}: {diff:.4f}V")
    
    rolling_window = min(20, len(voltage_series) // 10)
    rolling_mean = pd.Series(voltage_series).rolling(window=rolling_window, center=True).mean()
    rolling_std = pd.Series(voltage_series).rolling(window=rolling_window, center=True).std()
    
    outlier_indices = []
    for i in range(len(voltage_series)):
        if not np.isnan(rolling_mean.iloc[i]) and not np.isnan(rolling_std.iloc[i]):
            if abs(voltage_series[i] - rolling_mean.iloc[i]) > 3 * rolling_std.iloc[i]:
                outlier_indices.append(i)
                anomalies.append(f"Statistical outlier at index {i}: {voltage_series[i]:.4f}V")
    
    if len(voltage_series) > 20:
        expected_trend = np.linspace(voltage_series[0], voltage_series[-1], len(voltage_series))
        deviations = voltage_series - expected_trend
        
        large_deviations = np.where(np.abs(deviations) > 2 * np.std(deviations))[0]
        for idx in large_deviations:
            if idx not in outlier_indices:
                anomalies.append(f"Unexpected behavior at index {idx}: deviation {deviations[idx]:.4f}V")
    
    all_anomaly_indices = sorted(set(sudden_drops + outlier_indices + list(large_deviations) if len(voltage_series) > 20 else sudden_drops + outlier_indices))
    
    return anomalies, all_anomaly_indices

def save_comparison_to_db(name, dataframes, build_names, metrics_df, battery_type=None, extended_metadata=None, password=None, standard_params=None):
    """Save comparison to database with optional password protection and standard parameters"""
    if not Session:
        return False, "Database not configured"
    
    try:
        session = Session()
        
        data_list = []
        for df in dataframes:
            data_list.append(df.to_json(orient='split'))
        
        metrics_json = metrics_df.to_json(orient='split') if metrics_df is not None else None
        extended_metadata_json = json.dumps(extended_metadata) if extended_metadata else None
        standard_params_json = json.dumps(standard_params) if standard_params else None
        
        # Hash password if provided
        password_hash = None
        if password and password.strip():
            try:
                password_hash = ph.hash(password.strip())
            except Exception as e:
                session.close()
                return False, f"Error hashing password: {str(e)}"
        
        comparison = SavedComparison(
            name=name,
            num_builds=len(build_names),
            build_names=json.dumps(build_names),
            data_json=json.dumps(data_list),
            metrics_json=metrics_json,
            battery_type=battery_type,
            extended_metadata_json=extended_metadata_json,
            password_hash=password_hash,
            standard_params_json=standard_params_json
        )
        
        session.add(comparison)
        session.commit()
        session.close()
        
        protection_msg = " (password protected)" if password_hash else ""
        return True, f"Comparison saved successfully{protection_msg}"
    except Exception as e:
        return False, f"Error saving comparison: {str(e)}"

def update_comparison_in_db(comparison_id, name, dataframes, build_names, metrics_df, battery_type=None, extended_metadata=None, existing_password=None, new_password=None, standard_params=None):
    """Update existing comparison in database with password verification"""
    if not Session:
        return False, "Database not configured"
    
    try:
        session = Session()
        comparison = session.query(SavedComparison).filter_by(id=comparison_id).first()
        
        if not comparison:
            session.close()
            return False, "Comparison not found"
        
        # Verify existing password if comparison is protected
        if comparison.password_hash:
            if not existing_password or not existing_password.strip():
                session.close()
                return False, "Password required to update this protected comparison"
            
            try:
                ph.verify(comparison.password_hash, existing_password.strip())
            except VerifyMismatchError:
                session.close()
                return False, "Incorrect password"
            except Exception as e:
                session.close()
                return False, f"Error verifying password: {str(e)}"
        
        # Prepare data
        data_list = []
        for df in dataframes:
            data_list.append(df.to_json(orient='split'))
        
        metrics_json = metrics_df.to_json(orient='split') if metrics_df is not None else None
        extended_metadata_json = json.dumps(extended_metadata) if extended_metadata else None
        
        # Update comparison fields
        comparison.name = name
        comparison.num_builds = len(build_names)
        comparison.build_names = json.dumps(build_names)
        comparison.data_json = json.dumps(data_list)
        comparison.metrics_json = metrics_json
        comparison.battery_type = battery_type
        comparison.extended_metadata_json = extended_metadata_json
        comparison.standard_params_json = json.dumps(standard_params) if standard_params else None
        
        # Update password if new one provided
        if new_password and new_password.strip():
            try:
                comparison.password_hash = ph.hash(new_password.strip())
            except Exception as e:
                session.close()
                return False, f"Error hashing new password: {str(e)}"
        
        # Check password protection status before closing session
        is_protected = comparison.password_hash is not None
        
        session.commit()
        session.close()
        
        protection_msg = " (password protected)" if is_protected else ""
        return True, f"Comparison updated successfully{protection_msg}"
    except Exception as e:
        return False, f"Error updating comparison: {str(e)}"

def load_comparison_from_db(comparison_id, password=None):
    """Load comparison from database with optional password verification"""
    if not Session:
        return None, None, None, None, None, "Database not configured"
    
    try:
        session = Session()
        comparison = session.query(SavedComparison).filter_by(id=comparison_id).first()
        
        if not comparison:
            session.close()
            return None, None, None, None, None, "Comparison not found"
        
        # Check password if comparison is protected
        if comparison.password_hash:
            if not password or not password.strip():
                session.close()
                return None, None, None, None, None, "Password required for this comparison"
            
            try:
                ph.verify(comparison.password_hash, password.strip())
            except VerifyMismatchError:
                session.close()
                return None, None, None, None, None, "Incorrect password"
            except Exception as e:
                session.close()
                return None, None, None, None, None, f"Error verifying password: {str(e)}"
        
        build_names = json.loads(comparison.build_names)
        data_list = json.loads(comparison.data_json)
        
        # MIGRATION: Ensure unique build names for legacy comparisons with duplicates
        unique_build_names = []
        seen = {}
        for name in build_names:
            if name in seen:
                seen[name] += 1
                unique_name = f"{name} ({seen[name]})"
            else:
                seen[name] = 0
                unique_name = name
            unique_build_names.append(unique_name)
        build_names = unique_build_names
        
        dataframes = []
        for data_json in data_list:
            df = pd.read_json(io.StringIO(data_json), orient='split')
            dataframes.append(df)
        
        metrics_df = None
        if comparison.metrics_json:
            metrics_df = pd.read_json(io.StringIO(comparison.metrics_json), orient='split')
        
        battery_type = comparison.battery_type if hasattr(comparison, 'battery_type') else None
        extended_metadata = json.loads(comparison.extended_metadata_json) if comparison.extended_metadata_json else {}
        standard_params = json.loads(comparison.standard_params_json) if hasattr(comparison, 'standard_params_json') and comparison.standard_params_json else {}
        
        session.close()
        
        return dataframes, build_names, metrics_df, battery_type, extended_metadata, standard_params, "Loaded successfully"
    except Exception as e:
        return None, None, None, None, None, None, f"Error loading comparison: {str(e)}"

def get_all_saved_comparisons():
    """Get list of all saved comparisons with password protection status"""
    if not Session:
        return []
    
    try:
        session = Session()
        comparisons = session.query(SavedComparison).order_by(SavedComparison.created_at.desc()).all()
        
        result = []
        for comp in comparisons:
            result.append({
                'id': comp.id,
                'name': comp.name,
                'created_at': comp.created_at,
                'num_builds': comp.num_builds,
                'is_protected': comp.password_hash is not None
            })
        
        session.close()
        return result
    except Exception as e:
        st.error(f"Error fetching saved comparisons: {str(e)}")
        return []

def delete_comparison_from_db(comparison_id):
    """Delete a comparison from database"""
    if not Session:
        return False, "Database not configured"
    
    try:
        session = Session()
        comparison = session.query(SavedComparison).filter_by(id=comparison_id).first()
        
        if comparison:
            session.delete(comparison)
            session.commit()
            session.close()
            return True, "Comparison deleted successfully"
        else:
            session.close()
            return False, "Comparison not found"
    except Exception as e:
        return False, f"Error deleting comparison: {str(e)}"

st.sidebar.header("🔋 Battery Type Selection")
st.sidebar.markdown("Select the battery type for this analysis session")

if 'battery_type' not in st.session_state:
    st.session_state['battery_type'] = "General"

battery_type = st.sidebar.selectbox(
    "Battery Type:",
    options=["General", "Madhava", "Low-Voltage (0.9-1.5V)", "High-Voltage (27-35V)", "Custom"],
    key='battery_type',
    help="Select the battery type. All data and analysis will be organized by this type."
)

if battery_type == "Custom":
    custom_battery_name = st.sidebar.text_input(
        "Custom Battery Name:",
        value="",
        key='custom_battery_name',
        help="Enter a custom name for your battery type"
    )
    if custom_battery_name:
        battery_type = custom_battery_name

st.sidebar.info(f"📊 Current Battery Type: **{battery_type}**")

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Input Mode")

data_mode = st.sidebar.radio(
    "Select input mode:",
    ["Upload Files", "Real-Time Streaming"],
    key='data_mode'
)

st.sidebar.markdown("---")

if DATABASE_URL and Session:
    st.sidebar.markdown("### 💾 Load Saved Comparison")
    saved_comparisons = get_all_saved_comparisons()
    
    if saved_comparisons:
        # Create comparison options with 🔒 icon for protected comparisons
        comparison_options = {}
        comparison_metadata = {}
        for comp in saved_comparisons:
            lock_icon = "🔒 " if comp['is_protected'] else ""
            display_name = f"{lock_icon}{comp['name']} ({comp['created_at'].strftime('%Y-%m-%d %H:%M')})"
            comparison_options[display_name] = comp['id']
            comparison_metadata[comp['id']] = {
                'name': comp['name'],
                'is_protected': comp['is_protected']
            }
        
        selected_comparison = st.sidebar.selectbox(
            "Select a saved comparison:",
            options=['-- New Comparison --'] + list(comparison_options.keys()),
            key='load_comparison'
        )
        
        if selected_comparison != '-- New Comparison --':
            comparison_id = comparison_options[selected_comparison]
            comp_meta = comparison_metadata[comparison_id]
            
            # Show password input if comparison is protected
            load_password = None
            if comp_meta['is_protected']:
                load_password = st.sidebar.text_input(
                    "🔐 Password:",
                    type="password",
                    key='load_password_input',
                    help="This comparison is password-protected. Enter password to load."
                )
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📂 Load", key='load_btn'):
                    try:
                        loaded_dfs, loaded_names, loaded_metrics, loaded_battery_type, loaded_extended_meta, loaded_standard_params, msg = load_comparison_from_db(comparison_id, password=load_password)
                        if loaded_dfs:
                            st.session_state['loaded_dataframes'] = loaded_dfs
                            st.session_state['loaded_build_names'] = loaded_names
                            st.session_state['editing_comparison_id'] = comparison_id
                            st.session_state['editing_comparison_name'] = comp_meta['name']
                            st.session_state['editing_comparison_protected'] = comp_meta['is_protected']
                            if loaded_battery_type:
                                st.session_state['loaded_battery_type_preference'] = loaded_battery_type
                            if loaded_extended_meta:
                                st.session_state['build_metadata_extended'] = loaded_extended_meta
                            if loaded_standard_params:
                                st.session_state['loaded_standard_params'] = loaded_standard_params
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Error loading comparison: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            with col2:
                if st.button("🗑️ Delete", key='delete_btn'):
                    success, msg = delete_comparison_from_db(comparison_id)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    
    st.sidebar.markdown("---")

uploaded_files = []
build_names = []
dataframes = []
metadata_list = []

# Default performance specifications
min_activation_voltage = 1.0
use_standards = False
std_max_onload_voltage = None
std_max_oc_voltage = None
std_activation_time = None
std_duration = None

if data_mode == "Upload Files":
    st.sidebar.markdown("Upload battery discharge data files for comparison")
    
    num_builds = st.sidebar.number_input("Number of builds to compare:", min_value=1, max_value=50, value=2, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Performance Specifications")
    
    # Get standard params from loaded comparison, extracted file, or use defaults
    loaded_params = st.session_state.get('loaded_standard_params', {})
    extracted_params = st.session_state.get('extracted_standard_params', {})
    # Priority: loaded > extracted > default
    params_source = loaded_params if loaded_params else extracted_params
    default_min_voltage = params_source.get('min_activation_voltage', 1.0) if params_source else 1.0
    
    min_activation_voltage = st.sidebar.number_input(
        "Min. voltage for activation (V):",
        min_value=0.0,
        max_value=100.0,
        value=float(default_min_voltage),
        step=0.1,
        help="Minimum voltage threshold. Activation Time (Sec) = time when battery FIRST reaches ≥ this voltage. Duration (Sec) = time from first activation to last occurrence of cutoff voltage."
    )
    
    # Show info if parameters were auto-filled
    if loaded_params and any(v is not None for v in loaded_params.values()):
        st.sidebar.info("💾 Parameters restored from saved comparison")
    elif extracted_params and any(v is not None for v in extracted_params.values()):
        st.sidebar.info("📄 Parameters auto-loaded from Excel file")
    
    # Optional target duration for voltage lookup
    use_target_duration = st.sidebar.checkbox(
        "Specify target duration for voltage lookup",
        value=False,
        help="Enable to interpolate and display voltage at a specific time point during discharge"
    )
    
    target_duration_sec = None
    if use_target_duration:
        target_duration_sec = st.sidebar.number_input(
            "Target duration (seconds):",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0,
            help="Voltage will be interpolated at this many seconds after activation"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Standard Performance Benchmarks")
    st.sidebar.markdown("*Optional: Set target values for comparison*")
    
    use_standards = st.sidebar.checkbox("Enable standard performance comparison", value=False)
    
    if use_standards:
        use_onload_std = st.sidebar.checkbox("Set Std. Max On-Load Voltage", value=False, help="Optional: Enable if you want to compare against on-load voltage standard")
        
        if use_onload_std:
            std_max_onload_voltage = st.sidebar.number_input(
                "Std. Max On-Load Voltage (V):",
                min_value=0.0,
                max_value=100.0,
                value=1.5,
                step=0.1
            )
        else:
            std_max_onload_voltage = None
        
        # Use loaded/extracted values if available, otherwise defaults
        default_oc_voltage = params_source.get('std_max_oc_voltage', 1.6) if params_source else 1.6
        default_activation_time_ms = params_source.get('std_activation_time_ms', 1000.0) if params_source else 1000.0
        default_duration_sec = params_source.get('std_duration_sec', 400.0) if params_source else 400.0
        
        std_max_oc_voltage = st.sidebar.number_input(
            "Std. Max Open Circuit Voltage (V):",
            min_value=0.0,
            max_value=100.0,
            value=float(default_oc_voltage),
            step=0.1
        )
        
        std_activation_time_ms = st.sidebar.number_input(
            "Std. Max Activation Time (ms):",
            min_value=0.0,
            max_value=100000.0,
            value=float(default_activation_time_ms),
            step=10.0,
            help="Maximum acceptable time to reach min voltage in milliseconds"
        )
        # Convert ms to minutes for comparison with metrics
        std_activation_time = std_activation_time_ms / 60000.0 if std_activation_time_ms > 0 else None
        
        std_duration_sec = st.sidebar.number_input(
            "Target Min Duration (s):",
            min_value=0.0,
            max_value=100000.0,
            value=float(default_duration_sec),
            step=10.0,
            help="Target minimum acceptable discharge duration in seconds"
        )
        # Convert seconds to minutes for comparison with metrics
        std_duration = std_duration_sec / 60.0 if std_duration_sec > 0 else None
    else:
        std_max_onload_voltage = None
        std_max_oc_voltage = None
        std_activation_time = None
        std_duration = None
        std_activation_time_ms = None
        std_duration_sec = None
    
    st.sidebar.markdown("---")
    
    if 'build_metadata_extended' not in st.session_state:
        st.session_state['build_metadata_extended'] = {}
    
    # Initialize all form field session_state keys with defaults BEFORE widgets are rendered
    for i in range(num_builds):
        # Only initialize if not already set (preserves existing values and allows file extraction to update)
        st.session_state.setdefault(f'anode_weight_{i}', 0.0)
        st.session_state.setdefault(f'cathode_weight_{i}', 0.0)
        st.session_state.setdefault(f'heat_pellet_{i}', 0.0)
        st.session_state.setdefault(f'electrolyte_{i}', 0.0)
        st.session_state.setdefault(f'cells_series_{i}', 1)
        st.session_state.setdefault(f'stacks_parallel_{i}', 1)
        st.session_state.setdefault(f'calorific_value_{i}', 0.0)
    
    if 'loaded_dataframes' in st.session_state and 'loaded_build_names' in st.session_state:
        dataframes = st.session_state['loaded_dataframes']
        build_names = st.session_state['loaded_build_names']
        metadata_list = st.session_state.get('loaded_metadata', [{}] * len(dataframes))
        num_builds = len(dataframes)
        
        st.sidebar.success(f"✅ Loaded {num_builds} builds from database")
        
        # Show editing mode indicator if comparison was loaded for editing
        if 'editing_comparison_name' in st.session_state:
            st.sidebar.info(f"📝 Editing: **{st.session_state['editing_comparison_name']}**")
        
        # Show battery type preference from loaded comparison
        if 'loaded_battery_type_preference' in st.session_state:
            loaded_type = st.session_state['loaded_battery_type_preference']
            current_type = st.session_state.get('battery_type', 'General')
            if loaded_type != current_type:
                st.sidebar.warning(f"💡 This comparison was saved with battery type: **{loaded_type}**. Current type: **{current_type}**")
        
        if st.sidebar.button("Clear Loaded Data"):
            del st.session_state['loaded_dataframes']
            del st.session_state['loaded_build_names']
            if 'loaded_metadata' in st.session_state:
                del st.session_state['loaded_metadata']
            if 'editing_comparison_id' in st.session_state:
                del st.session_state['editing_comparison_id']
            if 'editing_comparison_name' in st.session_state:
                del st.session_state['editing_comparison_name']
            if 'editing_comparison_protected' in st.session_state:
                del st.session_state['editing_comparison_protected']
            if 'loaded_battery_type_preference' in st.session_state:
                del st.session_state['loaded_battery_type_preference']
            st.rerun()
    else:
        # Upload mode toggle
        use_multi_build_file = st.sidebar.checkbox(
            "📁 Single file for all builds", 
            value=True,
            help="Upload one Excel file containing all builds arranged horizontally. Each build should have a 'Build Number' label in the metadata."
        )
        
        # Clear stale multi-build state when switching to individual file mode
        if not use_multi_build_file and 'multi_build_results' in st.session_state:
            del st.session_state['multi_build_results']
            if 'multi_build_processed' in st.session_state:
                del st.session_state['multi_build_processed']
            if 'multi_build_file_id' in st.session_state:
                del st.session_state['multi_build_file_id']
        
        if use_multi_build_file:
            # Multi-build file upload mode
            st.sidebar.markdown("### Upload Multi-Build File")
            st.sidebar.info("Upload an Excel file with all builds arranged horizontally. Each build section should have 'Build Number' in the metadata.")
            
            multi_build_file = st.sidebar.file_uploader(
                "Upload Excel file with all builds",
                type=['xlsx', 'xls'],
                key="multi_build_file"
            )
            
            # Detect file change by comparing file IDs
            current_file_id = multi_build_file.file_id if multi_build_file else None
            previous_file_id = st.session_state.get('multi_build_file_id')
            file_changed = current_file_id is not None and current_file_id != previous_file_id
            
            # Process file if new or changed
            if multi_build_file and (file_changed or 'multi_build_processed' not in st.session_state):
                # Process the multi-build file
                build_results = load_multi_build_file(multi_build_file)
                
                if build_results:
                    st.session_state['multi_build_processed'] = True
                    st.session_state['multi_build_results'] = build_results
                    st.session_state['multi_build_file_id'] = current_file_id
                    st.session_state['multi_build_success'] = len(build_results)
                    
                    # Update num_builds based on detected builds
                    num_builds = len(build_results)
                    
                    # Initialize session_state for each build
                    for i, (df, metadata, std_params, ext_meta) in enumerate(build_results):
                        # Update extended metadata session_state
                        if ext_meta:
                            st.session_state[f'anode_weight_{i}'] = float(ext_meta.get('anode_weight_per_cell', 0) or 0)
                            st.session_state[f'cathode_weight_{i}'] = float(ext_meta.get('cathode_weight_per_cell', 0) or 0)
                            st.session_state[f'heat_pellet_{i}'] = float(ext_meta.get('heat_pellet_weight', 0) or 0)
                            st.session_state[f'electrolyte_{i}'] = float(ext_meta.get('electrolyte_weight', 0) or 0)
                            st.session_state[f'cells_series_{i}'] = int(ext_meta.get('cells_in_series', 1) or 1)
                            st.session_state[f'stacks_parallel_{i}'] = int(ext_meta.get('stacks_in_parallel', 1) or 1)
                            st.session_state[f'calorific_value_{i}'] = float(ext_meta.get('calorific_value_per_gram', 0) or 0)
                    
                    # Rerun to update UI immediately after processing
                    st.rerun()
            
            # Show success message after rerun
            if 'multi_build_success' in st.session_state:
                st.sidebar.success(f"✅ Detected {st.session_state['multi_build_success']} builds in file!")
            
            # Display detected builds and their metadata forms
            if 'multi_build_results' in st.session_state:
                build_results = st.session_state['multi_build_results']
                num_builds = len(build_results)
                
                for i, (df, metadata, std_params, ext_meta) in enumerate(build_results):
                    build_id = metadata.get('build_id', f'Build {i+1}')
                    battery_code = metadata.get('battery_code', '')
                    temperature = metadata.get('temperature', '')
                    
                    build_label = f"Build {build_id}"
                    if battery_code:
                        build_label += f" ({battery_code})"
                    if temperature:
                        build_label += f" @ {temperature}°C"
                    
                    st.sidebar.markdown(f"### {build_label}")
                    st.sidebar.caption(f"📊 {len(df)} data points")
                    
                    # Show metadata form for this build
                    with st.sidebar.expander("⚙️ Extended Build Metadata", expanded=False):
                        st.markdown("**Weight Inputs (per cell)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            anode_weight = st.number_input(
                                "Anode weight (g):", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                step=0.01,
                                key=f"anode_weight_{i}",
                                help="Weight of anode material per cell in grams"
                            )
                            cathode_weight = st.number_input(
                                "Cathode weight (g):", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                step=0.01,
                                key=f"cathode_weight_{i}",
                                help="Weight of cathode material per cell in grams"
                            )
                        with col2:
                            heat_pellet_weight = st.number_input(
                                "Heat pellet (g):", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                step=0.01,
                                key=f"heat_pellet_{i}",
                                help="Weight of heat pellet in grams"
                            )
                            electrolyte_weight = st.number_input(
                                "Electrolyte (g):", 
                                min_value=0.0, 
                                max_value=1000.0, 
                                step=0.01,
                                key=f"electrolyte_{i}",
                                help="Weight of electrolyte in grams"
                            )
                        
                        st.markdown("**Cell Configuration**")
                        col3, col4 = st.columns(2)
                        with col3:
                            cells_in_series = st.number_input(
                                "Cells in series:", 
                                min_value=1, 
                                max_value=100, 
                                step=1,
                                key=f"cells_series_{i}",
                                help="Number of cells connected in series"
                            )
                        with col4:
                            stacks_in_parallel = st.number_input(
                                "Stacks in parallel:", 
                                min_value=1, 
                                max_value=100, 
                                step=1,
                                key=f"stacks_parallel_{i}",
                                help="Number of stacks connected in parallel"
                            )
                        
                        st.markdown("**Energy Input**")
                        calorific_value = st.number_input(
                            "Calorific value (cal/g):", 
                            min_value=0.0, 
                            max_value=10000.0, 
                            step=1.0,
                            key=f"calorific_value_{i}",
                            help="Calorific value per gram in calories/g (will be converted to kJ)"
                        )
                        
                        # Calculate totals
                        total_anode = anode_weight * cells_in_series * stacks_in_parallel if anode_weight > 0 else 0
                        total_cathode = cathode_weight * cells_in_series * stacks_in_parallel if cathode_weight > 0 else 0
                        total_heat_pellet = heat_pellet_weight * cells_in_series * stacks_in_parallel if heat_pellet_weight > 0 else 0
                        total_electrolyte = electrolyte_weight * cells_in_series * stacks_in_parallel if electrolyte_weight > 0 else 0
                        total_stack_weight = (anode_weight + cathode_weight + heat_pellet_weight + electrolyte_weight) * cells_in_series * stacks_in_parallel if cells_in_series > 0 and stacks_in_parallel > 0 else 0
                        total_calorific = total_heat_pellet * calorific_value * 0.004184 if calorific_value > 0 and total_heat_pellet > 0 else 0
                        
                        st.session_state['build_metadata_extended'][i] = {
                            'anode_weight_per_cell': anode_weight,
                            'cathode_weight_per_cell': cathode_weight,
                            'heat_pellet_weight': heat_pellet_weight,
                            'electrolyte_weight': electrolyte_weight,
                            'cells_in_series': cells_in_series,
                            'stacks_in_parallel': stacks_in_parallel,
                            'calorific_value_per_gram': calorific_value,
                            'total_anode_weight': total_anode,
                            'total_cathode_weight': total_cathode,
                            'total_heat_pellet_weight': total_heat_pellet,
                            'total_electrolyte_weight': total_electrolyte,
                            'total_stack_weight': total_stack_weight,
                            'total_calorific_value': total_calorific
                        }
                
                # Clear button
                if st.sidebar.button("🗑️ Clear Multi-Build Data"):
                    if 'multi_build_processed' in st.session_state:
                        del st.session_state['multi_build_processed']
                    if 'multi_build_results' in st.session_state:
                        del st.session_state['multi_build_results']
                    if 'multi_build_file_id' in st.session_state:
                        del st.session_state['multi_build_file_id']
                    if 'multi_build_success' in st.session_state:
                        del st.session_state['multi_build_success']
                    st.rerun()
        else:
            # Individual file upload mode (original behavior)
            uploaded_files = []
            for i in range(num_builds):
                st.sidebar.markdown(f"### Build {i+1}")
                build_name = st.sidebar.text_input(f"Build {i+1} name:", value=f"Build {i+1}", key=f"name_{i}")
                uploaded_file = st.sidebar.file_uploader(
                    f"Upload discharge data for {build_name}",
                    type=['csv', 'xlsx', 'xls'],
                    key=f"file_{i}"
                )
                uploaded_files.append(uploaded_file)
                
                # Process uploaded file and extract metadata BEFORE form widgets are rendered
                if uploaded_file and f'file_processed_{i}_{uploaded_file.file_id}' not in st.session_state:
                    try:
                        df, metadata, standard_params, file_extended_metadata = load_data(uploaded_file)
                        if df is not None and file_extended_metadata and any(v is not None for v in file_extended_metadata.values()):
                            # Extract values
                            cells_in_series = file_extended_metadata.get('cells_in_series', 0) or 0
                            stacks_in_parallel = file_extended_metadata.get('stacks_in_parallel', 0) or 0
                            anode_per_cell = file_extended_metadata.get('anode_weight_per_cell', 0) or 0
                            cathode_per_cell = file_extended_metadata.get('cathode_weight_per_cell', 0) or 0
                            heat_pellet = file_extended_metadata.get('heat_pellet_weight', 0) or 0
                            electrolyte = file_extended_metadata.get('electrolyte_weight', 0) or 0
                            calorific = file_extended_metadata.get('calorific_value_per_gram', 0) or 0
                            
                            # Update session_state for form fields (before widgets are created)
                            st.session_state[f'anode_weight_{i}'] = float(anode_per_cell)
                            st.session_state[f'cathode_weight_{i}'] = float(cathode_per_cell)
                            st.session_state[f'heat_pellet_{i}'] = float(heat_pellet)
                            st.session_state[f'electrolyte_{i}'] = float(electrolyte)
                            st.session_state[f'cells_series_{i}'] = int(cells_in_series)
                            st.session_state[f'stacks_parallel_{i}'] = int(stacks_in_parallel)
                            st.session_state[f'calorific_value_{i}'] = float(calorific)
                            st.session_state[f'file_processed_{i}_{uploaded_file.file_id}'] = True
                            st.info(f"✅ Metadata extracted from Excel file for Build {i+1}")
                    except Exception as e:
                        st.error(f"Error processing file for Build {i+1}: {str(e)}")
            
            # Render metadata form for this build (appears right after its file uploader)
            with st.sidebar.expander("⚙️ Extended Build Metadata (Optional)", expanded=False):
                st.markdown("**Weight Inputs (per cell)**")
                col1, col2 = st.columns(2)
                with col1:
                    anode_weight = st.number_input(
                        "Anode weight (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        step=0.01,
                        key=f"anode_weight_{i}",
                        help="Weight of anode material per cell in grams"
                    )
                    cathode_weight = st.number_input(
                        "Cathode weight (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        step=0.01,
                        key=f"cathode_weight_{i}",
                        help="Weight of cathode material per cell in grams"
                    )
                with col2:
                    heat_pellet_weight = st.number_input(
                        "Heat pellet (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        step=0.01,
                        key=f"heat_pellet_{i}",
                        help="Weight of heat pellet in grams"
                    )
                    electrolyte_weight = st.number_input(
                        "Electrolyte (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        step=0.01,
                        key=f"electrolyte_{i}",
                        help="Weight of electrolyte in grams"
                    )
                
                st.markdown("**Cell Configuration**")
                col3, col4 = st.columns(2)
                with col3:
                    cells_in_series = st.number_input(
                        "Cells in series:", 
                        min_value=1, 
                        max_value=100, 
                        step=1,
                        key=f"cells_series_{i}",
                        help="Number of cells connected in series"
                    )
                with col4:
                    stacks_in_parallel = st.number_input(
                        "Stacks in parallel:", 
                        min_value=1, 
                        max_value=100, 
                        step=1,
                        key=f"stacks_parallel_{i}",
                        help="Number of stacks connected in parallel"
                    )
                
                st.markdown("**Energy Input**")
                calorific_value = st.number_input(
                    "Calorific value (cal/g):", 
                    min_value=0.0, 
                    max_value=10000.0, 
                    step=1.0,
                    key=f"calorific_value_{i}",
                    help="Calorific value per gram in calories/g (will be converted to kJ)"
                )
                
                # Calculate totals correctly:
                # Total weights for all cells in parallel:
                # Total anode = anode_per_cell × cells_in_series × stacks_in_parallel
                # Total cathode = cathode_per_cell × cells_in_series × stacks_in_parallel
                # Total heat pellet = heat_pellet_per_cell × cells_in_series × stacks_in_parallel
                # Total stack weight = (sum of all per-cell weights) × cells_in_series × stacks_in_parallel
                total_anode = anode_weight * cells_in_series * stacks_in_parallel if anode_weight > 0 else 0
                total_cathode = cathode_weight * cells_in_series * stacks_in_parallel if cathode_weight > 0 else 0
                total_heat_pellet = heat_pellet_weight * cells_in_series * stacks_in_parallel if heat_pellet_weight > 0 else 0
                total_electrolyte = electrolyte_weight * cells_in_series * stacks_in_parallel if electrolyte_weight > 0 else 0
                total_stack_weight = (anode_weight + cathode_weight + heat_pellet_weight + electrolyte_weight) * cells_in_series * stacks_in_parallel if cells_in_series > 0 and stacks_in_parallel > 0 else 0
                
                # Total calorific value (kJ) = total heat pellet weight (g) × calorific value (cal/g) × 0.004184 (cal to kJ conversion)
                total_calorific = total_heat_pellet * calorific_value * 0.004184 if calorific_value > 0 and total_heat_pellet > 0 else 0
                
                st.session_state['build_metadata_extended'][i] = {
                    'anode_weight_per_cell': anode_weight,
                    'cathode_weight_per_cell': cathode_weight,
                    'heat_pellet_weight': heat_pellet_weight,
                    'electrolyte_weight': electrolyte_weight,
                    'cells_in_series': cells_in_series,
                    'stacks_in_parallel': stacks_in_parallel,
                    'calorific_value_per_gram': calorific_value,
                    'total_anode_weight': total_anode,
                    'total_cathode_weight': total_cathode,
                    'total_heat_pellet_weight': total_heat_pellet,
                    'total_electrolyte_weight': total_electrolyte,
                    'total_stack_weight': total_stack_weight,
                    'total_calorific_value': total_calorific
                }
        
        # Process data for analysis based on upload mode
        if use_multi_build_file and 'multi_build_results' in st.session_state:
            # Multi-build mode: use the parsed results
            build_results = st.session_state['multi_build_results']
            for i, (df, metadata, std_params, ext_meta) in enumerate(build_results):
                build_id = metadata.get('build_id', f'Build {i+1}')
                battery_code = metadata.get('battery_code', '')
                temperature = metadata.get('temperature', '')
                
                build_name = f"Build {build_id}"
                if battery_code:
                    build_name += f" ({battery_code})"
                if temperature:
                    build_name += f" @ {temperature}°C"
                
                # Store standard parameters from first file that has them
                if std_params and any(v is not None for v in std_params.values()):
                    if 'extracted_standard_params' not in st.session_state:
                        st.session_state['extracted_standard_params'] = std_params
                
                # Ensure unique build names
                unique_name = build_name
                counter = 1
                while unique_name in build_names:
                    unique_name = f"{build_name} ({counter})"
                    counter += 1
                
                build_names.append(unique_name)
                dataframes.append(df)
                metadata_list.append(metadata if metadata else {})
            
            # Update num_builds to match actual number of builds
            num_builds = len(build_results)
        else:
            # Individual file mode: process uploaded files for data analysis
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file:
                    df, metadata, standard_params, file_extended_metadata = load_data(uploaded_file)
                    if df is not None:
                        build_name = st.session_state.get(f'name_{i}', f"Build {i+1}")
                        
                        if metadata and any(metadata.values()):
                            if metadata.get('build_id'):
                                build_name = f"{metadata['build_id']}"
                            if metadata.get('battery_code'):
                                build_name += f" ({metadata['battery_code']})"
                            if metadata.get('temperature'):
                                build_name += f" @ {metadata['temperature']}"
                        
                        # Store standard parameters from first file that has them
                        if standard_params and any(v is not None for v in standard_params.values()):
                            if 'extracted_standard_params' not in st.session_state:
                                st.session_state['extracted_standard_params'] = standard_params
                        
                        # Ensure unique build names to prevent duplicate indices in metrics_df
                        unique_name = build_name
                        counter = 1
                        while unique_name in build_names:
                            unique_name = f"{build_name} ({counter})"
                            counter += 1
                        
                        build_names.append(unique_name)
                        dataframes.append(df)
                        metadata_list.append(metadata if metadata else {})

else:
    st.sidebar.markdown("### 📡 Real-Time Data Streaming")
    st.sidebar.info("Monitor live battery discharge data")
    
    if 'streaming_data' not in st.session_state:
        st.session_state['streaming_data'] = {}
    
    stream_build_name = st.sidebar.text_input("Build name:", value="Live Build 1", key="stream_build_name")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        stream_voltage = st.number_input("Voltage (V):", min_value=0.0, max_value=100.0, value=4.2, step=0.01, key="stream_voltage")
    with col2:
        stream_current = st.number_input("Current (A):", min_value=0.0, max_value=50.0, value=1.0, step=0.1, key="stream_current")
    
    if st.sidebar.button("➕ Add Data Point", key="add_stream_point"):
        if stream_build_name not in st.session_state['streaming_data']:
            st.session_state['streaming_data'][stream_build_name] = {
                'Time': [],
                'Voltage': [],
                'Current': []
            }
        
        current_time = len(st.session_state['streaming_data'][stream_build_name]['Time'])
        st.session_state['streaming_data'][stream_build_name]['Time'].append(current_time)
        st.session_state['streaming_data'][stream_build_name]['Voltage'].append(stream_voltage)
        st.session_state['streaming_data'][stream_build_name]['Current'].append(stream_current)
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Stream Data", key="clear_stream"):
        st.session_state['streaming_data'] = {}
        st.rerun()
    
    if st.session_state['streaming_data']:
        for name, data in st.session_state['streaming_data'].items():
            if len(data['Time']) > 0:
                df = pd.DataFrame(data)
                dataframes.append(df)
                build_names.append(name)
                metadata_list.append({})
        
        if dataframes:
            st.sidebar.success(f"📊 {len(dataframes[0])} data points collected")
    
    num_builds = len(dataframes)

if len(dataframes) == num_builds and num_builds > 0:
    if data_mode == "Upload Files":
        st.success(f"✅ All {num_builds} files loaded successfully!")
    else:
        st.success(f"✅ Streaming {num_builds} build(s) with live data!")
    
    if metadata_list and any(any(m.values()) for m in metadata_list):
        st.subheader("📋 Build Information")
        build_info_data = []
        for idx, (name, metadata) in enumerate(zip(build_names, metadata_list)):
            info = {
                'Build Name': str(name),
                'Battery Code': str(metadata.get('battery_code', 'N/A')) if metadata else 'N/A',
                'Temperature': str(metadata.get('temperature', 'N/A')) if metadata else 'N/A',
                'Build ID': str(metadata.get('build_id', 'N/A')) if metadata else 'N/A'
            }
            build_info_data.append(info)
        
        build_info_df = pd.DataFrame(build_info_data)
        # Ensure all columns are string type to avoid pyarrow serialization errors
        for col in build_info_df.columns:
            build_info_df[col] = build_info_df[col].astype(str)
        st.dataframe(build_info_df, use_container_width=True, hide_index=True)
        st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Discharge Curves", "📈 Multi-Parameter Analysis", "📋 Metrics Comparison", "🔬 Advanced Analytics", "📄 Data Preview", "🌡️ Temperature Comparison"])
    
    with tab1:
        st.header("Voltage Discharge Curves")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        all_metrics = []
        
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            
            if voltage_col:
                df_plot = downsample_for_plotting(df)
                
                if len(df) > 10000:
                    st.info(f"ℹ️ Dataset '{name}' has {len(df):,} rows. Showing {len(df_plot):,} points for visualization (all data used for calculations).")
                
                x_axis = None
                x_label = ""
                
                if capacity_col and capacity_col in df_plot.columns:
                    x_axis = df_plot[capacity_col]
                    x_label = f"Capacity ({capacity_col})"
                elif time_col and time_col in df_plot.columns:
                    x_axis = df_plot[time_col]
                    x_label = f"Time ({time_col})"
                else:
                    x_axis = df_plot.index
                    x_label = "Data Point"
                
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=df_plot[voltage_col],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'<b>{name}</b><br>{x_label}: %{{x}}<br>Voltage: %{{y:.3f}} V<extra></extra>'
                ))
                
                extended_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
                metrics = calculate_metrics(
                    df, time_col, voltage_col, current_col, 
                    min_activation_voltage=min_activation_voltage,
                    extended_metadata=extended_meta,
                    target_duration_sec=target_duration_sec
                )
                metrics['Build'] = name
                all_metrics.append(metrics)
        
        fig.update_layout(
            xaxis_title=x_label if 'x_label' in locals() else "Time/Capacity",
            yaxis_title="Voltage (V)",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics based on number of builds
        if num_builds <= 4:
            # For small number of builds, use columns
            cols = st.columns(num_builds)
            for idx, metrics in enumerate(all_metrics):
                with cols[idx]:
                    st.markdown(f"### {metrics['Build']}")
                    if 'Max On-Load Voltage (V)' in metrics and metrics['Max On-Load Voltage (V)'] is not None:
                        st.metric("Max On-Load V", f"{metrics['Max On-Load Voltage (V)']:.3f} V")
                    if 'Max On-Load Time (s)' in metrics and metrics['Max On-Load Time (s)'] is not None:
                        st.metric("Max On-Load Time", f"{metrics['Max On-Load Time (s)']:.2f} s")
                    if 'Max Open Circuit Voltage (V)' in metrics and metrics['Max Open Circuit Voltage (V)'] is not None:
                        st.metric("Max OC V", f"{metrics['Max Open Circuit Voltage (V)']:.3f} V")
                    if 'Max OC Voltage Time (s)' in metrics and metrics['Max OC Voltage Time (s)'] is not None:
                        st.metric("Max OC Time", f"{metrics['Max OC Voltage Time (s)']:.2f} s")
                    if 'Weighted Average Voltage (V)' in metrics and metrics['Weighted Average Voltage (V)'] is not None:
                        st.metric("Weighted Avg V", f"{metrics['Weighted Average Voltage (V)']:.3f} V")
                    if 'Activation Time (Sec)' in metrics and metrics['Activation Time (Sec)'] is not None:
                        st.metric("Activation Time", f"{metrics['Activation Time (Sec)']:.2f} s")
                    if 'Duration (Sec)' in metrics and metrics['Duration (Sec)'] is not None:
                        st.metric("Duration", f"{metrics['Duration (Sec)']:.2f} s")
                    if 'Voltage at Targeted Duration (V)' in metrics and metrics['Voltage at Targeted Duration (V)'] is not None:
                        st.metric("V at Target Duration", f"{metrics['Voltage at Targeted Duration (V)']:.3f} V")
        else:
            # For many builds, use a table
            st.subheader("📊 Key Performance Metrics")
            metrics_summary = []
            for metrics in all_metrics:
                summary = {
                    'Build': metrics['Build'],
                    'Max On-Load V': metrics.get('Max On-Load Voltage (V)', 'N/A'),
                    'Max On-Load Time (s)': metrics.get('Max On-Load Time (s)', 'N/A'),
                    'Max OC V': metrics.get('Max Open Circuit Voltage (V)', 'N/A'),
                    'Max OC Time (s)': metrics.get('Max OC Voltage Time (s)', 'N/A'),
                    'Weighted Avg V': metrics.get('Weighted Average Voltage (V)', 'N/A'),
                    'Activation Time (s)': metrics.get('Activation Time (Sec)', 'N/A'),
                    'Duration (s)': metrics.get('Duration (Sec)', 'N/A')
                }
                if 'Voltage at Targeted Duration (V)' in metrics and metrics['Voltage at Targeted Duration (V)'] is not None:
                    summary['V at Target Duration'] = metrics.get('Voltage at Targeted Duration (V)', 'N/A')
                metrics_summary.append(summary)
            
            summary_df = pd.DataFrame(metrics_summary)
            
            # Format the dataframe
            format_dict = {}
            for col in summary_df.columns:
                if col != 'Build':
                    format_dict[col] = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Multi-Parameter Time-Series Analysis")
        
        has_time = any([detect_columns(df)[0] for df in dataframes])
        
        if has_time:
            for idx, (df, name) in enumerate(zip(dataframes, build_names)):
                time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                
                if time_col and time_col in df.columns:
                    df_plot = downsample_for_plotting(df)
                    
                    params = []
                    if voltage_col and voltage_col in df_plot.columns:
                        params.append(('Voltage (V)', voltage_col))
                    if current_col and current_col in df_plot.columns:
                        params.append(('Current (A)', current_col))
                    
                    if len(params) > 1:
                        fig = make_subplots(
                            rows=len(params), cols=1,
                            subplot_titles=[p[0] for p in params],
                            vertical_spacing=0.1,
                            shared_xaxes=True
                        )
                        
                        for i, (param_name, param_col) in enumerate(params):
                            fig.add_trace(
                                go.Scatter(
                                    x=df_plot[time_col],
                                    y=df_plot[param_col],
                                    mode='lines',
                                    name=param_name,
                                    line=dict(color=colors[idx % len(colors)], width=2)
                                ),
                                row=i+1, col=1
                            )
                            
                            fig.update_yaxes(title_text=param_name, row=i+1, col=1)
                        
                        fig.update_xaxes(title_text=f"Time ({time_col})", row=len(params), col=1)
                        fig.update_layout(
                            height=400 * len(params),
                            showlegend=False,
                            title_text=f"{name} - Time Series Analysis",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if voltage_col and current_col and voltage_col in df_plot.columns and current_col in df_plot.columns:
                            power = df_plot[voltage_col] * df_plot[current_col].abs()
                            
                            fig_power = go.Figure()
                            fig_power.add_trace(go.Scatter(
                                x=df_plot[time_col],
                                y=power,
                                mode='lines',
                                name='Power',
                                line=dict(color=colors[idx % len(colors)], width=2),
                                fill='tozeroy'
                            ))
                            
                            fig_power.update_layout(
                                title=f"{name} - Power Over Time",
                                xaxis_title=f"Time ({time_col})",
                                yaxis_title="Power (W)",
                                height=300,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig_power, use_container_width=True)
                    else:
                        st.info(f"{name}: Not enough parameters for multi-parameter analysis")
        else:
            st.warning("Time column not detected in the datasets. Please ensure your data includes a time column.")
    
    with tab3:
        st.header("Metrics Comparison Table")
        
        all_build_metrics = []
        
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            extended_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
            metrics = calculate_metrics(
                df, time_col, voltage_col, current_col,
                min_activation_voltage=min_activation_voltage,
                extended_metadata=extended_meta,
                target_duration_sec=target_duration_sec
            )
            metrics['Build'] = name
            all_build_metrics.append(metrics)
        
        if all_build_metrics:
            metrics_df = pd.DataFrame(all_build_metrics)
            metrics_df = metrics_df.set_index('Build')
            
            # Export buttons row
            export_col1, export_col2, export_col3, export_col4 = st.columns(4)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with export_col1:
                excel_data = export_to_excel(dataframes, build_names, metrics_df)
                st.download_button(
                    label="📥 Excel",
                    data=excel_data,
                    file_name=f"battery_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with export_col2:
                csv_data = export_to_csv(dataframes, build_names, metrics_df)
                if csv_data:
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name=f"battery_metrics_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            with export_col3:
                # Generate detailed text report
                try:
                    report_text = generate_detailed_report(
                        metrics_df, build_names, metadata_list,
                        min_activation_voltage,
                        use_standards, std_max_onload_voltage,
                        std_max_oc_voltage, std_activation_time, std_duration,
                        std_activation_time_ms, std_duration_sec
                    )
                    st.download_button(
                        label="📄 Report",
                        data=report_text.encode('utf-8'),
                        file_name=f"battery_report_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Report generation error: {str(e)}")
            with export_col4:
                try:
                    # Prepare analytics and extended metadata for PDF
                    analytics_list_for_pdf = []
                    extended_metadata_list_for_pdf = []
                    
                    for idx, (df, name) in enumerate(zip(dataframes, build_names)):
                        time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                        
                        # Get analytics for this build
                        analytics = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
                        analytics_list_for_pdf.append(analytics)
                        
                        # Get extended metadata for this build
                        ext_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
                        extended_metadata_list_for_pdf.append(ext_meta)
                    
                    # Calculate correlations if we have enough data
                    correlations_for_pdf = None
                    duration_correlations_for_pdf = None
                    if len(all_build_metrics) >= 2:
                        correlations_for_pdf = calculate_correlation_analysis(all_build_metrics, analytics_list_for_pdf)
                        duration_correlations_for_pdf = calculate_duration_correlations(all_build_metrics, extended_metadata_list_for_pdf)
                    
                    # Generate PDF report
                    pdf_data = generate_pdf_report(
                        metrics_df, build_names, metadata_list,
                        min_activation_voltage,
                        use_standards, std_max_onload_voltage,
                        std_max_oc_voltage, std_activation_time, std_duration,
                        std_activation_time_ms, std_duration_sec,
                        extended_metadata_list_for_pdf,
                        analytics_list_for_pdf,
                        correlations_for_pdf,
                        duration_correlations_for_pdf
                    )
                    st.download_button(
                        label="📕 PDF",
                        data=pdf_data,
                        file_name=f"battery_report_{timestamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {str(e)}")
            
            # Save form row (full width for better layout)
            st.markdown("---")
            if DATABASE_URL and Session:
                # Save/Update comparison form
                is_editing = 'editing_comparison_id' in st.session_state
                default_name = st.session_state.get('editing_comparison_name', f"Comparison_{timestamp}")
                
                with st.form(key='save_comparison_form'):
                    comparison_name = st.text_input(
                        "Comparison name:",
                        value=default_name,
                        help="Enter a name for this comparison"
                    )
                    
                    # Password protection options
                    protect_with_password = st.checkbox(
                        "🔐 Password protect",
                        value=False,
                        help="Add password protection to this comparison"
                    )
                    
                    save_password = None
                    save_password_confirm = None
                    existing_password_for_update = None
                    
                    if protect_with_password:
                        st.warning("⚠️ **Warning**: If you forget this password, you will permanently lose access to this comparison. There is no password recovery.")
                        save_password = st.text_input(
                            "Password:",
                            type="password",
                            help="Enter a password to protect this comparison"
                        )
                        save_password_confirm = st.text_input(
                            "Confirm password:",
                            type="password",
                            help="Re-enter the password to confirm"
                        )
                    
                    # For editing protected comparisons, require existing password
                    if is_editing and st.session_state.get('editing_comparison_protected', False):
                        st.info("📝 This is a protected comparison. Enter the existing password to update it.")
                        existing_password_for_update = st.text_input(
                            "Existing password:",
                            type="password",
                            help="Enter the current password for this comparison"
                        )
                        if protect_with_password:
                            st.info("💡 Tip: Enter a new password above to change it, or leave blank to keep the existing password.")
                    
                    # Show appropriate save buttons
                    col_save1, col_save2 = st.columns(2)
                    
                    with col_save1:
                        if is_editing:
                            update_clicked = st.form_submit_button("🔄 Update Existing", use_container_width=True)
                        else:
                            update_clicked = False
                    
                    with col_save2:
                        save_new_clicked = st.form_submit_button("💾 Save As New", use_container_width=True)
                    
                    # Handle form submission
                    if update_clicked or save_new_clicked:
                        # Validate inputs
                        if not comparison_name:
                            st.error("Please enter a comparison name")
                        elif protect_with_password and (not save_password or not save_password_confirm):
                            st.error("Please enter and confirm the password")
                        elif protect_with_password and save_password != save_password_confirm:
                            st.error("Passwords do not match")
                        else:
                            current_battery_type = st.session_state.get('battery_type', 'General')
                            current_extended_meta = st.session_state.get('build_metadata_extended', {})
                            
                            # Collect current standard params (from loaded or current state)
                            current_standard_params = {
                                'min_activation_voltage': min_activation_voltage,
                                'std_max_oc_voltage': std_max_oc_voltage,
                                'std_activation_time_ms': std_activation_time_ms if use_standards else None,
                                'std_duration_sec': std_duration_sec if use_standards else None
                            }
                            
                            if update_clicked:
                                # Update existing comparison
                                comparison_id = st.session_state.get('editing_comparison_id')
                                success, msg = update_comparison_in_db(
                                    comparison_id,
                                    comparison_name,
                                    dataframes,
                                    build_names,
                                    metrics_df,
                                    battery_type=current_battery_type,
                                    extended_metadata=current_extended_meta,
                                    existing_password=existing_password_for_update,
                                    new_password=save_password if protect_with_password else None,
                                    standard_params=current_standard_params
                                )
                            else:
                                # Save as new comparison
                                success, msg = save_comparison_to_db(
                                    comparison_name,
                                    dataframes,
                                    build_names,
                                    metrics_df,
                                    battery_type=current_battery_type,
                                    extended_metadata=current_extended_meta,
                                    password=save_password if protect_with_password else None,
                                    standard_params=current_standard_params
                                )
                            
                            if success:
                                st.success(msg)
                                # Clear editing context after successful save
                                if 'editing_comparison_id' in st.session_state:
                                    del st.session_state['editing_comparison_id']
                                if 'editing_comparison_name' in st.session_state:
                                    del st.session_state['editing_comparison_name']
                                if 'editing_comparison_protected' in st.session_state:
                                    del st.session_state['editing_comparison_protected']
                            else:
                                st.error(msg)
            
            # Format metrics dataframe, handling None/NaN values gracefully
            def format_value(val):
                if pd.isna(val) or val is None:
                    return "N/A"
                elif isinstance(val, (int, float)):
                    return f"{val:.4f}"
                else:
                    return str(val)
            
            # Configure column display for better header wrapping
            column_config = {}
            for col in metrics_df.columns:
                column_config[col] = st.column_config.TextColumn(
                    col,
                    width="medium",
                    help=None
                )
            
            st.dataframe(
                metrics_df.map(format_value), 
                use_container_width=True,
                column_config=column_config,
                height=400
            )
            
            st.subheader("Statistical Summary")
            
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                stats_data = []
                for col in numeric_cols:
                    stats_data.append({
                        'Metric': col,
                        'Mean': metrics_df[col].mean(),
                        'Std Dev': metrics_df[col].std(),
                        'Min': metrics_df[col].min(),
                        'Max': metrics_df[col].max(),
                        'Range': metrics_df[col].max() - metrics_df[col].min()
                    })
                
                stats_df = pd.DataFrame(stats_data)
                format_dict = {col: '{:.4f}' for col in stats_df.columns if col != 'Metric'}
                st.dataframe(stats_df.style.format(format_dict), use_container_width=True)
                
                if len(metrics_df) == 2:
                    st.subheader("Build-to-Build Comparison")
                    comparison_data = []
                    build1, build2 = metrics_df.index[0], metrics_df.index[1]
                    
                    for col in numeric_cols:
                        val1 = metrics_df.loc[build1, col]
                        val2 = metrics_df.loc[build2, col]
                        diff = val2 - val1
                        
                        # Handle both scalar and Series cases
                        if isinstance(val1, pd.Series):
                            val1 = val1.iloc[0] if len(val1) > 0 else 0
                        if isinstance(val2, pd.Series):
                            val2 = val2.iloc[0] if len(val2) > 0 else 0
                        if isinstance(diff, pd.Series):
                            diff = diff.iloc[0] if len(diff) > 0 else 0
                        
                        pct_change = (diff / val1 * 100) if val1 != 0 else 0
                        
                        comparison_data.append({
                            'Metric': col,
                            f'{build1}': val1,
                            f'{build2}': val2,
                            'Difference': diff,
                            '% Change': pct_change
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.style.format({
                        f'{build1}': '{:.4f}',
                        f'{build2}': '{:.4f}',
                        'Difference': '{:.4f}',
                        '% Change': '{:.2f}%'
                    }), use_container_width=True)
                
                # Standard Performance Comparison
                if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time_ms, std_duration_sec]):
                    st.subheader("⭐ Performance vs. Standard Benchmarks")
                    st.markdown("Compare each build against specified standard performance levels")
                    
                    standard_comparison_data = []
                    
                    for build_name in metrics_df.index:
                        build_data = {'Build': build_name}
                        
                        # Max On-Load Voltage comparison
                        if std_max_onload_voltage and 'Max On-Load Voltage (V)' in metrics_df.columns:
                            actual = safe_scalar(metrics_df.loc[build_name, 'Max On-Load Voltage (V)'])
                            if pd.notna(actual):
                                diff = actual - std_max_onload_voltage
                                build_data['Max On-Load V (Actual)'] = actual
                                build_data['Max On-Load V (Std)'] = std_max_onload_voltage
                                build_data['Max On-Load V (Diff)'] = diff
                        
                        # Max Open Circuit Voltage comparison
                        if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
                            actual = safe_scalar(metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)'])
                            if pd.notna(actual):
                                diff = actual - std_max_oc_voltage
                                build_data['Max OC V (Actual)'] = actual
                                build_data['Max OC V (Std)'] = std_max_oc_voltage
                                build_data['Max OC V (Diff)'] = diff
                        
                        # Activation Time comparison (in seconds)
                        if std_activation_time_ms and 'Activation Time (Sec)' in metrics_df.columns:
                            actual_sec = safe_scalar(metrics_df.loc[build_name, 'Activation Time (Sec)'])
                            if pd.notna(actual_sec):
                                std_sec = std_activation_time_ms / 1000.0  # Convert ms to seconds
                                diff = actual_sec - std_sec
                                build_data['Activation Time (Actual s)'] = actual_sec
                                build_data['Activation Time (Std s)'] = std_sec
                                build_data['Activation Time (Diff s)'] = diff
                        
                        # Duration comparison (in seconds)
                        if std_duration_sec and 'Duration (Sec)' in metrics_df.columns:
                            actual_sec = safe_scalar(metrics_df.loc[build_name, 'Duration (Sec)'])
                            if pd.notna(actual_sec):
                                diff = actual_sec - std_duration_sec
                                build_data['Duration (Actual s)'] = actual_sec
                                build_data['Duration (Target s)'] = std_duration_sec
                                build_data['Duration (Diff s)'] = diff
                        
                        standard_comparison_data.append(build_data)
                    
                    if standard_comparison_data:
                        std_comp_df = pd.DataFrame(standard_comparison_data)
                        
                        # Format numeric columns
                        format_dict = {}
                        for col in std_comp_df.columns:
                            if col not in ['Build']:
                                format_dict[col] = '{:.4f}'
                        
                        st.dataframe(std_comp_df.style.format(format_dict), use_container_width=True)
    
    with tab4:
        st.header("Advanced Analytics & Anomaly Detection")
        
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            
            st.subheader(f"📊 {name}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Advanced Metrics")
                analytics = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
                
                if analytics:
                    analytics_df = pd.DataFrame([analytics]).T
                    analytics_df.columns = ['Value']
                    # Convert Value column to string to avoid pyarrow serialization errors with mixed types
                    analytics_df['Value'] = analytics_df['Value'].astype(str)
                    
                    def format_value(val):
                        # Format numeric string representations
                        try:
                            num_val = float(val)
                            return f"{num_val:.4f}"
                        except (ValueError, TypeError):
                            return str(val)
                    
                    st.dataframe(analytics_df.style.format(format_value), use_container_width=True)
                else:
                    st.warning(f"⚠️ Unable to calculate discharge curve metrics for {name}")
                    st.info("Possible reasons: insufficient data points, invalid time/voltage columns, or data quality issues.")
            
            with col2:
                st.markdown("#### Key Insights")
                if analytics:
                    if 'Average Degradation Rate (mV/min)' in analytics:
                        st.metric("Degradation Rate", f"{analytics['Average Degradation Rate (mV/min)']:.2f} mV/min")
                    if 'Energy Efficiency (%)' in analytics:
                        st.metric("Energy Efficiency", f"{analytics['Energy Efficiency (%)']:.1f}%")
                    if 'Voltage Retention (%)' in analytics:
                        st.metric("Voltage Retention", f"{analytics['Voltage Retention (%)']:.1f}%")
            
            # Anomaly detection removed per user request
            
            st.markdown("---")
        
        # Correlation Analysis Section (only if we have multiple builds with extended metadata)
        if num_builds >= 2:
            st.markdown("---")
            st.header("🔗 Correlation Analysis")
            st.markdown("*Analyzing relationships between performance metrics and discharge curve characteristics*")
            
            # Collect all metrics and analytics for correlation analysis
            all_metrics_for_corr = []
            all_analytics_for_corr = []
            
            for idx, (df, name) in enumerate(zip(dataframes, build_names)):
                time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                extended_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
                
                metrics = calculate_metrics(
                    df, time_col, voltage_col, current_col,
                    min_activation_voltage=min_activation_voltage,
                    extended_metadata=extended_meta,
                    target_duration_sec=target_duration_sec
                )
                analytics = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
                
                all_metrics_for_corr.append(metrics)
                all_analytics_for_corr.append(analytics)
            
            # Calculate correlations
            correlations = calculate_correlation_analysis(all_metrics_for_corr, all_analytics_for_corr)
            
            # Calculate duration correlations
            all_extended_metadata = []
            for idx in range(num_builds):
                extended_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
                all_extended_metadata.append(extended_meta)
            
            duration_correlations = calculate_duration_correlations(all_metrics_for_corr, all_extended_metadata)
            
            if correlations:
                st.success(f"✅ Found {len(correlations)} performance-discharge correlation relationships")
                
                for corr_name, corr_data in correlations.items():
                    with st.expander(f"📊 {corr_name}", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Display correlation metrics
                            st.metric("Correlation Coefficient", f"{corr_data['correlation']:.3f}")
                            st.metric("Strength", corr_data['strength'])
                            st.metric("Direction", corr_data['direction'])
                            st.metric("Significance", corr_data['significance'])
                        
                        with col2:
                            # Display interpretation
                            st.markdown("**Interpretation:**")
                            if 'interpretation' in corr_data:
                                st.info(corr_data['interpretation'])
                            
                            abs_corr = abs(corr_data['correlation'])
                            if abs_corr > 0.7:
                                st.markdown("- **Strong relationship**: Changes in one variable are highly predictive of changes in the other")
                            elif abs_corr > 0.4:
                                st.markdown("- **Moderate relationship**: Some predictive value between the variables")
                            else:
                                st.markdown("- **Weak relationship**: Limited predictive value")
                            
                            if corr_data['p_value'] < 0.05:
                                st.markdown("- ✅ **Statistically significant** (p < 0.05): This relationship is unlikely to be due to random chance")
                            else:
                                st.markdown("- ⚠️ **Not statistically significant**: This relationship may be due to random chance")
            else:
                st.info("ℹ️ Performance-discharge correlation analysis requires at least 2 builds with extended metadata.")
            
            # Duration Correlations Section
            if duration_correlations:
                st.markdown("---")
                st.subheader("⏱️ Duration vs Battery Construction Correlations")
                st.markdown("*Analyzing how battery construction parameters affect discharge duration*")
                
                # Display values table first
                if 'values_table' in duration_correlations:
                    st.markdown("**📋 Actual Values Used in Analysis:**")
                    values_df = pd.DataFrame(duration_correlations['values_table'])
                    # Convert all columns to strings to avoid PyArrow mixed-type errors
                    for col in values_df.columns:
                        values_df[col] = values_df[col].astype(str)
                    st.dataframe(values_df, use_container_width=True)
                    st.markdown("")
                
                # Display correlation results (excluding values_table key)
                corr_count = len([k for k in duration_correlations.keys() if k != 'values_table'])
                if corr_count > 0:
                    st.success(f"✅ Found {corr_count} duration correlation relationships")
                    
                    for corr_name, corr_data in duration_correlations.items():
                        if corr_name == 'values_table':
                            continue
                        
                        with st.expander(f"📊 {corr_name}", expanded=True):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric("Correlation Coefficient", f"{corr_data['correlation']:.3f}")
                                st.metric("Strength", corr_data['strength'])
                                st.metric("Direction", corr_data['direction'])
                                st.metric("Significance", corr_data['significance'])
                            
                            with col2:
                                st.markdown("**Interpretation:**")
                                abs_corr = abs(corr_data['correlation'])
                                if abs_corr > 0.7:
                                    st.markdown("- **Strong relationship**: Construction parameter strongly influences discharge duration")
                                elif abs_corr > 0.4:
                                    st.markdown("- **Moderate relationship**: Construction parameter moderately affects duration")
                                else:
                                    st.markdown("- **Weak relationship**: Limited influence on duration")
                                
                                if corr_data['p_value'] < 0.05:
                                    st.markdown("- ✅ **Statistically significant** (p < 0.05)")
                                else:
                                    st.markdown("- ⚠️ **Not statistically significant**")
            else:
                st.info("ℹ️ Duration correlation analysis requires at least 2 builds with extended metadata (weights, calorific value) to be entered.")
                st.markdown("**To enable this analysis:**")
                st.markdown("1. Enter extended build metadata (anode/cathode weights, calorific value) for at least 2 builds")
                st.markdown("2. The analysis will calculate correlations between:")
                st.markdown("   - Duration vs Total Anode Weight")
                st.markdown("   - Duration vs Total Cathode Weight")
                st.markdown("   - Duration vs Total Calorific Value")
    
    with tab5:
        st.header("Data Preview")
        
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            with st.expander(f"📄 {name} - Data Preview ({len(df)} rows)"):
                time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                
                # Display extracted metadata if available
                build_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
                if build_meta and any(v and v != 0 for v in build_meta.values()):
                    st.markdown("**📋 Extracted Metadata:**")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        if build_meta.get('anode_weight_per_cell'):
                            st.metric("Anode (per cell)", f"{build_meta['anode_weight_per_cell']:.2f} g")
                        if build_meta.get('cathode_weight_per_cell'):
                            st.metric("Cathode (per cell)", f"{build_meta['cathode_weight_per_cell']:.2f} g")
                    with meta_cols[1]:
                        if build_meta.get('cells_in_series'):
                            st.metric("Cells in Series", f"{build_meta['cells_in_series']}")
                        if build_meta.get('stacks_in_parallel'):
                            st.metric("Stacks in Parallel", f"{build_meta['stacks_in_parallel']}")
                    with meta_cols[2]:
                        if build_meta.get('calorific_value_per_gram'):
                            st.metric("Calorific Value", f"{build_meta['calorific_value_per_gram']:.0f} cal/g")
                        if build_meta.get('total_stack_weight'):
                            st.metric("Total Stack Weight", f"{build_meta['total_stack_weight']:.2f} g")
                    st.markdown("---")
                
                st.markdown("**Detected Columns:**")
                col_info = f"- Time: `{time_col if time_col else 'Not detected'}`\n"
                col_info += f"- Voltage: `{voltage_col if voltage_col else 'Not detected'}`\n"
                col_info += f"- Current: `{current_col if current_col else 'Not detected'}`\n"
                col_info += f"- Capacity: `{capacity_col if capacity_col else 'Not detected'}`"
                st.markdown(col_info)
                
                st.dataframe(df.head(20), use_container_width=True)
                
                st.markdown(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    with tab6:
        st.header("Temperature-Based Performance Comparison")
        
        temp_data = []
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            metadata = metadata_list[idx] if idx < len(metadata_list) else {}
            
            temp = None
            if metadata and metadata.get('temperature'):
                temp_str = str(metadata['temperature'])
                import re
                temp_match = re.search(r'(-?\d+(?:\.\d+)?)', temp_str)
                if temp_match:
                    temp = float(temp_match.group(1))
            
            if temp is None:
                temp = extract_temperature_from_name(name)
            
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            extended_meta = st.session_state.get('build_metadata_extended', {}).get(idx, {})
            metrics = calculate_metrics(
                df, time_col, voltage_col, current_col,
                min_activation_voltage=min_activation_voltage,
                extended_metadata=extended_meta,
                target_duration_sec=target_duration_sec
            )
            advanced = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
            
            battery_code = metadata.get('battery_code', '') if metadata else ''
            build_id = metadata.get('build_id', '') if metadata else ''
            
            temp_data.append({
                'Build Name': name,
                'Battery Code': battery_code if battery_code else 'N/A',
                'Build ID': build_id if build_id else 'N/A',
                'Temperature (°C)': temp if temp is not None else 'Not specified',
                'Duration (s)': metrics.get('Duration (Sec)', 0),
                'Weighted Avg V': metrics.get('Weighted Average Voltage (V)', 0),
                'Avg Degradation (mV/min)': advanced.get('Average Degradation Rate (mV/min)', 0),
                'Voltage Retention (%)': advanced.get('Voltage Retention (%)', 0),
            })
        
        if temp_data:
            temp_df = pd.DataFrame(temp_data)
            
            has_temps = any(isinstance(t['Temperature (°C)'], (int, float)) for t in temp_data)
            
            if has_temps:
                st.success("🌡️ Temperature information detected!")
                st.info("**Tip:** You can include temperature in file metadata or build names (e.g., '25C', '-20C', '0C') for automatic temperature detection.")
                
                numeric_temp_df = temp_df[temp_df['Temperature (°C)'].apply(lambda x: isinstance(x, (int, float)))].copy()
                numeric_temp_df = numeric_temp_df.sort_values('Temperature (°C)')
                
                st.subheader("Performance vs Temperature")
                st.dataframe(numeric_temp_df, use_container_width=True)
                
                if len(numeric_temp_df) >= 2:
                    st.subheader("Temperature Impact Visualization")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=numeric_temp_df['Temperature (°C)'],
                            y=numeric_temp_df['Duration (s)'],
                            mode='lines+markers',
                            name='Discharge Duration',
                            marker=dict(size=10)
                        ))
                        fig1.update_layout(
                            title='Discharge Duration vs Temperature',
                            xaxis_title='Temperature (°C)',
                            yaxis_title='Discharge Duration (s)',
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=numeric_temp_df['Temperature (°C)'],
                            y=numeric_temp_df['Weighted Avg V'],
                            mode='lines+markers',
                            name='Weighted Avg Voltage',
                            marker=dict(size=10, color='orange')
                        ))
                        fig2.update_layout(
                            title='Weighted Average Voltage vs Temperature',
                            xaxis_title='Temperature (°C)',
                            yaxis_title='Weighted Average Voltage (V)',
                            height=400
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.subheader("Voltage Retention vs Temperature")
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=numeric_temp_df['Temperature (°C)'],
                        y=numeric_temp_df['Voltage Retention (%)'],
                        mode='lines+markers',
                        name='Voltage Retention',
                        marker=dict(size=10, color='green')
                    ))
                    fig3.update_layout(
                        title='Voltage Retention vs Temperature',
                        xaxis_title='Temperature (°C)',
                        yaxis_title='Voltage Retention (%)',
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    st.subheader("Temperature Performance Summary")
                    best_temp_idx = numeric_temp_df['Duration (s)'].idxmax()
                    worst_temp_idx = numeric_temp_df['Duration (s)'].idxmin()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Best Performance Temperature",
                            f"{numeric_temp_df.loc[best_temp_idx, 'Temperature (°C)']}°C",
                            f"{numeric_temp_df.loc[best_temp_idx, 'Duration (s)']:.2f} s"
                        )
                    with col2:
                        st.metric(
                            "Worst Performance Temperature",
                            f"{numeric_temp_df.loc[worst_temp_idx, 'Temperature (°C)']}°C",
                            f"{numeric_temp_df.loc[worst_temp_idx, 'Duration (s)']:.2f} s"
                        )
                    with col3:
                        duration_range = numeric_temp_df['Duration (s)'].max() - numeric_temp_df['Duration (s)'].min()
                        avg_duration = numeric_temp_df['Duration (s)'].mean()
                        pct_variation = (duration_range / avg_duration * 100) if avg_duration > 0 else 0
                        st.metric(
                            "Duration Variation",
                            f"{pct_variation:.1f}%",
                            f"Range: {duration_range:.2f} s"
                        )
            else:
                st.warning("⚠️ No temperature information detected in build names.")
                st.info("**Tip:** Include temperature in your build names (e.g., '25C', '-20C', '0C') to enable temperature-based comparison.")
                st.dataframe(temp_df, use_container_width=True)

else:
    if data_mode == "Upload Files":
        st.info(f"👆 Please upload discharge data files using the sidebar to begin analysis.")
    else:
        st.info(f"👆 Use the sidebar to add real-time data points for live monitoring.")
    
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Two Input Modes
        
        **1. Upload Files Mode**
        - Upload 2-3 CSV or Excel files with battery discharge data
        - Files are automatically analyzed and compared
        - Save comparisons to database for later review
        
        **2. Real-Time Streaming Mode**
        - Manually enter voltage and current readings as they occur
        - Build live discharge curves as data accumulates
        - Perfect for monitoring ongoing battery tests
        
        ### Data Format Requirements
        
        Your discharge data files should contain the following columns (column names will be auto-detected):
        
        - **Time**: Time stamps for each measurement
          - Numeric values: assumed to be in **minutes**
          - Datetime strings: automatically converted (e.g., "2024-01-01 10:00:00")
          - Timedelta strings: automatically converted (e.g., "00:15:30")
        - **Voltage**: Battery voltage measurements (e.g., "Voltage", "Voltage(V)", "volt")
        - **Current**: Current measurements - optional (e.g., "Current", "Current(A)", "Amp")
        - **Capacity**: Capacity measurements - optional (e.g., "Capacity", "Cap", "Capacity(Ah)")
        
        **Note**: For numeric time columns, ensure values are in **minutes** for accurate calculations.
        
        ### Features
        
        1. **Discharge Curves**: Compare voltage curves side-by-side
        2. **Multi-Parameter Analysis**: View voltage, current, and power over time
        3. **Metrics Comparison**: See calculated metrics and statistics with export options
        4. **Advanced Analytics**: Degradation rates, efficiency metrics, and cycle life estimation
        5. **Anomaly Detection**: Automatic detection of unusual discharge patterns
        6. **Data Preview**: Inspect your raw data
        
        ### Supported File Formats
        
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Battery Discharge Analysis Tool v1.0 - Compare discharge performance across different builds")
