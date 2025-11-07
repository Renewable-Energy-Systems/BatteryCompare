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
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import json
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
else:
    engine = None
    Session = None

st.set_page_config(page_title="Battery Discharge Analysis", layout="wide")

st.title("🔋 Battery Discharge Data Analysis")
st.markdown("Compare discharge curves and performance metrics across different builds")

def extract_metadata_from_file(uploaded_file):
    """
    Extract metadata from the top rows of the file.
    Supports two formats:
    1. Metadata in columns 0-1 (original format)
    2. Metadata in columns 4-5 (new format with empty leading columns)
    
    Returns: (metadata_dict, data_start_row)
    """
    metadata = {
        'battery_code': None,
        'temperature': None,
        'build_id': None
    }
    
    try:
        if uploaded_file.name.endswith('.csv'):
            temp_df = pd.read_csv(uploaded_file, nrows=10, header=None)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            temp_df = pd.read_excel(uploaded_file, nrows=10, header=None)
        else:
            return metadata, 0
        
        uploaded_file.seek(0)
        
        data_start_row = 0
        found_header = False
        
        for idx, row in temp_df.iterrows():
            # Check all columns for metadata and headers
            row_strings = [str(val).lower() if pd.notna(val) else "" for val in row]
            
            # Check if any column contains header keywords
            header_found_in_row = any('time' in s or 'voltage' in s or 'current' in s for s in row_strings)
            
            if header_found_in_row:
                # This is the header row
                data_start_row = idx
                found_header = True
                break
            
            # Check for metadata in any column pair
            for col_idx in range(len(row) - 1):
                label = str(row[col_idx]).lower() if pd.notna(row[col_idx]) else ""
                value = row[col_idx + 1]
                
                if 'battery' in label and 'code' in label and pd.notna(value):
                    metadata['battery_code'] = str(value)
                elif ('temperature' in label or 'temp' in label) and pd.notna(value):
                    metadata['temperature'] = str(value)
                elif 'build' in label and ('number' in label or 'id' in label) and pd.notna(value):
                    metadata['build_id'] = str(value)
        
        if not found_header and data_start_row == 0:
            data_start_row = 0
        
        return metadata, data_start_row
    except:
        return metadata, 0

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file with metadata extraction"""
    try:
        metadata, skip_rows = extract_metadata_from_file(uploaded_file)
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skip_rows)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, skiprows=skip_rows)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None, None
        
        # Drop columns that are completely empty (all NaN)
        df = df.dropna(axis=1, how='all')
        
        # Drop rows that are completely empty
        df = df.dropna(axis=0, how='all')
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        return df, metadata
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

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

def calculate_metrics(df, time_col, voltage_col, current_col=None, min_activation_voltage=1.0, extended_metadata=None):
    """Calculate key battery metrics including new performance metrics
    
    Args:
        df: DataFrame with battery discharge data
        time_col: Name of time column
        voltage_col: Name of voltage column
        current_col: Name of current column
        min_activation_voltage: Minimum voltage threshold for activation
        extended_metadata: Dict with extended build metadata (weights, calorific value, etc.)
    """
    metrics = {}
    
    if voltage_col and voltage_col in df.columns:
        metrics['Max Voltage (V)'] = float(df[voltage_col].max())
        metrics['Min Voltage (V)'] = float(df[voltage_col].min())
        metrics['Average Voltage (V)'] = float(df[voltage_col].mean())
        metrics['Voltage Range (V)'] = metrics['Max Voltage (V)'] - metrics['Min Voltage (V)']
        
        # Calculate max open circuit voltage and max on-load voltage
        if current_col and current_col in df.columns:
            # Open circuit: when current is 0 or very close to 0 (< 0.01A)
            open_circuit_mask = df[current_col].abs() < 0.01
            on_load_mask = df[current_col].abs() >= 0.01
            
            if open_circuit_mask.any():
                metrics['Max Open Circuit Voltage (V)'] = float(df.loc[open_circuit_mask, voltage_col].max())
            else:
                metrics['Max Open Circuit Voltage (V)'] = None
            
            if on_load_mask.any():
                metrics['Max On-Load Voltage (V)'] = float(df.loc[on_load_mask, voltage_col].max())
            else:
                metrics['Max On-Load Voltage (V)'] = None
        else:
            # If no current column, assume all measurements are on-load
            metrics['Max On-Load Voltage (V)'] = float(df[voltage_col].max())
            metrics['Max Open Circuit Voltage (V)'] = None
    
    if current_col and current_col in df.columns:
        metrics['Max Current (A)'] = float(df[current_col].max())
        metrics['Average Current (A)'] = float(df[current_col].mean())
    
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
        
        metrics['Total Time (min)'] = time_range_minutes
        
        # Calculate activation time and duration
        # Activation Time (Sec): The time when battery FIRST reaches >= min_activation_voltage
        # Duration (Sec): The TOTAL cumulative time when voltage >= min_activation_voltage
        if voltage_col and voltage_col in df.columns:
            # Create mask for when voltage is >= minimum activation voltage
            above_threshold_mask = df[voltage_col] >= min_activation_voltage
            
            if above_threshold_mask.any():
                # Find the FIRST time when voltage >= min_activation_voltage
                first_activation_idx = above_threshold_mask.idxmax()
                
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
                
                # Duration: cumulative sum of time periods when voltage >= threshold
                # Calculate time differences between consecutive points
                time_diffs = time_in_seconds.diff().fillna(0)
                
                # Sum up time intervals where voltage >= threshold
                duration_sec = time_diffs[above_threshold_mask].sum()
                
                metrics['Activation Time (Sec)'] = float(activation_time_sec)
                metrics['Duration (Sec)'] = float(duration_sec)
                
                # Also keep minutes versions for backward compatibility
                metrics['Activation Time (min)'] = float(activation_time_sec / 60)
                metrics['Duration (min)'] = float(duration_sec / 60)
            else:
                # Voltage never reaches the threshold
                metrics['Activation Time (Sec)'] = None
                metrics['Duration (Sec)'] = None
                metrics['Activation Time (min)'] = None
                metrics['Duration (min)'] = None
        
        if voltage_col and voltage_col in df.columns and time_range_minutes > 0:
            voltage_drop = df[voltage_col].iloc[0] - df[voltage_col].iloc[-1]
            metrics['Discharge Rate (V/min)'] = float(voltage_drop / time_range_minutes)
    
    if voltage_col and current_col and voltage_col in df.columns and current_col in df.columns:
        power = df[voltage_col] * df[current_col].abs()
        metrics['Average Power (W)'] = float(power.mean())
        metrics['Max Power (W)'] = float(power.max())
        
        if time_col and time_col in df.columns and len(df) > 1:
            time_diff = time_series.diff().fillna(0)
            
            if pd.api.types.is_timedelta64_dtype(time_diff):
                time_diff_minutes = time_diff.dt.total_seconds() / 60
            else:
                time_diff_minutes = time_diff
            
            energy = (power * time_diff_minutes).sum() / 60
            metrics['Total Energy (Wh)'] = float(energy)
    
    # Calculate extended performance metrics if metadata is provided
    if extended_metadata and current_col and current_col in df.columns and time_col and time_col in df.columns:
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
                time_diff_seconds = pd.Series(time_in_seconds).diff().fillna(0)
                current_abs = df[current_col].abs()
                
                # Total ampere-seconds = sum of (current * time_diff)
                total_ampere_seconds = (current_abs * time_diff_seconds).sum()
                metrics['Total Ampere-Seconds (A·s)'] = float(total_ampere_seconds)
                
                # Calculate per-gram metrics if weights are provided
                total_anode_weight = extended_metadata.get('total_anode_weight', 0)
                total_cathode_weight = extended_metadata.get('total_cathode_weight', 0)
                total_stack_weight = extended_metadata.get('total_stack_weight', 0)
                
                # Ampere-seconds per gram of anode
                if total_anode_weight > 0:
                    metrics['A·s per gram Anode'] = float(total_ampere_seconds / total_anode_weight)
                else:
                    metrics['A·s per gram Anode'] = None
                
                # Ampere-seconds per gram of cathode
                if total_cathode_weight > 0:
                    metrics['A·s per gram Cathode'] = float(total_ampere_seconds / total_cathode_weight)
                else:
                    metrics['A·s per gram Cathode'] = None
                
                # Calorific value per gram of total stack weight
                calorific_value_per_gram = extended_metadata.get('calorific_value_per_gram', 0)
                if calorific_value_per_gram > 0 and total_stack_weight > 0:
                    # This is already a per-gram value, but we include it in metrics for completeness
                    metrics['Calorific Value per gram Stack (kJ/g)'] = float(calorific_value_per_gram)
                    metrics['Total Stack Weight (g)'] = float(total_stack_weight)
                else:
                    metrics['Calorific Value per gram Stack (kJ/g)'] = None
                    metrics['Total Stack Weight (g)'] = float(total_stack_weight) if total_stack_weight > 0 else None
    
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
    report.append("      Duration (Sec) = TOTAL cumulative time when voltage >= min voltage")
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
    if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time, std_duration]):
        report.append("-" * 80)
        report.append("STANDARD PERFORMANCE BENCHMARKS COMPARISON")
        report.append("-" * 80)
        report.append("\nStandard Values:")
        if std_max_onload_voltage:
            report.append(f"  Max On-Load Voltage: {std_max_onload_voltage} V")
        if std_max_oc_voltage:
            report.append(f"  Max Open Circuit Voltage: {std_max_oc_voltage} V")
        if std_activation_time:
            # Display in ms if provided, otherwise show in minutes
            if std_activation_time_ms:
                report.append(f"  Max Activation Time: {std_activation_time_ms} ms ({std_activation_time:.4f} min)")
            else:
                report.append(f"  Max Activation Time: {std_activation_time} min")
        if std_duration:
            # Display in seconds if provided, otherwise show in minutes
            if std_duration_sec:
                report.append(f"  Min Duration: {std_duration_sec} s ({std_duration:.4f} min)")
            else:
                report.append(f"  Min Duration: {std_duration} min")
        report.append("")
        
        for build_name in metrics_df.index:
            report.append(f"\n{build_name} Performance:")
            report.append("-" * 40)
            
            passes = 0
            fails = 0
            
            # Check each metric
            if std_max_onload_voltage and 'Max On-Load Voltage (V)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Max On-Load Voltage (V)']
                if pd.notna(actual):
                    diff = actual - std_max_onload_voltage
                    status = 'PASS' if actual >= std_max_onload_voltage else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    report.append(f"  Max On-Load Voltage: {actual:.4f} V (Std: {std_max_onload_voltage} V, Diff: {diff:+.4f} V) - {status}")
            
            if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)']
                if pd.notna(actual):
                    diff = actual - std_max_oc_voltage
                    status = 'PASS' if actual >= std_max_oc_voltage else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    report.append(f"  Max Open Circuit Voltage: {actual:.4f} V (Std: {std_max_oc_voltage} V, Diff: {diff:+.4f} V) - {status}")
            
            if std_activation_time and 'Activation Time (min)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Activation Time (min)']
                if pd.notna(actual):
                    diff = actual - std_activation_time
                    status = 'PASS' if actual <= std_activation_time else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    report.append(f"  Activation Time: {actual:.4f} min (Std: {std_activation_time} min, Diff: {diff:+.4f} min) - {status}")
            
            if std_duration and 'Duration (min)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Duration (min)']
                if pd.notna(actual):
                    diff = actual - std_duration
                    status = 'PASS' if actual >= std_duration else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    report.append(f"  Duration: {actual:.4f} min (Std: {std_duration} min, Diff: {diff:+.4f} min) - {status}")
            
            total_tests = passes + fails
            if total_tests > 0:
                pass_rate = (passes / total_tests) * 100
                report.append(f"\n  Overall: {passes}/{total_tests} tests passed ({pass_rate:.1f}%)")
        
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
                val1 = metrics_df.loc[build1, col]
                val2 = metrics_df.loc[build2, col]
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
                        std_duration_sec=None):
    """Generate a comprehensive PDF report with all metrics and comparisons"""
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
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    story.append(Paragraph("BATTERY DISCHARGE ANALYSIS", title_style))
    story.append(Paragraph("Detailed Performance Report", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    info_data = [
        ['Generated:', timestamp],
        ['Number of Builds:', str(len(build_names))],
        ['Min. Voltage for Activation:', f"{min_activation_voltage} V"]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Build Information", heading_style))
    build_data = [['Build', 'Battery Code', 'Temperature', 'Build ID']]
    for i, (name, metadata) in enumerate(zip(build_names, metadata_list if metadata_list else [{}]*len(build_names))):
        build_data.append([
            name,
            metadata.get('battery_code', 'N/A') if metadata else 'N/A',
            metadata.get('temperature', 'N/A') if metadata else 'N/A',
            metadata.get('build_id', 'N/A') if metadata else 'N/A'
        ])
    
    build_table = Table(build_data, colWidths=[2*inch, 1.5*inch, 1.2*inch, 1.3*inch])
    build_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ]))
    story.append(build_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Performance Metrics", heading_style))
    
    if metrics_df is not None and not metrics_df.empty:
        key_metrics = ['Max On-Load Voltage (V)', 'Max Open Circuit Voltage (V)', 
                      'Activation Time (Sec)', 'Duration (Sec)']
        
        metrics_data = [['Build'] + [m for m in key_metrics if m in metrics_df.columns]]
        
        for build_name in metrics_df.index:
            row = [build_name]
            for metric in key_metrics:
                if metric in metrics_df.columns:
                    value = metrics_df.loc[build_name, metric]
                    if pd.notna(value):
                        row.append(f"{value:.3f}")
                    else:
                        row.append("N/A")
            metrics_data.append(row)
        
        col_widths = [2*inch] + [1.5*inch] * (len(metrics_data[0]) - 1)
        metrics_table = Table(metrics_data, colWidths=col_widths)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
    
    if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time, std_duration]):
        story.append(PageBreak())
        story.append(Paragraph("Standard Performance Benchmarks", heading_style))
        
        std_data = [['Metric', 'Standard Value']]
        if std_max_onload_voltage:
            std_data.append(['Max On-Load Voltage', f"{std_max_onload_voltage} V"])
        if std_max_oc_voltage:
            std_data.append(['Max Open Circuit Voltage', f"{std_max_oc_voltage} V"])
        if std_activation_time_ms:
            std_data.append(['Max Activation Time', f"{std_activation_time_ms} ms ({std_activation_time:.4f} min)"])
        if std_duration_sec:
            std_data.append(['Min Duration', f"{std_duration_sec} s ({std_duration:.4f} min)"])
        
        std_table = Table(std_data, colWidths=[3*inch, 3*inch])
        std_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ]))
        story.append(std_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Performance Comparison vs Standards", heading_style))
        
        for build_name in metrics_df.index:
            comp_data = [['Metric', 'Actual', 'Standard', 'Difference', 'Status']]
            passes = 0
            fails = 0
            
            if std_max_onload_voltage and 'Max On-Load Voltage (V)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Max On-Load Voltage (V)']
                if pd.notna(actual):
                    diff = actual - std_max_onload_voltage
                    status = 'PASS' if actual >= std_max_onload_voltage else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    comp_data.append(['Max On-Load V', f"{actual:.3f} V", f"{std_max_onload_voltage} V", f"{diff:+.3f} V", status])
            
            if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)']
                if pd.notna(actual):
                    diff = actual - std_max_oc_voltage
                    status = 'PASS' if actual >= std_max_oc_voltage else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    comp_data.append(['Max OC V', f"{actual:.3f} V", f"{std_max_oc_voltage} V", f"{diff:+.3f} V", status])
            
            if std_activation_time and 'Activation Time (Sec)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Activation Time (Sec)']
                if pd.notna(actual):
                    diff_sec = actual - (std_activation_time_ms / 1000 if std_activation_time_ms else std_activation_time * 60)
                    status = 'PASS' if actual <= (std_activation_time_ms / 1000 if std_activation_time_ms else std_activation_time * 60) else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    std_val = f"{std_activation_time_ms} ms" if std_activation_time_ms else f"{std_activation_time} min"
                    comp_data.append(['Activation Time', f"{actual:.3f} sec", std_val, f"{diff_sec:+.3f} sec", status])
            
            if std_duration and 'Duration (Sec)' in metrics_df.columns:
                actual = metrics_df.loc[build_name, 'Duration (Sec)']
                if pd.notna(actual):
                    diff_sec = actual - (std_duration_sec if std_duration_sec else std_duration * 60)
                    status = 'PASS' if actual >= (std_duration_sec if std_duration_sec else std_duration * 60) else 'FAIL'
                    if status == 'PASS':
                        passes += 1
                    else:
                        fails += 1
                    std_val = f"{std_duration_sec} sec" if std_duration_sec else f"{std_duration} min"
                    comp_data.append(['Duration', f"{actual:.3f} sec", std_val, f"{diff_sec:+.3f} sec", status])
            
            story.append(Paragraph(f"<b>{build_name}</b>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            comp_table = Table(comp_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 0.8*inch])
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BACKGROUND', (0, 1), (-1, -1), rl_colors.lightgrey),
            ]))
            
            for i in range(1, len(comp_data)):
                status = comp_data[i][-1]
                if status == 'PASS':
                    comp_table.setStyle(TableStyle([
                        ('BACKGROUND', (-1, i), (-1, i), rl_colors.lightgreen),
                        ('TEXTCOLOR', (-1, i), (-1, i), rl_colors.darkgreen),
                    ]))
                elif status == 'FAIL':
                    comp_table.setStyle(TableStyle([
                        ('BACKGROUND', (-1, i), (-1, i), rl_colors.lightcoral),
                        ('TEXTCOLOR', (-1, i), (-1, i), rl_colors.darkred),
                    ]))
            
            story.append(comp_table)
            
            total_tests = passes + fails
            if total_tests > 0:
                pass_rate = (passes / total_tests) * 100
                summary = f"Overall: {passes}/{total_tests} tests passed ({pass_rate:.1f}%)"
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(summary, styles['Normal']))
            
            story.append(Spacer(1, 0.2*inch))
    
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
    
    return analytics

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
    calorific_values = []
    mean_slopes = []
    slope_variabilities = []
    
    for metrics, analytics in zip(all_metrics_list, all_analytics_list):
        # Ampere-seconds per gram metrics
        if metrics.get('A·s per gram Anode') is not None:
            ampere_secs_anode.append(metrics['A·s per gram Anode'])
        if metrics.get('A·s per gram Cathode') is not None:
            ampere_secs_cathode.append(metrics['A·s per gram Cathode'])
        
        # Calorific value
        if metrics.get('Calorific Value per gram Stack (kJ/g)') is not None:
            calorific_values.append(metrics['Calorific Value per gram Stack (kJ/g)'])
        
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
    
    if len(calorific_values) >= 2 and len(slope_variabilities) >= 2 and len(calorific_values) == len(slope_variabilities):
        try:
            corr_coef, p_value = stats.pearsonr(calorific_values, slope_variabilities)
            correlations['Calorific Value vs Curve Stability'] = {
                'correlation': float(corr_coef),
                'p_value': float(p_value),
                'strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.4 else 'Weak',
                'direction': 'Positive' if corr_coef > 0 else 'Negative',
                'significance': 'Significant (p<0.05)' if p_value < 0.05 else 'Not significant',
                'interpretation': 'Higher calorific value correlates with less stable discharge' if corr_coef > 0 else 'Higher calorific value correlates with more stable discharge'
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

def save_comparison_to_db(name, dataframes, build_names, metrics_df, battery_type=None, extended_metadata=None):
    """Save comparison to database with battery type and extended metadata"""
    if not Session:
        return False, "Database not configured"
    
    try:
        session = Session()
        
        data_list = []
        for df in dataframes:
            data_list.append(df.to_json(orient='split'))
        
        metrics_json = metrics_df.to_json(orient='split') if metrics_df is not None else None
        extended_metadata_json = json.dumps(extended_metadata) if extended_metadata else None
        
        comparison = SavedComparison(
            name=name,
            num_builds=len(build_names),
            build_names=json.dumps(build_names),
            data_json=json.dumps(data_list),
            metrics_json=metrics_json,
            battery_type=battery_type,
            extended_metadata_json=extended_metadata_json
        )
        
        session.add(comparison)
        session.commit()
        session.close()
        
        return True, "Comparison saved successfully"
    except Exception as e:
        return False, f"Error saving comparison: {str(e)}"

def load_comparison_from_db(comparison_id):
    """Load comparison from database with battery type and extended metadata"""
    if not Session:
        return None, None, None, None, None, "Database not configured"
    
    try:
        session = Session()
        comparison = session.query(SavedComparison).filter_by(id=comparison_id).first()
        
        if not comparison:
            session.close()
            return None, None, None, None, None, "Comparison not found"
        
        build_names = json.loads(comparison.build_names)
        data_list = json.loads(comparison.data_json)
        
        dataframes = []
        for data_json in data_list:
            df = pd.read_json(io.StringIO(data_json), orient='split')
            dataframes.append(df)
        
        metrics_df = None
        if comparison.metrics_json:
            metrics_df = pd.read_json(io.StringIO(comparison.metrics_json), orient='split')
        
        battery_type = comparison.battery_type if hasattr(comparison, 'battery_type') else None
        extended_metadata = json.loads(comparison.extended_metadata_json) if comparison.extended_metadata_json else {}
        
        session.close()
        
        return dataframes, build_names, metrics_df, battery_type, extended_metadata, "Loaded successfully"
    except Exception as e:
        return None, None, None, None, None, f"Error loading comparison: {str(e)}"

def get_all_saved_comparisons():
    """Get list of all saved comparisons"""
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
                'num_builds': comp.num_builds
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
        comparison_options = {f"{comp['name']} ({comp['created_at'].strftime('%Y-%m-%d %H:%M')})": comp['id'] 
                             for comp in saved_comparisons}
        
        selected_comparison = st.sidebar.selectbox(
            "Select a saved comparison:",
            options=['-- New Comparison --'] + list(comparison_options.keys()),
            key='load_comparison'
        )
        
        if selected_comparison != '-- New Comparison --':
            comparison_id = comparison_options[selected_comparison]
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📂 Load", key='load_btn'):
                    loaded_dfs, loaded_names, loaded_metrics, loaded_battery_type, loaded_extended_meta, msg = load_comparison_from_db(comparison_id)
                    if loaded_dfs:
                        st.session_state['loaded_dataframes'] = loaded_dfs
                        st.session_state['loaded_build_names'] = loaded_names
                        if loaded_battery_type:
                            st.session_state['battery_type'] = loaded_battery_type
                        if loaded_extended_meta:
                            st.session_state['build_metadata_extended'] = loaded_extended_meta
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            
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
    
    min_activation_voltage = st.sidebar.number_input(
        "Min. voltage for activation (V):",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="Minimum voltage threshold. Activation Time (Sec) = time when battery FIRST reaches ≥ this voltage. Duration (Sec) = TOTAL cumulative time when voltage ≥ this threshold."
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
        
        std_max_oc_voltage = st.sidebar.number_input(
            "Std. Max Open Circuit Voltage (V):",
            min_value=0.0,
            max_value=100.0,
            value=1.6,
            step=0.1
        )
        
        std_activation_time_ms = st.sidebar.number_input(
            "Std. Max Activation Time (ms):",
            min_value=0.0,
            max_value=100000.0,
            value=1000.0,
            step=10.0,
            help="Maximum acceptable time to reach min voltage in milliseconds"
        )
        # Convert ms to minutes for comparison with metrics
        std_activation_time = std_activation_time_ms / 60000.0 if std_activation_time_ms > 0 else None
        
        std_duration_sec = st.sidebar.number_input(
            "Std. Min Duration (s):",
            min_value=0.0,
            max_value=100000.0,
            value=400.0,
            step=10.0,
            help="Minimum acceptable discharge duration in seconds"
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
    
    if 'loaded_dataframes' in st.session_state and 'loaded_build_names' in st.session_state:
        dataframes = st.session_state['loaded_dataframes']
        build_names = st.session_state['loaded_build_names']
        metadata_list = st.session_state.get('loaded_metadata', [{}] * len(dataframes))
        num_builds = len(dataframes)
        
        st.sidebar.success(f"✅ Loaded {num_builds} builds from database")
        
        if st.sidebar.button("Clear Loaded Data"):
            del st.session_state['loaded_dataframes']
            del st.session_state['loaded_build_names']
            if 'loaded_metadata' in st.session_state:
                del st.session_state['loaded_metadata']
            st.rerun()
    else:
        for i in range(num_builds):
            st.sidebar.markdown(f"### Build {i+1}")
            build_name = st.sidebar.text_input(f"Build {i+1} name:", value=f"Build {i+1}", key=f"name_{i}")
            uploaded_file = st.sidebar.file_uploader(
                f"Upload discharge data for {build_name}",
                type=['csv', 'xlsx', 'xls'],
                key=f"file_{i}"
            )
            
            with st.sidebar.expander("⚙️ Extended Build Metadata (Optional)", expanded=False):
                st.markdown("**Weight Inputs (per cell)**")
                col1, col2 = st.columns(2)
                with col1:
                    anode_weight = st.number_input(
                        "Anode weight (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        value=0.0, 
                        step=0.01,
                        key=f"anode_weight_{i}",
                        help="Weight of anode material per cell in grams"
                    )
                    cathode_weight = st.number_input(
                        "Cathode weight (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        value=0.0, 
                        step=0.01,
                        key=f"cathode_weight_{i}",
                        help="Weight of cathode material per cell in grams"
                    )
                with col2:
                    heat_pellet_weight = st.number_input(
                        "Heat pellet (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        value=0.0, 
                        step=0.01,
                        key=f"heat_pellet_{i}",
                        help="Weight of heat pellet in grams"
                    )
                    electrolyte_weight = st.number_input(
                        "Electrolyte (g):", 
                        min_value=0.0, 
                        max_value=1000.0, 
                        value=0.0, 
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
                        value=1, 
                        step=1,
                        key=f"cells_series_{i}",
                        help="Number of cells connected in series"
                    )
                with col4:
                    stacks_in_parallel = st.number_input(
                        "Stacks in parallel:", 
                        min_value=1, 
                        max_value=100, 
                        value=1, 
                        step=1,
                        key=f"stacks_parallel_{i}",
                        help="Number of stacks connected in parallel"
                    )
                
                st.markdown("**Energy Input**")
                calorific_value = st.number_input(
                    "Calorific value (kJ/g):", 
                    min_value=0.0, 
                    max_value=1000.0, 
                    value=0.0, 
                    step=0.1,
                    key=f"calorific_value_{i}",
                    help="Calorific value per gram in kJ/g"
                )
                
                st.session_state['build_metadata_extended'][i] = {
                    'anode_weight_per_cell': anode_weight,
                    'cathode_weight_per_cell': cathode_weight,
                    'heat_pellet_weight': heat_pellet_weight,
                    'electrolyte_weight': electrolyte_weight,
                    'cells_in_series': cells_in_series,
                    'stacks_in_parallel': stacks_in_parallel,
                    'calorific_value_per_gram': calorific_value,
                    'total_anode_weight': anode_weight * cells_in_series if anode_weight > 0 else 0,
                    'total_cathode_weight': cathode_weight * cells_in_series if cathode_weight > 0 else 0,
                    'total_stack_weight': (anode_weight + cathode_weight) * cells_in_series + heat_pellet_weight + electrolyte_weight if (anode_weight + cathode_weight) > 0 else 0,
                    'total_calorific_value': calorific_value * ((anode_weight + cathode_weight) * cells_in_series + heat_pellet_weight + electrolyte_weight) if calorific_value > 0 else 0
                }
            
            if uploaded_file:
                uploaded_files.append(uploaded_file)
                df, metadata = load_data(uploaded_file)
                if df is not None:
                    if metadata and any(metadata.values()):
                        if metadata.get('build_id'):
                            build_name = f"{metadata['build_id']}"
                        if metadata.get('battery_code'):
                            build_name += f" ({metadata['battery_code']})"
                        if metadata.get('temperature'):
                            build_name += f" @ {metadata['temperature']}"
                    
                    build_names.append(build_name)
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
                    extended_metadata=extended_meta
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
                    if 'Max Open Circuit Voltage (V)' in metrics and metrics['Max Open Circuit Voltage (V)'] is not None:
                        st.metric("Max OC V", f"{metrics['Max Open Circuit Voltage (V)']:.3f} V")
                    if 'Activation Time (Sec)' in metrics and metrics['Activation Time (Sec)'] is not None:
                        st.metric("Activation Time", f"{metrics['Activation Time (Sec)']:.2f} sec")
                    if 'Duration (Sec)' in metrics and metrics['Duration (Sec)'] is not None:
                        st.metric("Duration", f"{metrics['Duration (Sec)']:.2f} sec")
        else:
            # For many builds, use a table
            st.subheader("📊 Key Performance Metrics")
            metrics_summary = []
            for metrics in all_metrics:
                summary = {
                    'Build': metrics['Build'],
                    'Max On-Load V': metrics.get('Max On-Load Voltage (V)', 'N/A'),
                    'Max OC V': metrics.get('Max Open Circuit Voltage (V)', 'N/A'),
                    'Activation Time (min)': metrics.get('Activation Time (min)', 'N/A'),
                    'Duration (min)': metrics.get('Duration (min)', 'N/A')
                }
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
                extended_metadata=extended_meta
            )
            metrics['Build'] = name
            all_build_metrics.append(metrics)
        
        if all_build_metrics:
            metrics_df = pd.DataFrame(all_build_metrics)
            metrics_df = metrics_df.set_index('Build')
            
            col1, col2, col3, col4, col5, col6 = st.columns([0.5, 1, 1, 1, 1, 1])
            with col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_data = export_to_excel(dataframes, build_names, metrics_df)
                st.download_button(
                    label="📥 Excel",
                    data=excel_data,
                    file_name=f"battery_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col3:
                csv_data = export_to_csv(dataframes, build_names, metrics_df)
                if csv_data:
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name=f"battery_metrics_{timestamp}.csv",
                        mime="text/csv"
                    )
            with col4:
                # Generate detailed text report
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
                    mime="text/plain"
                )
            with col5:
                # Generate PDF report
                pdf_data = generate_pdf_report(
                    metrics_df, build_names, metadata_list,
                    min_activation_voltage,
                    use_standards, std_max_onload_voltage,
                    std_max_oc_voltage, std_activation_time, std_duration,
                    std_activation_time_ms, std_duration_sec
                )
                st.download_button(
                    label="📕 PDF",
                    data=pdf_data,
                    file_name=f"battery_report_{timestamp}.pdf",
                    mime="application/pdf"
                )
            with col6:
                if DATABASE_URL and Session:
                    if st.button("💾 Save", key='save_comparison_btn'):
                        comparison_name = st.text_input("Comparison name:", value=f"Comparison_{timestamp}", key='comparison_name_input')
                        if comparison_name:
                            current_battery_type = st.session_state.get('battery_type', 'General')
                            current_extended_meta = st.session_state.get('build_metadata_extended', {})
                            success, msg = save_comparison_to_db(
                                comparison_name, dataframes, build_names, metrics_df,
                                battery_type=current_battery_type,
                                extended_metadata=current_extended_meta
                            )
                            if success:
                                st.success(msg)
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
            
            st.dataframe(metrics_df.map(format_value), use_container_width=True)
            
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
                if use_standards and any([std_max_onload_voltage, std_max_oc_voltage, std_activation_time, std_duration]):
                    st.subheader("⭐ Performance vs. Standard Benchmarks")
                    st.markdown("Compare each build against specified standard performance levels")
                    
                    standard_comparison_data = []
                    
                    for build_name in metrics_df.index:
                        build_data = {'Build': build_name}
                        
                        # Max On-Load Voltage comparison
                        if std_max_onload_voltage and 'Max On-Load Voltage (V)' in metrics_df.columns:
                            actual = metrics_df.loc[build_name, 'Max On-Load Voltage (V)']
                            if pd.notna(actual):
                                diff = actual - std_max_onload_voltage
                                status = '✅ Pass' if actual >= std_max_onload_voltage else '❌ Fail'
                                build_data['Max On-Load V (Actual)'] = actual
                                build_data['Max On-Load V (Std)'] = std_max_onload_voltage
                                build_data['Max On-Load V (Diff)'] = diff
                                build_data['Max On-Load V (Status)'] = status
                        
                        # Max Open Circuit Voltage comparison
                        if std_max_oc_voltage and 'Max Open Circuit Voltage (V)' in metrics_df.columns:
                            actual = metrics_df.loc[build_name, 'Max Open Circuit Voltage (V)']
                            if pd.notna(actual):
                                diff = actual - std_max_oc_voltage
                                status = '✅ Pass' if actual >= std_max_oc_voltage else '❌ Fail'
                                build_data['Max OC V (Actual)'] = actual
                                build_data['Max OC V (Std)'] = std_max_oc_voltage
                                build_data['Max OC V (Diff)'] = diff
                                build_data['Max OC V (Status)'] = status
                        
                        # Activation Time comparison (lower is better, so pass if <= standard)
                        if std_activation_time and 'Activation Time (min)' in metrics_df.columns:
                            actual = metrics_df.loc[build_name, 'Activation Time (min)']
                            if pd.notna(actual):
                                diff = actual - std_activation_time
                                status = '✅ Pass' if actual <= std_activation_time else '❌ Fail'
                                build_data['Activation Time (Actual)'] = actual
                                build_data['Activation Time (Std)'] = std_activation_time
                                build_data['Activation Time (Diff)'] = diff
                                build_data['Activation Time (Status)'] = status
                        
                        # Duration comparison (higher is better, so pass if >= standard)
                        if std_duration and 'Duration (min)' in metrics_df.columns:
                            actual = metrics_df.loc[build_name, 'Duration (min)']
                            if pd.notna(actual):
                                diff = actual - std_duration
                                status = '✅ Pass' if actual >= std_duration else '❌ Fail'
                                build_data['Duration (Actual)'] = actual
                                build_data['Duration (Std)'] = std_duration
                                build_data['Duration (Diff)'] = diff
                                build_data['Duration (Status)'] = status
                        
                        standard_comparison_data.append(build_data)
                    
                    if standard_comparison_data:
                        std_comp_df = pd.DataFrame(standard_comparison_data)
                        
                        # Format numeric columns
                        format_dict = {}
                        for col in std_comp_df.columns:
                            if col not in ['Build'] and '(Status)' not in col:
                                format_dict[col] = '{:.4f}'
                        
                        st.dataframe(std_comp_df.style.format(format_dict), use_container_width=True)
                        
                        # Overall pass/fail summary
                        st.markdown("### 📊 Overall Performance Summary")
                        for build_name in metrics_df.index:
                            status_cols = [col for col in std_comp_df.columns if '(Status)' in col]
                            build_row = std_comp_df[std_comp_df['Build'] == build_name].iloc[0]
                            
                            passes = sum(1 for col in status_cols if col in build_row and '✅' in str(build_row[col]))
                            fails = sum(1 for col in status_cols if col in build_row and '❌' in str(build_row[col]))
                            total_tests = passes + fails
                            
                            if total_tests > 0:
                                pass_rate = (passes / total_tests) * 100
                                if pass_rate == 100:
                                    st.success(f"**{build_name}**: {passes}/{total_tests} tests passed ({pass_rate:.0f}%) ✅")
                                elif pass_rate >= 50:
                                    st.warning(f"**{build_name}**: {passes}/{total_tests} tests passed ({pass_rate:.0f}%)")
                                else:
                                    st.error(f"**{build_name}**: {passes}/{total_tests} tests passed ({pass_rate:.0f}%) ❌")
    
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
                    st.info("Not enough data for advanced analytics")
            
            with col2:
                st.markdown("#### Key Insights")
                if analytics:
                    if 'Average Degradation Rate (mV/min)' in analytics:
                        st.metric("Degradation Rate", f"{analytics['Average Degradation Rate (mV/min)']:.2f} mV/min")
                    if 'Energy Efficiency (%)' in analytics:
                        st.metric("Energy Efficiency", f"{analytics['Energy Efficiency (%)']:.1f}%")
                    if 'Voltage Retention (%)' in analytics:
                        st.metric("Voltage Retention", f"{analytics['Voltage Retention (%)']:.1f}%")
            
            st.markdown("#### Anomaly Detection")
            anomalies, anomaly_indices = detect_anomalies(df, voltage_col, time_col)
            
            if anomalies:
                st.warning(f"⚠️ Detected {len(anomalies)} potential anomalies")
                
                with st.expander(f"View {len(anomalies)} anomalies"):
                    for anomaly in anomalies[:20]:
                        st.text(f"• {anomaly}")
                    if len(anomalies) > 20:
                        st.text(f"... and {len(anomalies) - 20} more")
                
                if voltage_col and len(anomaly_indices) > 0:
                    df_plot = downsample_for_plotting(df)
                    fig_anomaly = go.Figure()
                    
                    if time_col and time_col in df_plot.columns:
                        x_data = df_plot[time_col]
                        x_label = f"Time ({time_col})"
                    else:
                        x_data = df_plot.index
                        x_label = "Index"
                    
                    fig_anomaly.add_trace(go.Scatter(
                        x=x_data,
                        y=df_plot[voltage_col],
                        mode='lines',
                        name='Voltage',
                        line=dict(color='blue', width=2)
                    ))
                    
                    anomaly_x = [df[time_col if time_col else 'index'].iloc[i] if time_col and hasattr(df[time_col], 'iloc') else i for i in anomaly_indices if i < len(df)]
                    anomaly_y = [df[voltage_col].iloc[i] for i in anomaly_indices if i < len(df)]
                    
                    fig_anomaly.add_trace(go.Scatter(
                        x=anomaly_x,
                        y=anomaly_y,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    
                    fig_anomaly.update_layout(
                        title=f"{name} - Voltage with Anomalies Highlighted",
                        xaxis_title=x_label,
                        yaxis_title="Voltage (V)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_anomaly, use_container_width=True)
            else:
                st.success("✅ No significant anomalies detected")
            
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
                    extended_metadata=extended_meta
                )
                analytics = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
                
                all_metrics_for_corr.append(metrics)
                all_analytics_for_corr.append(analytics)
            
            # Calculate correlations
            correlations = calculate_correlation_analysis(all_metrics_for_corr, all_analytics_for_corr)
            
            if correlations:
                st.success(f"✅ Found {len(correlations)} correlation relationships")
                
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
                st.info("ℹ️ Correlation analysis requires at least 2 builds with extended metadata (weights, calorific value) to be entered.")
                st.markdown("**To enable correlation analysis:**")
                st.markdown("1. Enter extended build metadata (anode/cathode weights, calorific value) for at least 2 builds")
                st.markdown("2. The analysis will automatically calculate correlations between:")
                st.markdown("   - Ampere-seconds per gram vs discharge slope")
                st.markdown("   - Calorific value vs curve stability")
                st.markdown("   - Anode vs cathode performance")
    
    with tab5:
        st.header("Data Preview")
        
        for df, name in zip(dataframes, build_names):
            with st.expander(f"📄 {name} - Data Preview ({len(df)} rows)"):
                time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                
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
                extended_metadata=extended_meta
            )
            advanced = calculate_advanced_analytics(df, time_col, voltage_col, current_col)
            
            battery_code = metadata.get('battery_code', '') if metadata else ''
            build_id = metadata.get('build_id', '') if metadata else ''
            
            temp_data.append({
                'Build Name': name,
                'Battery Code': battery_code if battery_code else 'N/A',
                'Build ID': build_id if build_id else 'N/A',
                'Temperature (°C)': temp if temp is not None else 'Not specified',
                'Total Time (min)': metrics.get('Total Time (min)', 0),
                'Voltage Range (V)': metrics.get('Voltage Range (V)', 0),
                'Avg Degradation (mV/min)': advanced.get('Average Degradation Rate (mV/min)', 0),
                'Voltage Retention (%)': advanced.get('Voltage Retention (%)', 0),
                'Total Energy (Wh)': advanced.get('Total Energy Discharged (Wh)', metrics.get('Total Energy (Wh)', 0)),
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
                            y=numeric_temp_df['Total Time (min)'],
                            mode='lines+markers',
                            name='Discharge Time',
                            marker=dict(size=10)
                        ))
                        fig1.update_layout(
                            title='Discharge Time vs Temperature',
                            xaxis_title='Temperature (°C)',
                            yaxis_title='Total Discharge Time (min)',
                            height=400
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=numeric_temp_df['Temperature (°C)'],
                            y=numeric_temp_df['Total Energy (Wh)'],
                            mode='lines+markers',
                            name='Energy',
                            marker=dict(size=10, color='orange')
                        ))
                        fig2.update_layout(
                            title='Energy Output vs Temperature',
                            xaxis_title='Temperature (°C)',
                            yaxis_title='Total Energy (Wh)',
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
                    best_temp_idx = numeric_temp_df['Total Energy (Wh)'].idxmax()
                    worst_temp_idx = numeric_temp_df['Total Energy (Wh)'].idxmin()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Best Performance Temperature",
                            f"{numeric_temp_df.loc[best_temp_idx, 'Temperature (°C)']}°C",
                            f"{numeric_temp_df.loc[best_temp_idx, 'Total Energy (Wh)']:.2f} Wh"
                        )
                    with col2:
                        st.metric(
                            "Worst Performance Temperature",
                            f"{numeric_temp_df.loc[worst_temp_idx, 'Temperature (°C)']}°C",
                            f"{numeric_temp_df.loc[worst_temp_idx, 'Total Energy (Wh)']:.2f} Wh"
                        )
                    with col3:
                        energy_range = numeric_temp_df['Total Energy (Wh)'].max() - numeric_temp_df['Total Energy (Wh)'].min()
                        avg_energy = numeric_temp_df['Total Energy (Wh)'].mean()
                        pct_variation = (energy_range / avg_energy * 100) if avg_energy > 0 else 0
                        st.metric(
                            "Energy Variation",
                            f"{pct_variation:.1f}%",
                            f"Range: {energy_range:.2f} Wh"
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
