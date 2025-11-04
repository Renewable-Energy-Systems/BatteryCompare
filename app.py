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

def calculate_metrics(df, time_col, voltage_col, current_col=None, min_activation_voltage=1.0, end_discharge_voltage=0.9):
    """Calculate key battery metrics including new performance metrics"""
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
        
        # Calculate activation time (time to reach min_activation_voltage during discharge)
        # For discharge curves, voltage starts high and drops, so we look for when it drops to the threshold
        if voltage_col and voltage_col in df.columns:
            activation_mask = df[voltage_col] <= min_activation_voltage
            if activation_mask.any():
                first_activation_idx = activation_mask.idxmax()
                activation_time = time_series.iloc[first_activation_idx] - time_series.iloc[0]
                
                if pd.api.types.is_timedelta64_dtype(activation_time):
                    activation_time_minutes = activation_time.total_seconds() / 60
                elif isinstance(activation_time, pd.Timedelta):
                    activation_time_minutes = activation_time.total_seconds() / 60
                else:
                    activation_time_minutes = float(activation_time)
                
                metrics['Activation Time (min)'] = activation_time_minutes
            else:
                metrics['Activation Time (min)'] = None
            
            # Calculate duration (time to reach end_discharge_voltage)
            end_mask = df[voltage_col] <= end_discharge_voltage
            if end_mask.any():
                first_end_idx = end_mask.idxmax()
                duration_time = time_series.iloc[first_end_idx] - time_series.iloc[0]
                
                if pd.api.types.is_timedelta64_dtype(duration_time):
                    duration_minutes = duration_time.total_seconds() / 60
                elif isinstance(duration_time, pd.Timedelta):
                    duration_minutes = duration_time.total_seconds() / 60
                else:
                    duration_minutes = float(duration_time)
                
                metrics['Duration (min)'] = duration_minutes
            else:
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
    
    return analytics

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

def save_comparison_to_db(name, dataframes, build_names, metrics_df):
    """Save comparison to database"""
    if not Session:
        return False, "Database not configured"
    
    try:
        session = Session()
        
        data_list = []
        for df in dataframes:
            data_list.append(df.to_json(orient='split'))
        
        metrics_json = metrics_df.to_json(orient='split') if metrics_df is not None else None
        
        comparison = SavedComparison(
            name=name,
            num_builds=len(build_names),
            build_names=json.dumps(build_names),
            data_json=json.dumps(data_list),
            metrics_json=metrics_json
        )
        
        session.add(comparison)
        session.commit()
        session.close()
        
        return True, "Comparison saved successfully"
    except Exception as e:
        return False, f"Error saving comparison: {str(e)}"

def load_comparison_from_db(comparison_id):
    """Load comparison from database"""
    if not Session:
        return None, None, None, "Database not configured"
    
    try:
        session = Session()
        comparison = session.query(SavedComparison).filter_by(id=comparison_id).first()
        
        if not comparison:
            session.close()
            return None, None, None, "Comparison not found"
        
        build_names = json.loads(comparison.build_names)
        data_list = json.loads(comparison.data_json)
        
        dataframes = []
        for data_json in data_list:
            df = pd.read_json(io.StringIO(data_json), orient='split')
            dataframes.append(df)
        
        metrics_df = None
        if comparison.metrics_json:
            metrics_df = pd.read_json(io.StringIO(comparison.metrics_json), orient='split')
        
        session.close()
        
        return dataframes, build_names, metrics_df, "Loaded successfully"
    except Exception as e:
        return None, None, None, f"Error loading comparison: {str(e)}"

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
                    loaded_dfs, loaded_names, loaded_metrics, msg = load_comparison_from_db(comparison_id)
                    if loaded_dfs:
                        st.session_state['loaded_dataframes'] = loaded_dfs
                        st.session_state['loaded_build_names'] = loaded_names
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
end_discharge_voltage = 0.9
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
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="Minimum voltage threshold to calculate activation time"
    )
    
    end_discharge_voltage = st.sidebar.number_input(
        "Voltage at end of discharge (V):",
        min_value=0.0,
        max_value=10.0,
        value=0.9,
        step=0.01,
        help="Voltage threshold to calculate discharge duration"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Standard Performance Benchmarks")
    st.sidebar.markdown("*Optional: Set target values for comparison*")
    
    use_standards = st.sidebar.checkbox("Enable standard performance comparison", value=False)
    
    if use_standards:
        std_max_onload_voltage = st.sidebar.number_input(
            "Std. Max On-Load Voltage (V):",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.01
        )
        
        std_max_oc_voltage = st.sidebar.number_input(
            "Std. Max Open Circuit Voltage (V):",
            min_value=0.0,
            max_value=10.0,
            value=1.6,
            step=0.01
        )
        
        std_activation_time = st.sidebar.number_input(
            "Std. Max Activation Time (min):",
            min_value=0.0,
            max_value=1000.0,
            value=1.0,
            step=0.1,
            help="Maximum acceptable time to reach min voltage"
        )
        
        std_duration = st.sidebar.number_input(
            "Std. Min Duration (min):",
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=1.0,
            help="Minimum acceptable discharge duration"
        )
    else:
        std_max_onload_voltage = None
        std_max_oc_voltage = None
        std_activation_time = None
        std_duration = None
    
    st.sidebar.markdown("---")
    
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
                'Build Name': name,
                'Battery Code': metadata.get('battery_code', 'N/A') if metadata else 'N/A',
                'Temperature': metadata.get('temperature', 'N/A') if metadata else 'N/A',
                'Build ID': metadata.get('build_id', 'N/A') if metadata else 'N/A'
            }
            build_info_data.append(info)
        
        build_info_df = pd.DataFrame(build_info_data)
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
                
                metrics = calculate_metrics(
                    df, time_col, voltage_col, current_col, 
                    min_activation_voltage=min_activation_voltage,
                    end_discharge_voltage=end_discharge_voltage
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
                    if 'Activation Time (min)' in metrics and metrics['Activation Time (min)'] is not None:
                        st.metric("Activation Time", f"{metrics['Activation Time (min)']:.2f} min")
                    if 'Duration (min)' in metrics and metrics['Duration (min)'] is not None:
                        st.metric("Duration", f"{metrics['Duration (min)']:.2f} min")
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
        
        for df, name in zip(dataframes, build_names):
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            metrics = calculate_metrics(
                df, time_col, voltage_col, current_col,
                min_activation_voltage=min_activation_voltage,
                end_discharge_voltage=end_discharge_voltage
            )
            metrics['Build'] = name
            all_build_metrics.append(metrics)
        
        if all_build_metrics:
            metrics_df = pd.DataFrame(all_build_metrics)
            metrics_df = metrics_df.set_index('Build')
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_data = export_to_excel(dataframes, build_names, metrics_df)
                st.download_button(
                    label="📥 Export Excel",
                    data=excel_data,
                    file_name=f"battery_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col3:
                csv_data = export_to_csv(dataframes, build_names, metrics_df)
                if csv_data:
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"battery_metrics_{timestamp}.csv",
                        mime="text/csv"
                    )
            with col4:
                if DATABASE_URL and Session:
                    if st.button("💾 Save to DB", key='save_comparison_btn'):
                        comparison_name = st.text_input("Comparison name:", value=f"Comparison_{timestamp}", key='comparison_name_input')
                        if comparison_name:
                            success, msg = save_comparison_to_db(comparison_name, dataframes, build_names, metrics_df)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
            
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
            
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
                    
                    def format_value(val):
                        if isinstance(val, (int, float, np.number)):
                            return f"{val:.4f}"
                        else:
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
            metrics = calculate_metrics(
                df, time_col, voltage_col, current_col,
                min_activation_voltage=min_activation_voltage,
                end_discharge_voltage=end_discharge_voltage
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
