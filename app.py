import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import io

st.set_page_config(page_title="Battery Discharge Analysis", layout="wide")

st.title("🔋 Battery Discharge Data Analysis")
st.markdown("Compare discharge curves and performance metrics across different builds")

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_columns(df):
    """Auto-detect relevant columns in the dataset"""
    columns = df.columns.str.lower()
    
    time_col = None
    voltage_col = None
    current_col = None
    capacity_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower:
            time_col = col
        elif 'voltage' in col_lower or 'volt' in col_lower:
            voltage_col = col
        elif 'current' in col_lower or 'amp' in col_lower:
            current_col = col
        elif 'capacity' in col_lower or 'cap' in col_lower:
            capacity_col = col
    
    return time_col, voltage_col, current_col, capacity_col

def calculate_metrics(df, time_col, voltage_col, current_col=None):
    """Calculate key battery metrics"""
    metrics = {}
    
    if voltage_col and voltage_col in df.columns:
        metrics['Max Voltage (V)'] = df[voltage_col].max()
        metrics['Min Voltage (V)'] = df[voltage_col].min()
        metrics['Average Voltage (V)'] = df[voltage_col].mean()
        metrics['Voltage Range (V)'] = metrics['Max Voltage (V)'] - metrics['Min Voltage (V)']
    
    if current_col and current_col in df.columns:
        metrics['Max Current (A)'] = df[current_col].max()
        metrics['Average Current (A)'] = df[current_col].mean()
    
    if time_col and time_col in df.columns:
        metrics['Total Time (min)'] = (df[time_col].max() - df[time_col].min())
        
        if voltage_col and voltage_col in df.columns:
            voltage_drop = df[voltage_col].iloc[0] - df[voltage_col].iloc[-1]
            time_range = df[time_col].iloc[-1] - df[time_col].iloc[0]
            if time_range > 0:
                metrics['Discharge Rate (V/min)'] = voltage_drop / time_range
    
    if voltage_col and current_col and voltage_col in df.columns and current_col in df.columns:
        power = df[voltage_col] * df[current_col].abs()
        metrics['Average Power (W)'] = power.mean()
        metrics['Max Power (W)'] = power.max()
        
        if time_col and time_col in df.columns and len(df) > 1:
            time_diff = df[time_col].diff().fillna(0)
            energy = (power * time_diff).sum() / 60
            metrics['Total Energy (Wh)'] = energy
    
    return metrics

st.sidebar.header("📁 Data Upload")
st.sidebar.markdown("Upload 2-3 battery discharge data files for comparison")

num_builds = st.sidebar.radio("Number of builds to compare:", [2, 3], index=0)

uploaded_files = []
build_names = []
dataframes = []

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
        build_names.append(build_name)
        df = load_data(uploaded_file)
        if df is not None:
            dataframes.append(df)

if len(dataframes) == num_builds:
    st.success(f"✅ All {num_builds} files loaded successfully!")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Discharge Curves", "📈 Multi-Parameter Analysis", "📋 Metrics Comparison", "📄 Data Preview"])
    
    with tab1:
        st.header("Voltage Discharge Curves")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        all_metrics = []
        
        for idx, (df, name) in enumerate(zip(dataframes, build_names)):
            time_col, voltage_col, current_col, capacity_col = detect_columns(df)
            
            if voltage_col:
                x_axis = None
                x_label = ""
                
                if capacity_col and capacity_col in df.columns:
                    x_axis = df[capacity_col]
                    x_label = f"Capacity ({capacity_col})"
                elif time_col and time_col in df.columns:
                    x_axis = df[time_col]
                    x_label = f"Time ({time_col})"
                else:
                    x_axis = df.index
                    x_label = "Data Point"
                
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=df[voltage_col],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    hovertemplate=f'<b>{name}</b><br>{x_label}: %{{x}}<br>Voltage: %{{y:.3f}} V<extra></extra>'
                ))
                
                metrics = calculate_metrics(df, time_col, voltage_col, current_col)
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
        
        col1, col2, col3 = st.columns(num_builds)
        for idx, metrics in enumerate(all_metrics):
            with [col1, col2, col3][idx]:
                st.markdown(f"### {metrics['Build']}")
                if 'Max Voltage (V)' in metrics:
                    st.metric("Max Voltage", f"{metrics['Max Voltage (V)']:.3f} V")
                if 'Min Voltage (V)' in metrics:
                    st.metric("Min Voltage", f"{metrics['Min Voltage (V)']:.3f} V")
                if 'Average Voltage (V)' in metrics:
                    st.metric("Avg Voltage", f"{metrics['Average Voltage (V)']:.3f} V")
    
    with tab2:
        st.header("Multi-Parameter Time-Series Analysis")
        
        has_time = any([detect_columns(df)[0] for df in dataframes])
        
        if has_time:
            for idx, (df, name) in enumerate(zip(dataframes, build_names)):
                time_col, voltage_col, current_col, capacity_col = detect_columns(df)
                
                if time_col and time_col in df.columns:
                    params = []
                    if voltage_col and voltage_col in df.columns:
                        params.append(('Voltage (V)', voltage_col))
                    if current_col and current_col in df.columns:
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
                                    x=df[time_col],
                                    y=df[param_col],
                                    mode='lines',
                                    name=param_name,
                                    line=dict(color=colors[idx % len(colors)], width=2)
                                ),
                                row=i+1, col=1
                            )
                            
                            fig.update_yaxis(title_text=param_name, row=i+1, col=1)
                        
                        fig.update_xaxis(title_text=f"Time ({time_col})", row=len(params), col=1)
                        fig.update_layout(
                            height=400 * len(params),
                            showlegend=False,
                            title_text=f"{name} - Time Series Analysis",
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if voltage_col and current_col and voltage_col in df.columns and current_col in df.columns:
                            power = df[voltage_col] * df[current_col].abs()
                            
                            fig_power = go.Figure()
                            fig_power.add_trace(go.Scatter(
                                x=df[time_col],
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
            metrics = calculate_metrics(df, time_col, voltage_col, current_col)
            metrics['Build'] = name
            all_build_metrics.append(metrics)
        
        if all_build_metrics:
            metrics_df = pd.DataFrame(all_build_metrics)
            metrics_df = metrics_df.set_index('Build')
            
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
                st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
                
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
    
    with tab4:
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

else:
    st.info(f"👆 Please upload {num_builds} discharge data files using the sidebar to begin analysis.")
    
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        ### Data Format Requirements
        
        Your discharge data files should contain the following columns (column names will be auto-detected):
        
        - **Time**: Time stamps for each measurement (e.g., "Time", "time", "Time(s)")
        - **Voltage**: Battery voltage measurements (e.g., "Voltage", "Voltage(V)", "volt")
        - **Current**: Current measurements - optional (e.g., "Current", "Current(A)", "Amp")
        - **Capacity**: Capacity measurements - optional (e.g., "Capacity", "Cap", "Capacity(Ah)")
        
        ### Steps to Analyze
        
        1. Select the number of builds you want to compare (2 or 3)
        2. Give each build a descriptive name
        3. Upload the corresponding CSV or Excel file for each build
        4. Explore the different tabs:
           - **Discharge Curves**: Compare voltage curves side-by-side
           - **Multi-Parameter Analysis**: View voltage, current, and power over time
           - **Metrics Comparison**: See calculated metrics and statistics
           - **Data Preview**: Inspect your raw data
        
        ### Supported File Formats
        
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("Battery Discharge Analysis Tool v1.0 - Compare discharge performance across different builds")
