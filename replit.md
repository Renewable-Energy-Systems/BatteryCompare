# Battery Discharge Analysis Application

## Overview

This is a Streamlit-based web application for analyzing and comparing **primary (non-rechargeable) battery** discharge data across different temperature conditions. The application supports battery type organization, allowing users to select and track different battery chemistries separately (Madhava, low-voltage, high-voltage, or custom types). The application is optimized for high-frequency discharge data (e.g., 10ms sampling intervals) from battery testing equipment. Users can upload **unlimited** discharge datasets (CSV/Excel), visualize discharge curves, compare performance metrics against each other and standard benchmarks, and save comparisons for future reference. It provides interactive visualizations using Plotly and statistical analysis capabilities using SciPy.

## Recent Updates (Nov 2025)

**Metrics Redesign - Seconds-Only Display** (Nov 13, 2025):
- **Removed Metrics**: Eliminated Min Voltage, Voltage Range, Total Energy (Wh), and all minute-based time displays
- **New Advanced Metrics**:
  - **Weighted Average Voltage (V)**: Time-weighted average voltage during discharge period (activation to last cutoff)
  - **Voltage at Targeted Duration (V)**: Optional metric using linear interpolation to find voltage at user-specified target time
  - **Max OC Voltage Time (s)**: Timestamp when maximum open circuit voltage occurred
  - **Max On-Load Current (A)**: Current value when maximum on-load voltage occurred
  - **Max On-Load Time (s)**: Timestamp when maximum on-load voltage occurred
- **Time Display Standardization**: ALL timing values now display exclusively in seconds (no minutes)
  - Activation Time: Displayed in seconds only
  - Duration: Displayed in seconds only
  - Total Time: Displayed in seconds only
  - Discharge Rate: Now V/sec (formerly V/min)
- **Standard Benchmarks Update**: 
  - Max Activation Time: Input in milliseconds, displayed as "ms (s)" in reports
  - Min Duration: Input in seconds only
  - All export functions (text, PDF, Excel, CSV) use seconds-based metrics
- **Target Duration Feature**: Optional checkbox to specify target duration for voltage interpolation analysis

**Extended Build Metadata & Advanced Metrics** (Nov 10, 2025):
- **Extended Build Information Form**: 7 input fields per build for detailed battery construction data
  - Weight inputs: Anode weight per cell, Cathode weight per cell, Heat pellet weight per cell, Electrolyte weight per cell
  - Configuration: Cells in series, Stacks in parallel
  - Energy: Calorific value per gram stack (kJ/g)
- **Automatic Weight Calculations**: Total weights for all cells in parallel
  - Total Anode Weight (all cells) = anode_weight_per_cell × stacks_in_parallel
  - Total Cathode Weight (all cells) = cathode_weight_per_cell × stacks_in_parallel
  - Total Stack Weight = sum of all components × cells_in_series
- **Advanced Performance Metrics**:
  - **Total Ampere-Seconds (A·s)**: Calculated ONLY during discharge period (from activation to last occurrence of cutoff voltage)
  - **A·s per gram Anode**: Total A·s / Total anode weight of all cells in parallel
  - **A·s per gram Cathode**: Total A·s / Total cathode weight of all cells in parallel
- **Discharge Curve Analysis**:
  - **ΔV/ΔT at 5-second intervals**: Voltage rate of change analysis for curve characterization
  - **Constant ΔV/ΔT Region**: Finds longest time period where ΔV/ΔT remains within ±5% deviation
  - **Actionable Output**: Reports as "from X seconds to Y seconds" with duration
  - **Plateau Detection**: Uses absolute tolerance floor (1mV/s) to handle near-flat discharge curves
- **Correlation Analysis**:
  - **Performance Correlations**: A·s/gram (anode/cathode) vs discharge slope
  - **Duration Correlations**: Actual duration vs total anode weight, total cathode weight, and total calorific value
  - **Values Table**: Shows actual values used in correlation computations
- **Database Schema Updates**:
  - Added `battery_type` column to SavedComparison table for filtering saved analyses by battery chemistry
  - Added `extended_metadata_json` column for persisting build construction details

**Battery Type Organization** (Nov 2025):
- Battery type selector at app start (General, Madhava, Low-Voltage, High-Voltage, Custom)
- All analysis and data organized by selected battery type
- Session state management for battery type persistence

**Unlimited Build Support**:
- Removed 3-build limitation - now supports 1-50 builds per comparison
- Dynamic visualization adapts to number of builds (columns for ≤4, table view for >4)
- Enhanced file processing with automatic cleanup of empty columns/rows

**Core Performance Metrics**:
- **Max On-Load Voltage**: Maximum voltage when current is flowing (≥0.01A)
- **Max Open Circuit Voltage**: Maximum voltage when current is near zero (<0.01A)
- **Activation Time (Sec)**: The time when battery FIRST reaches ≥ minimum voltage for activation
- **Duration (Sec)**: Time from activation to LAST occurrence of cutoff voltage (the time span from when battery first activates to the last time it reaches the cutoff voltage)

**Standard Performance Benchmarks**:
- Optional comparison against target performance levels
- Shows actual values, standard values, and differences for each metric
- Configurable standard values for metrics:
  - Max On-Load Voltage: Optional (can be NULL via checkbox)
  - Max Open Circuit Voltage: Required when benchmarks enabled
  - Max Activation Time: Input in milliseconds (ms)
  - Min Duration: Input in seconds (s)

**Detailed Reporting**:
- Comprehensive text report generation with all test data
- Includes: test configuration, build information, performance metrics, statistical summary
- Standard benchmark comparisons showing actual vs standard values with differences
- Build-to-build comparisons with differences and percentage changes
- Downloadable in multiple formats: Excel, CSV, formatted text report, and **PDF**
- **PDF Reports**: Professional formatted reports with tables, build information, and performance comparisons

## Current Use Case

**Primary Battery Temperature Testing**:
- Battery type: Primary (non-rechargeable) batteries
- Test conditions: Multiple temperatures (e.g., 25°C, 0°C, -20°C)
- Data characteristics: High-frequency sampling (10ms intervals, 300k+ data points per test)
- Duration: Typical tests run for ~3000 seconds (50 minutes)
- Metrics focus: Voltage retention, energy output, degradation rates across temperatures

**Voltage Range Support**:
- Low-voltage batteries: 0.9-1.0V typical operating range
- High-voltage batteries: 27-35V typical operating range (e.g., madhava project)
- All input fields support 0-100V range to accommodate diverse battery types

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit framework
- **Rationale**: Streamlit enables rapid development of data-focused web applications with minimal boilerplate code, making it ideal for scientific/engineering data analysis tools
- **Layout**: Wide layout configuration for better visualization of multiple discharge curves
- **Interactivity**: File upload widgets for CSV/Excel data ingestion, interactive Plotly charts for data exploration

### Backend Architecture

**Application Structure**: Single-file Streamlit application (app.py)
- **Rationale**: For a data analysis tool of this scope, a monolithic structure keeps all logic accessible and maintainable
- **Main entry point**: main.py serves as a minimal launcher (currently underutilized)

**Data Processing Layer**:
- **Pandas**: Primary data manipulation and analysis library for handling discharge curve datasets
- **NumPy**: Numerical computations and array operations
- **SciPy (stats module)**: Statistical analysis of battery performance metrics

**Visualization Layer**:
- **Plotly Graph Objects & Express**: Interactive charting library
- **Subplots capability**: Enables side-by-side comparison of multiple discharge curves
- **Rationale**: Plotly chosen over static plotting libraries (matplotlib) for superior interactivity, zoom capabilities, and hover tooltips essential for detailed data analysis

### Data Storage

**Database**: PostgreSQL (via SQLAlchemy ORM)
- **Connection**: Environment-variable based configuration (`DATABASE_URL`)
- **Rationale**: Relational database chosen for structured storage of saved comparisons with querying capabilities
- **Graceful degradation**: Application checks for database availability and functions without persistence if unavailable

**Schema Design**:
```
SavedComparison:
- id (Primary Key)
- name (Comparison identifier)
- created_at (Timestamp)
- num_builds (Count of builds in comparison)
- build_names (Text field - serialized build identifiers)
- data_json (Text field - serialized discharge data)
- metrics_json (Text field - serialized performance metrics)
- battery_type (VARCHAR(50) - Battery chemistry type: General, Madhava, Low-Voltage, High-Voltage, Custom)
- extended_metadata_json (Text field - JSON serialized extended build metadata: weights, configuration, calorific value)
```

**Design Decision**: JSON serialization in text columns
- **Rationale**: Provides flexibility for varying data structures across different comparison types without rigid schema constraints
- **Trade-off**: Sacrifices query performance on nested data for schema flexibility
- **Alternative considered**: Normalized relational schema with separate tables for builds and metrics (rejected due to complexity for read-heavy use case)
- **Extended Metadata Structure**: Stored as JSON to accommodate variable number of builds and optional fields per build

### Authentication & Authorization

**Current State**: No authentication implemented
- **Implication**: Application designed for single-user or trusted environment usage
- **Future consideration**: Would require adding Streamlit authentication or external auth provider for multi-tenant deployments

### File Processing

**Supported Formats**: CSV and Excel (.xlsx, .xls)
- **Rationale**: Covers the two most common formats for exporting battery test data from laboratory equipment
- **Processing**: In-memory file handling using io module for uploaded files

**File Structure with Metadata**:
The application supports extracting metadata from the top rows of data files:
- Row 1: `Battery Code` | `<battery_identifier>` (e.g., "48")
- Row 2: `Temperature` | `<temperature_value>` (e.g., "25")
- Row 3: `Build Number` | `<build_identifier>` (e.g., "1")
- Row 4+: Data table with column headers and measurements

**Flexible Metadata Extraction** (Nov 2025 Update):
- Metadata can be in any column position (e.g., columns 0-1 or 4-5)
- Application automatically scans all columns to locate metadata labels
- Empty columns and rows are automatically removed
- Column name variations handled (e.g., "Dicharge Current" typo)

**Metadata Usage**:
- Battery Code: Identifies the specific battery being tested
- Temperature: Test temperature (extracted and used for temperature comparison analysis)
- Build Number/ID: Build identifier for comparing different builds at the same temperature
- If metadata is not present in file, the application falls back to parsing build names

**Comparison Scenarios Supported**:
1. Different builds at the same temperature (compare battery quality/performance)
2. Same battery at different temperatures (temperature performance analysis)
3. Mixed comparisons (multiple batteries and temperatures)

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework and UI components

### Data Processing & Analysis
- **Pandas**: Tabular data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and statistical analysis

### Visualization
- **Plotly**: Interactive charting (both graph_objects and express modules)

### Database & ORM
- **SQLAlchemy**: Database abstraction layer and ORM
- **PostgreSQL**: Relational database (connection via DATABASE_URL environment variable)
  - Note: Database is optional; application operates without persistence if unavailable

### Python Standard Library
- **io**: In-memory file operations for upload handling
- **datetime**: Timestamp generation for saved comparisons
- **os**: Environment variable access
- **json**: Data serialization for database storage

### Environment Configuration
- **DATABASE_URL**: PostgreSQL connection string (optional)
  - Format: `postgresql://user:password@host:port/database`
  - If not provided, application runs without save/load functionality