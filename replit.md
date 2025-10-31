# Battery Discharge Analysis Application

## Overview

This is a Streamlit-based web application for analyzing and comparing **primary (non-rechargeable) battery** discharge data across different temperature conditions. The application is optimized for high-frequency discharge data (e.g., 10ms sampling intervals) from battery testing equipment. Users can upload discharge datasets (CSV/Excel), visualize discharge curves, compare temperature-dependent performance metrics, and save comparisons for future reference. It provides interactive visualizations using Plotly and statistical analysis capabilities using SciPy.

## Current Use Case

**Primary Battery Temperature Testing**:
- Battery type: Primary (non-rechargeable) batteries
- Test conditions: Multiple temperatures (e.g., 25°C, 0°C, -20°C)
- Data characteristics: High-frequency sampling (10ms intervals, 300k+ data points per test)
- Duration: Typical tests run for ~3000 seconds (50 minutes)
- Metrics focus: Voltage retention, energy output, degradation rates across temperatures

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
```

**Design Decision**: JSON serialization in text columns
- **Rationale**: Provides flexibility for varying data structures across different comparison types without rigid schema constraints
- **Trade-off**: Sacrifices query performance on nested data for schema flexibility
- **Alternative considered**: Normalized relational schema with separate tables for builds and metrics (rejected due to complexity for read-heavy use case)

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
- Row 1: `Battery Code` | `<battery_identifier>` (e.g., "ALK-AA-001")
- Row 2: `Temperature` | `<temperature_value>` (e.g., "25°C")
- Row 3: `Build ID` | `<build_identifier>` (e.g., "Build_A")
- Row 4+: Data table with column headers and measurements

**Metadata Usage**:
- Battery Code: Identifies the specific battery being tested
- Temperature: Test temperature (extracted and used for temperature comparison analysis)
- Build ID: Build identifier for comparing different builds at the same temperature
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