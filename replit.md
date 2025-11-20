# Battery Discharge Analysis Application

## Overview

This Streamlit application analyzes and compares primary (non-rechargeable) battery discharge data across various temperatures. It supports organizing data by battery type (e.g., Madhava, low-voltage, high-voltage, custom), handles high-frequency discharge data from testing equipment, and allows users to upload unlimited datasets (CSV/Excel). Key features include interactive discharge curve visualizations using Plotly, performance metric comparisons against benchmarks, and statistical analysis via SciPy. The application aims to provide detailed insights into battery performance, such as voltage retention and degradation rates, especially for temperature testing scenarios.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit framework is used for rapid development of data-focused web applications. It utilizes a wide layout for optimal visualization and interactive widgets for file uploads and data exploration.

### Backend Architecture

**Application Structure**: A monolithic `app.py` structure centralizes logic for maintainability.
**Data Processing Layer**: Employs Pandas for data manipulation, NumPy for numerical operations, and SciPy for statistical analysis.
**Visualization Layer**: Plotly (Graph Objects & Express) provides interactive charts, including subplots for comparing multiple discharge curves, chosen for its superior interactivity over static alternatives.
**File Processing**: Supports CSV and Excel formats, processing files in-memory. It includes robust metadata extraction from file headers (Battery Code, Temperature, Build Number) and handles variations in column positions and empty data.

### Data Storage

**Database**: PostgreSQL (via SQLAlchemy ORM) is used for structured storage of saved comparisons, configured via `DATABASE_URL`. The application is designed to function without persistence if the database is unavailable.
**Schema Design**: The `SavedComparison` table stores comparison metadata, serialized discharge data, and performance metrics, including `battery_type` and `extended_metadata_json`. JSON serialization is used for flexibility with varying data structures, prioritizing schema flexibility over query performance on nested data.

### System Design Choices

**Metrics Redesign**: All timing values are standardized to seconds. Advanced metrics include Weighted Average Voltage, Voltage at Targeted Duration (interpolated), Max OC Voltage Time, Max On-Load Current, and Max On-Load Time.
**Extended Build Metadata**: Supports detailed battery construction data (e.g., weights, cell configuration, calorific value) with automatic total weight calculations and advanced performance metrics like Ampere-Seconds per gram of Anode/Cathode.
**Discharge Curve Analysis**: Includes ΔV/ΔT analysis at 5-second intervals, detection of constant ΔV/ΔT regions, and plateau detection with actionable outputs.
**Correlation Analysis**: Examines performance correlations (A·s/gram vs discharge slope) and duration correlations (duration vs total anode/cathode weight, calorific value).
**Password Protection**: Optional, per-comparison password protection uses Argon2id hashing with salting for security, storing only hashes.
**Edit Saved Data**: Allows loading saved comparisons for modification, with options to update existing or save as new, supporting password-protected entries.
**Battery Type Organization**: A selector at application start organizes all analysis and data by chosen battery chemistry.
**Unlimited Build Support**: The application supports 1-50 builds per comparison, dynamically adapting visualizations.
**Standard Performance Benchmarks**: Enables comparison against configurable target performance levels for metrics like Max On-Load Voltage, Max Open Circuit Voltage, Max Activation Time, and Min Duration.
**Detailed Reporting**: Generates comprehensive text, Excel, CSV, and PDF reports including test configuration, build info, performance metrics, statistical summaries, and benchmark comparisons.

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework.

### Data Processing & Analysis
- **Pandas**: Tabular data manipulation.
- **NumPy**: Numerical computing.
- **SciPy**: Scientific computing and statistical analysis.

### Visualization
- **Plotly**: Interactive charting (graph_objects and express).

### Database & ORM
- **SQLAlchemy**: Database abstraction layer.
- **PostgreSQL**: Relational database (optional, via `DATABASE_URL`).

### Python Standard Library
- **io**: In-memory file operations.
- **datetime**: Timestamp generation.
- **os**: Environment variable access.
- **json**: Data serialization.
## Recent Implementation (November 2025)

### Standard Parameters & Extended Metadata Persistence ✅
**Core Feature**: Complete save/load workflow for standard performance parameters and extended build metadata.

**Implementation**:
- Added `standard_params_json` column to `saved_comparisons` table
- Auto-extraction pipeline parses Excel metadata (basic info, standard benchmarks, extended build data)
- Form fields auto-populate from extracted/loaded data with visual confirmation
- Save persists all parameters to database; Load restores complete session state
- Unique build name generation prevents duplicate indices (auto-appends counter: "Build 1", "Build 1 (1)")
- `safe_scalar()` helper function handles pandas Series/scalar conversions

**Verification**: Database query confirms data persistence; save/load workflow tested end-to-end successfully.

**Migration Fix**: Loader-side deduplication automatically handles legacy comparisons with duplicate build names, ensuring backward compatibility. All saved comparisons (new and old) work reliably.
