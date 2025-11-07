# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demand optimization project that uses genetic algorithms to optimize device scheduling based on power availability profiles. The goal is to minimize energy storage requirements by optimally scheduling flexible loads (devices) to match available power generation.

## Core Architecture

### Module Structure

The codebase is organized into focused Python modules:

- **base_package.py**: Core genetic algorithm functions (objective function, population creation, crossover, mutation, selection)
- **algo_package.py**: High-level optimization orchestration, power profile creation, and multiprocessing support
- **appli.py**: Application-level functions that coordinate multi-device optimization
- **GUI.py**: Complete tkinter/customtkinter GUI application for interactive use (legacy)
- **streamlit_app.py**: Modern web-based Streamlit application (recommended interface)
- **plot_package.py**: Matplotlib visualization functions for schedules and power profiles
- **test_package.py**: Utility functions for generating random test configs and fixeds

### Key Data Structures

**Config (Configuration)**: Represents a flexible device that can be scheduled
- Format: `[deb, fin, nb_device, total_devices, P_device, name]`
- `deb` (int): First hour device can run (0-23)
- `fin` (int): Last hour device can run (1-24)
- `nb_device` (int): Max devices per hour
- `total_devices` (int): Total device-hours per day
- `P_device` (float): Power capacity in kW
- `name` (str): Device name for visualization

**Fixed**: Represents a device with predetermined power consumption
- Format: `[schedule, name]`
- `schedule` (list): 24-element list with power consumption per hour
- `name` (str): Device name

**Power Profile**: Dictionary mapping time periods to hourly power availability
- Keys: Period numbers (1-12 for monthly, 1-365 for daily, etc.)
- Values: 24-element lists with power available per hour

**Schedule**: 24-element list representing number of devices running each hour

### Optimization Flow

1. **Power Profile Creation** (`algo_package.power_profile_creation`):
   - Imports time-series power data (CSV/XLSX)
   - Aggregates to hourly averages
   - Groups by time period (daily/weekly/monthly/yearly)

2. **Single Device Optimization** (`algo_package.algo`):
   - Uses genetic algorithm to find optimal schedule for one config
   - Accounts for existing fixed loads
   - Returns best schedule found

3. **Multi-Device Sequential Optimization** (`appli.final_2`):
   - Optimizes each config sequentially (highest capacity first recommended)
   - Each optimized schedule becomes a "fixed" for subsequent optimizations
   - Uses multiprocessing via `optimisation_mp`

4. **Objective Function** (`base_package.f`):
   - Calculates total energy storage required over all months
   - Penalizes times when demand exceeds available power
   - Formula: `sum over all hours of max(demand - power_available, 0)`

## Common Commands

### Running the Streamlit Web Application (Recommended)
```bash
streamlit run streamlit_app.py
```
This launches a modern web-based interface accessible at `http://localhost:8501`. See `STREAMLIT_README.md` for detailed usage instructions.

### Running the Legacy GUI Application
```bash
python GUI.py
```
Requires tkinter and customtkinter packages. The Streamlit app is recommended for better user experience.

### Development Mode
- **Streamlit**: Set `DEVELOPMENT = True` in streamlit_app.py (line 17)
- **tkinter GUI**: Set `development = True` in GUI.py (line 5)

This enables faster testing with reduced iterations.

### Running Benchmarks
```bash
python benchmark.py
# or
jupyter notebook benchmark.ipynb
```

### Testing with Random Data
```python
import test_package as test
configs = test.create_configs(3)  # 3 random configs
fixeds = test.create_fixeds(2)    # 2 random fixeds
```

## Important Implementation Details

### Genetic Algorithm Parameters

The algorithm uses these parameters:
- `len_pop`: Population size (default: 200 for production, 20 for development)
- `n_iter`: Number of generations (default: 200 for production, 1 for development)
- `n_sample`: Number of independent runs (default: 200 for production, 1 for development)

Each generation maintains the best individual and creates:
- 4 mutations of the best (using `p_mut=0.3`)
- Remainder filled via crossover of selected parents (using `p_crossover=0.9`)
- All children are constrained to meet total_devices requirement

### Constraint Handling

The `to_be_constrained` function ensures schedules meet the `total_devices` constraint by randomly adding/removing devices from valid hours until the sum equals `total_devices`. This is called after every crossover and mutation.

### Multiprocessing

Both `algo_package.optimisation_mp` and `appli.final` use `concurrent.futures.ProcessPoolExecutor` to parallelize optimization runs. The code reserves 2 CPUs (uses `os.cpu_count()-2` processes) to avoid overloading the system.

### GUI Architecture

#### Legacy tkinter GUI (GUI.py)
The GUI uses a component-based structure with CustomTkinter:
- `App`: Main window
- `Instance`: Root frame containing all application state
- `ImportFrame`: File selection
- `ConfigFixedFrame`: Container for config and fixed frames
- `ConfigFrame`/`FixedFrame`: Scrollable lists of configs/fixeds
- `ConfigRow`/`FixedRow`: Individual config/fixed display widgets
- `ConfigPage`/`FixedPage`: PopUp forms for adding/editing
- `CalculationSettings`: Parameter selection
- `ProfilePage`/`SchedulesPage`: Result visualization

State is stored in `Instance.configs` and `Instance.fixeds` as dictionaries indexed by creation order.

#### Streamlit Web App (streamlit_app.py)
The Streamlit app uses a page-based navigation structure:
- **Import Data**: File upload and preview
- **Manage Configurations**: CRUD operations for device configs
- **Manage Fixed Loads**: CRUD operations for fixed consumption
- **Run Optimization**: Parameter selection and execution
- **View Results**: Interactive visualization of results

State is managed via `st.session_state`:
- `configs`: Dictionary of configurations (indexed by counter)
- `fixeds`: Dictionary of fixed loads (indexed by counter)
- `filepath`: Path to uploaded power data file
- `power_profile`: Computed power profile after optimization
- `best_schedules`: Optimized schedules for each device
- `optimization_done`: Boolean flag indicating if optimization completed

The Streamlit app provides better state management, responsive design, and can be deployed to web servers for remote access.

## Data Files

- **config.txt**: Example configurations (format: "Nb of configs:Configs" on each line)
- **data/power_fixed.csv/xlsx**: Example power generation data
- **power_axis.csv**: Additional power data
- Various **data_*.txt** files: Saved optimization results

## Known Patterns

### Power Profile Types
- "daily": Each day has its own 24-hour profile
- "weekly": 7 profiles (one per day of week)
- "monthly": 12 profiles (one per month)
- "yearly": 1 profile (average across all days)

### File Import
The `importation` function in algo_package.py handles both CSV and XLSX files. Column selection can be by index (1-based) or name.

### Plotting
All plotting functions return `matplotlib.figure.Figure` objects when `GUI=True` to enable embedding in tkinter windows via `FigureCanvasTkAgg`.

## Development Notes

- The benchmark files compare different optimization strategies and parameter sets
- Line 178-180 in algo_package.py executes on import, creating a default power profile
- The GUI includes "FAST" buttons when development mode is enabled for quick testing with random data
