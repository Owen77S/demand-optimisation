"""
Streamlit web application for Demand Optimization
Replaces the tkinter GUI with a modern web interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import appli
import plot_package as plot
import time
from estimator_function import estimate_calculation_time

# Configure Streamlit page
st.set_page_config(
    page_title="Demand Optimization",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card-like containers */
    .stContainer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Better button styling */
    .stButton>button {
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }

    /* Improved expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        font-weight: 600;
    }

    /* Better metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Improve spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    .status-success {
        background-color: #d4edda;
        color: #155724;
    }

    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }

    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Development mode (set to False for production)
DEVELOPMENT = False

# Initialize session state
if 'configs' not in st.session_state:
    st.session_state.configs = {}
    st.session_state.config_counter = 0

if 'fixeds' not in st.session_state:
    st.session_state.fixeds = {}
    st.session_state.fixed_counter = 0

if 'filepath' not in st.session_state:
    st.session_state.filepath = None

if 'power_profile' not in st.session_state:
    st.session_state.power_profile = None

if 'best_schedules' not in st.session_state:
    st.session_state.best_schedules = None

if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False

# Helper functions for validation
def validate_config(deb, fin, nb_devices, total_devices):
    """Validate configuration parameters"""
    errors = []

    if deb >= fin:
        errors.append("Beginning hour must be less than end hour")

    if deb < 0 or deb > 24 or fin < 0 or fin > 24:
        errors.append("Hours must be between 0 and 24")

    if (fin - deb) * nb_devices < total_devices:
        errors.append("Total devices cannot be reached with current settings. Consider: increasing time span, increasing devices per hour, or decreasing total devices")

    return errors

def validate_fixed(schedule):
    """Validate fixed schedule"""
    errors = []

    try:
        schedule_values = [float(val) for val in schedule]
        if any(val < 0 for val in schedule_values):
            errors.append("All power values must be positive or zero")
    except ValueError:
        errors.append("All values must be valid numbers")

    return errors, schedule_values if not errors else None

# Main title with improved styling
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    st.markdown('<h1 class="main-title">Demand Optimization System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Optimize device scheduling based on power availability using genetic algorithms</p>', unsafe_allow_html=True)

# Sidebar for navigation with status indicators
st.sidebar.title("Navigation")

# Check status for each page
data_status = "✅" if st.session_state.filepath else "⭕"
config_status = "✅" if len(st.session_state.configs) > 0 else "⭕"
fixed_status = "✅" if len(st.session_state.fixeds) > 0 else "⭕"
results_status = "✅" if st.session_state.optimization_done else "⭕"

# Navigation with status
page_options = [
    f"{data_status} Import Data",
    f"{config_status} Manage Configurations",
    f"{fixed_status} Manage Fixed Loads",
    f"{results_status} Run Optimization",
    f"{results_status} View Results"
]

selected = st.sidebar.radio("Go to", page_options, label_visibility="collapsed")

# Extract the actual page name
page = selected.split(" ", 1)[1] if " " in selected else selected

# Add workflow progress indicator
st.sidebar.divider()
st.sidebar.subheader("Workflow Progress")

progress_steps = [
    ("Import Data", st.session_state.filepath is not None),
    ("Add Configurations", len(st.session_state.configs) > 0),
    ("Run Optimization", st.session_state.optimization_done)
]

for step_name, is_complete in progress_steps:
    if is_complete:
        st.sidebar.markdown(f"✅ {step_name}")
    else:
        st.sidebar.markdown(f"⭕ {step_name}")

# Show summary stats in sidebar
if st.session_state.configs or st.session_state.fixeds:
    st.sidebar.divider()
    st.sidebar.subheader("Quick Stats")

    if st.session_state.configs:
        st.sidebar.metric("Configurations", len(st.session_state.configs))

    if st.session_state.fixeds:
        st.sidebar.metric("Fixed Loads", len(st.session_state.fixeds))

    if st.session_state.optimization_done:
        st.sidebar.success("✅ Optimization Complete")

# ====================
# PAGE 1: IMPORT DATA
# ====================
if page == "Import Data":
    st.header("Import Power Data")

    # Show welcome message if this is first time
    if not st.session_state.filepath and len(st.session_state.configs) == 0:
        st.info("**Welcome to Demand Optimization System!** Start by loading power generation data below.")

    st.markdown("""
    Upload a CSV or Excel file containing power generation data, or use the default sample data.
    The file should have hourly power values with timestamps.
    """)

    # Quick start section with default data button
    with st.container():
        st.subheader("Quick Start")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("Load Default Power Data (data/power_fixed.csv)", type="primary", use_container_width=True):
                import os
                default_filepath = os.path.join(os.path.dirname(__file__), "data/power_fixed.csv")

                if os.path.exists(default_filepath):
                    st.session_state.filepath = default_filepath
                    st.session_state.column_name = "System power generated | (kW)"
                    st.success("✅ Default power data loaded successfully!")
                    st.info("Loaded: data/power_fixed.csv with power generation data")
                else:
                    st.error("Default file not found at: {default_filepath}")

    st.divider()
    st.subheader("Or Upload Your Own Data")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: .csv, .xlsx, .xls"
    )

    if uploaded_file is not None:
        # Save file temporarily
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.filepath = tmp_file.name

        # Show file preview
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

        # Load and preview data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(st.session_state.filepath)
            else:
                df = pd.read_excel(st.session_state.filepath)

            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Column selection
            col1, col2 = st.columns(2)

            with col1:
                column_name = st.selectbox(
                    "Select power data column",
                    options=df.columns.tolist(),
                    help="Choose the column containing power generation values"
                )

            with col2:
                st.metric("Selected Column", column_name)
                st.metric("Data Points", len(df))

            st.session_state.column_name = column_name

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    elif st.session_state.filepath:
        st.success("✅ Data File Currently Loaded")
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"File: {st.session_state.filepath}")
                if hasattr(st.session_state, 'column_name'):
                    st.info(f"Column: {st.session_state.column_name}")
            with col2:
                if st.button("Clear & Upload New", use_container_width=True):
                    st.session_state.filepath = None
                    st.session_state.column_name = None
                    st.rerun()

        # Display power profile visualization
        st.divider()

# ====================
# PAGE 2: CONFIGURATIONS
# ====================
elif page == "Manage Configurations":
    st.header("Device Configurations")

    st.markdown("""
    Add flexible devices that can be scheduled. Each configuration defines:
    - Time window for operation
    - Number of devices available per hour
    - Total device-hours required per day
    - Power capacity per device
    """)

    # Quick start section with default configurations button
    if len(st.session_state.configs) == 0:
        with st.container():
            st.subheader("Quick Start")
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("Load Default Configurations", type="primary", use_container_width=True):
                    # Add default config 1: Water pumps
                    st.session_state.config_counter += 1
                    config1 = [7, 19, 6, 18, 12, 'Water pumps']
                    st.session_state.configs[st.session_state.config_counter] = config1

                    # Add default config 2: Grinders
                    st.session_state.config_counter += 1
                    config2 = [7, 19, 6, 60, 15, 'Grinders']
                    st.session_state.configs[st.session_state.config_counter] = config2

                    st.success("✅ Default configurations loaded successfully!")
                    st.info("Added: Water pumps (12 kW) and Grinders (15 kW)")
                    st.rerun()

        st.divider()

    # Add new configuration
    with st.expander("Add New Configuration", expanded=len(st.session_state.configs) == 0):
        with st.form("new_config_form"):
            st.markdown("**Define Device Parameters**")

            col1, col2 = st.columns(2)

            with col1:
                name = st.text_input(
                    "Device Name",
                    placeholder="e.g., Water Pumps",
                    help="A descriptive name for this device type"
                )
                deb = st.number_input(
                    "Start Hour (0-23)",
                    min_value=0,
                    max_value=23,
                    value=7,
                    help="Earliest hour the device can operate (24-hour format)"
                )
                fin = st.number_input(
                    "End Hour (1-24)",
                    min_value=1,
                    max_value=24,
                    value=19,
                    help="Latest hour the device can operate (24-hour format)"
                )

            with col2:
                P_device = st.number_input(
                    "Power Capacity (kW)",
                    min_value=0.1,
                    value=12.0,
                    step=0.1,
                    help="Power consumption per device in kilowatts"
                )
                nb_devices = st.number_input(
                    "Max Devices per Hour",
                    min_value=1,
                    value=6,
                    step=1,
                    help="Maximum number of devices that can run simultaneously"
                )
                total_devices = st.number_input(
                    "Total Device-Hours per Day",
                    min_value=1,
                    value=18,
                    step=1,
                    help="Total device operation time required (e.g., 3 devices × 6 hours = 18)"
                )

            # Show constraint validation info
            if fin > deb:
                max_possible = (fin - deb) * nb_devices
                st.info(f"With current settings, maximum possible device-hours: **{max_possible}**")

            submitted = st.form_submit_button("✅ Add Configuration", use_container_width=True, type="primary")

            if submitted:
                errors = validate_config(deb, fin, nb_devices, total_devices)

                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state.config_counter += 1
                    config = [deb, fin, nb_devices, total_devices, P_device, name]
                    st.session_state.configs[st.session_state.config_counter] = config
                    st.success(f"✅ Configuration '{name}' added successfully!")
                    st.rerun()

    # Display existing configurations
    if st.session_state.configs:
        st.subheader(f"Current Configurations ({len(st.session_state.configs)})")

        for config_id, config in st.session_state.configs.items():
            deb, fin, nb_devices, total_devices, P_device, name = config

            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"### {name}")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Time Window", f"{deb}:00 - {fin}:00", delta=f"{fin-deb} hours")
                    with col_b:
                        st.metric("Power", f"{P_device} kW")
                    with col_c:
                        st.metric("Devices/Hour", f"{nb_devices}")
                    with col_d:
                        st.metric("Total Hours", f"{total_devices}")

                with col2:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("Edit", key=f"edit_{config_id}", use_container_width=True):
                        st.session_state[f'editing_{config_id}'] = True
                        st.rerun()

                with col3:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("Delete", key=f"delete_{config_id}", use_container_width=True):
                        del st.session_state.configs[config_id]
                        st.success(f"Deleted '{name}'")
                        st.rerun()

                # Edit form
                if st.session_state.get(f'editing_{config_id}', False):
                    with st.form(f"edit_config_form_{config_id}"):
                        st.markdown("**Edit Configuration**")

                        col1, col2 = st.columns(2)

                        with col1:
                            new_name = st.text_input("Device Name", value=name)
                            new_deb = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=deb)
                            new_fin = st.number_input("End Hour (1-24)", min_value=1, max_value=24, value=fin)

                        with col2:
                            new_P_device = st.number_input("Power Capacity (kW)", min_value=0.1, value=float(P_device), step=0.1)
                            new_nb_devices = st.number_input("Devices per Hour", min_value=1, value=nb_devices, step=1)
                            new_total_devices = st.number_input("Total Device-Hours per Day", min_value=1, value=total_devices, step=1)

                        col_save, col_cancel = st.columns(2)

                        with col_save:
                            save_edit = st.form_submit_button("Save Changes")

                        with col_cancel:
                            cancel_edit = st.form_submit_button("Cancel")

                        if save_edit:
                            errors = validate_config(new_deb, new_fin, new_nb_devices, new_total_devices)

                            if errors:
                                for error in errors:
                                    st.error(error)
                            else:
                                st.session_state.configs[config_id] = [new_deb, new_fin, new_nb_devices, new_total_devices, new_P_device, new_name]
                                st.session_state[f'editing_{config_id}'] = False
                                st.success("Changes saved!")
                                st.rerun()

                        if cancel_edit:
                            st.session_state[f'editing_{config_id}'] = False
                            st.rerun()

                st.divider()
    else:
        st.info("No configurations added yet. Add your first configuration above!")

# ====================
# PAGE 3: FIXED LOADS
# ====================
elif page == "Manage Fixed Loads":
    st.header("Fixed Power Consumption")

    st.markdown("""
    Add devices with predetermined hourly power consumption that cannot be rescheduled.
    """)

    # Add new fixed
    with st.expander("Add New Fixed Load", expanded=len(st.session_state.fixeds) == 0):
        with st.form("new_fixed_form"):
            name = st.text_input("Device Name", placeholder="e.g., HVAC System")

            st.markdown("**Hourly Power Consumption (kW)**")

            # Create 4 rows of 6 columns for 24 hours
            schedule_inputs = []
            for row in range(4):
                cols = st.columns(6)
                for col_idx, col in enumerate(cols):
                    hour = row * 6 + col_idx
                    with col:
                        val = st.number_input(
                            f"{hour}:00-{hour+1}:00",
                            min_value=0.0,
                            value=0.0,
                            step=0.1,
                            key=f"fixed_new_{hour}",
                            label_visibility="visible"
                        )
                        schedule_inputs.append(val)

            submitted = st.form_submit_button("Add Fixed Load")

            if submitted:
                errors, schedule_values = validate_fixed(schedule_inputs)

                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state.fixed_counter += 1
                    fixed = [schedule_values, name if name else "Unnamed Device"]
                    st.session_state.fixeds[st.session_state.fixed_counter] = fixed
                    st.success(f"✅ Fixed load '{name}' added successfully!")
                    st.rerun()

    # Display existing fixed loads
    if st.session_state.fixeds:
        st.subheader(f"Current Fixed Loads ({len(st.session_state.fixeds)})")

        for fixed_id, fixed in st.session_state.fixeds.items():
            schedule, name = fixed

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"### {name}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Consumption", f"{sum(schedule):.2f} kWh")
                    with col_b:
                        st.metric("Peak Power", f"{max(schedule):.2f} kW")
                    with col_c:
                        st.metric("Average Power", f"{sum(schedule)/24:.2f} kW")

                with col2:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("View", key=f"viz_{fixed_id}", use_container_width=True):
                        st.session_state[f'visualizing_{fixed_id}'] = not st.session_state.get(f'visualizing_{fixed_id}', False)
                        st.rerun()

                with col3:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("Edit", key=f"edit_fixed_{fixed_id}", use_container_width=True):
                        st.session_state[f'editing_fixed_{fixed_id}'] = True
                        st.rerun()

                with col4:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if st.button("Delete", key=f"delete_fixed_{fixed_id}", use_container_width=True):
                        del st.session_state.fixeds[fixed_id]
                        st.success(f"Deleted '{name}'")
                        st.rerun()

                # Visualization
                if st.session_state.get(f'visualizing_{fixed_id}', False):
                    fig = plot.show_fixed(fixed)
                    st.pyplot(fig)

                # Edit form
                if st.session_state.get(f'editing_fixed_{fixed_id}', False):
                    with st.form(f"edit_fixed_form_{fixed_id}"):
                        st.markdown("**Edit Fixed Load**")

                        new_name = st.text_input("Device Name", value=name)

                        st.markdown("**Hourly Power Consumption (kW)**")

                        new_schedule_inputs = []
                        for row in range(4):
                            cols = st.columns(6)
                            for col_idx, col in enumerate(cols):
                                hour = row * 6 + col_idx
                                with col:
                                    val = st.number_input(
                                        f"{hour}:00-{hour+1}:00",
                                        min_value=0.0,
                                        value=float(schedule[hour]),
                                        step=0.1,
                                        key=f"fixed_edit_{fixed_id}_{hour}",
                                        label_visibility="visible"
                                    )
                                    new_schedule_inputs.append(val)

                        col_save, col_cancel = st.columns(2)

                        with col_save:
                            save_edit = st.form_submit_button("Save Changes")

                        with col_cancel:
                            cancel_edit = st.form_submit_button("Cancel")

                        if save_edit:
                            errors, schedule_values = validate_fixed(new_schedule_inputs)

                            if errors:
                                for error in errors:
                                    st.error(error)
                            else:
                                st.session_state.fixeds[fixed_id] = [schedule_values, new_name]
                                st.session_state[f'editing_fixed_{fixed_id}'] = False
                                st.success("Changes saved!")
                                st.rerun()

                        if cancel_edit:
                            st.session_state[f'editing_fixed_{fixed_id}'] = False
                            st.rerun()

                st.divider()
    else:
        st.info("No fixed loads added yet. Add your first fixed load above!")

# ====================
# PAGE 4: RUN OPTIMIZATION
# ====================
elif page == "Run Optimization":
    st.header("Run Optimization")

    # Check prerequisites with better visual status
    ready = True
    issues = []

    if not st.session_state.filepath:
        ready = False
        issues.append(("error", "No power data file uploaded", "Go to 'Import Data' to upload or load default data"))
    else:
        issues.append(("success", "Power data file uploaded", f"File: {st.session_state.filepath}"))

    if len(st.session_state.configs) == 0:
        ready = False
        issues.append(("error", "No device configurations added", "Go to 'Manage Configurations' to add devices"))
    else:
        issues.append(("success", f"{len(st.session_state.configs)} device configuration(s) added", ""))

    if len(st.session_state.fixeds) > 0:
        total_fixed = sum([sum(fixed[0]) for fixed in st.session_state.fixeds.values()])
        issues.append(("info", f"{len(st.session_state.fixeds)} fixed load(s) added", f"Total consumption: {total_fixed:.1f} kWh/day"))
    else:
        issues.append(("info", "No fixed loads added (optional)", "Fixed loads are not required"))

    # Display status with cards
    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    for idx, (status_type, title, description) in enumerate(issues):
        with [col1, col2, col3][idx]:
            if status_type == "error":
                st.error(f"**{title}**")
                st.caption(description)
            elif status_type == "success":
                st.success(f"**{title}**")
                st.caption(description)
            else:
                st.info(f"**{title}**")
                st.caption(description)

    st.divider()

    if ready:
        st.subheader("Algorithm Parameters")

        st.markdown("""
        Configure the genetic algorithm parameters. Higher values improve solution quality but increase computation time.
        """)

        # Parameter presets
        col_preset1, col_preset2, col_preset3 = st.columns(3)

        with col_preset1:
            if st.button("Fast (Testing)", use_container_width=True):
                st.session_state.len_pop = 10
                st.session_state.n_iter = 10
                st.session_state.n_sample = 10

        with col_preset2:
            if st.button("Balanced (Recommended)", use_container_width=True, type="primary"):
                st.session_state.len_pop = 100
                st.session_state.n_iter = 100
                st.session_state.n_sample = 100

        with col_preset3:
            if st.button("High Quality (Slow)", use_container_width=True):
                st.session_state.len_pop = 200
                st.session_state.n_iter = 200
                st.session_state.n_sample = 200

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if DEVELOPMENT:
                len_pop = st.slider("Population Size", min_value=10, max_value=500,
                                   value=st.session_state.get('len_pop', 10), step=10)
            else:
                len_pop = st.slider("Population Size", min_value=10, max_value=500,
                                   value=st.session_state.get('len_pop', 100), step=10)
            st.caption("Number of schedules in each generation")

        with col2:
            if DEVELOPMENT:
                n_iter = st.slider("Iterations", min_value=1, max_value=500,
                                  value=st.session_state.get('n_iter', 1), step=10)
            else:
                n_iter = st.slider("Iterations", min_value=10, max_value=500,
                                  value=st.session_state.get('n_iter', 100), step=10)
            st.caption("Number of generations to evolve")

        with col3:
            if DEVELOPMENT:
                n_sample = st.slider("Sample Runs", min_value=1, max_value=500,
                                    value=st.session_state.get('n_sample', 1), step=10)
            else:
                n_sample = st.slider("Sample Runs", min_value=10, max_value=500,
                                    value=st.session_state.get('n_sample', 100), step=10)
            st.caption("Independent optimization runs")

        # Estimated computation time with ML estimator
        nb_configs = len(st.session_state.configs)
        nb_fixeds = len(st.session_state.fixeds)
        estimated_time = estimate_calculation_time(nb_configs, nb_fixeds, len_pop, n_iter, n_sample)

        col_time1, col_time2, col_time3 = st.columns(3)
        with col_time1:
            minutes = estimated_time / 60
            if minutes < 1:
                time_display = f"{estimated_time:.1f}s"
            else:
                time_display = f"{minutes:.1f}min ({estimated_time:.0f}s)"
            st.metric("Estimated Time", time_display, help="ML-based prediction (R²=96%)")
        with col_time2:
            complexity = len_pop * n_iter * n_sample
            st.metric("Complexity Score", f"{complexity:,}", help="Higher = better quality but slower")
        with col_time3:
            st.metric("Devices", f"{nb_configs} configs + {nb_fixeds} fixeds")

        st.divider()

        # Run optimization button
        if st.button("Start Optimization", type="primary", use_container_width=True):

            # Prepare data
            configs = list(st.session_state.configs.values())
            fixeds = list(st.session_state.fixeds.values())

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            try:
                start_time = time.time()

                status_text.text("Loading power profile...")
                progress_bar.progress(5)

                # Get column name
                column = st.session_state.column_name if hasattr(st.session_state, 'column_name') else 2

                # Run optimization
                status_text.text("Running optimization... This may take a few minutes.")
                time_text.text(f"Estimated remaining time: {estimated_time:.0f}s")
                progress_bar.progress(10)

                # Start optimization (we'll update progress based on estimated time)
                optimization_start = time.time()

                power_profile, best_schedules = appli.calculation(
                    st.session_state.filepath,
                    column,
                    "monthly",
                    configs,
                    fixeds,
                    DEVELOPMENT,
                    [len_pop, n_iter, n_sample]
                )

                # Calculate actual elapsed time
                elapsed_time = time.time() - optimization_start

                progress_bar.progress(90)
                status_text.text("Generating results...")
                time_text.text(f"Actual calculation time: {elapsed_time:.1f}s")

                # Store results
                st.session_state.power_profile = power_profile
                st.session_state.best_schedules = best_schedules
                st.session_state.optimization_done = True
                st.session_state.optimization_params = [len_pop, n_iter, n_sample]

                progress_bar.progress(100)
                total_time = time.time() - start_time
                status_text.text("✅ Optimization completed successfully!")
                time_text.text(f"Total time: {total_time:.1f}s | Calculation: {elapsed_time:.1f}s | Estimated: {estimated_time:.1f}s")

                st.success("Optimization completed! Go to 'View Results' to see the optimized schedules.")
                st.balloons()

            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.exception(e)
    else:
        st.warning("Please complete all prerequisites before running optimization.")

# ====================
# PAGE 5: VIEW RESULTS
# ====================
elif page == "View Results":
    st.header("Optimization Results")

    if not st.session_state.optimization_done:
        # Enhanced empty state with call-to-action
        st.warning("No optimization results available yet.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("Complete these steps to see results:")
            st.markdown("""
            1. ✅ Import power data
            2. ✅ Add device configurations
            3. ✅ Run optimization
            """)

            if st.button("Go to Run Optimization", type="primary", use_container_width=True):
                st.session_state.force_page = "Run Optimization"
                st.rerun()

    else:
        # Display optimization parameters with better layout
        with st.expander("Optimization Parameters Used", expanded=False):
            params = st.session_state.optimization_params
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Population Size", params[0])
            with col2:
                st.metric("Iterations", params[1])
            with col3:
                st.metric("Sample Runs", params[2])
            with col4:
                complexity = params[0] * params[1] * params[2]
                st.metric("Complexity Score", f"{complexity:,}")

        # Summary metrics
        st.subheader("Optimization Summary")
        col1, col2, col3, col4 = st.columns(4)

        configs = list(st.session_state.configs.values())
        total_capacity = sum([config[4] for config in configs])
        total_device_hours = sum([config[3] for config in configs])

        with col1:
            st.metric("Devices Optimized", len(configs))
        with col2:
            st.metric("Total Capacity", f"{total_capacity:.1f} kW")
        with col3:
            st.metric("Total Device-Hours", total_device_hours)
        with col4:
            if st.session_state.fixeds:
                total_fixed = sum([sum(fixed[0]) for fixed in st.session_state.fixeds.values()])
                st.metric("Fixed Load", f"{total_fixed:.1f} kWh/day")
            else:
                st.metric("Fixed Load", "None")

        st.divider()

        # Tabs for different visualizations
        tab1, tab2 = st.tabs(["Power Profiles", "Device Schedules"])

        with tab1:
            st.subheader("Monthly Power Profiles")
            st.markdown("Red line shows available power. Stacked areas show device consumption (fixed + optimized).")

            configs = list(st.session_state.configs.values())
            fixeds = list(st.session_state.fixeds.values())

            try:
                fig = plot.plot_profile(
                    st.session_state.power_profile,
                    st.session_state.best_schedules,
                    configs,
                    fixeds,
                    GUI=True
                )
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating power profile visualization: {str(e)}")

        with tab2:
            st.subheader("Optimized Device Schedules")

            configs = list(st.session_state.configs.values())

            # Sort configs by capacity (descending)
            sorted_configs = sorted(configs, key=lambda x: x[4], reverse=True)

            # Device selector with tabs for each device
            device_names = [config[5] for config in sorted_configs]

            # Create tabs for each device
            device_tabs = st.tabs(device_names)

            for tab_idx, device_tab in enumerate(device_tabs):
                with device_tab:
                    # Display schedule
                    schedule = st.session_state.best_schedules[tab_idx]
                    config = sorted_configs[tab_idx]

                    # Display statistics first
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Peak Usage", f"{max(schedule)} devices")

                    with col2:
                        st.metric("Total Device-Hours", f"{sum(schedule)} hours")

                    with col3:
                        st.metric("Power Capacity", f"{config[4]} kW")

                    with col4:
                        peak_power = max(schedule) * config[4]
                        st.metric("Peak Power", f"{peak_power:.1f} kW")

                    st.divider()

                    # Create visualization with better styling
                    fig = Figure(figsize=(12, 5), dpi=100)
                    ax = fig.add_subplot()

                    # Enhanced bar chart
                    hours = range(24)
                    bars = ax.bar(hours, schedule, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1.5)

                    # Highlight peak hours
                    max_val = max(schedule)
                    for i, bar in enumerate(bars):
                        if schedule[i] == max_val:
                            bar.set_color('orangered')

                    ax.set_xlabel("Hour of Day", fontsize=12, fontweight='bold')
                    ax.set_ylabel(f"Number of {config[5]} Running", fontsize=12, fontweight='bold')
                    ax.set_title(f"Optimized Daily Schedule for {config[5]}", fontsize=14, fontweight='bold', pad=20)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_xlim(-0.5, 23.5)
                    ax.set_xticks(range(0, 24, 2))
                    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], rotation=45)

                    # Add value labels on bars
                    for i, v in enumerate(schedule):
                        if v > 0:
                            ax.text(i, v + max_val*0.02, str(int(v)), ha='center', va='bottom', fontsize=9)

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Show schedule table for this device
                    with st.expander("View Detailed Schedule Table"):
                        schedule_df = pd.DataFrame({
                            'Hour': [f"{h}:00 - {h+1}:00" for h in range(24)],
                            'Devices Running': schedule,
                            'Power (kW)': [s * config[4] for s in schedule]
                        })
                        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

                        # Add export button
                        csv = schedule_df.to_csv(index=False)
                        st.download_button(
                            label="Download Schedule as CSV",
                            data=csv,
                            file_name=f"schedule_{config[5].replace(' ', '_')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

# Enhanced Footer
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center;'>
    <h4 style='color: #1f77b4; margin-bottom: 0.5rem;'>Demand Optimization System</h4>
    <p style='color: #666; font-size: 0.85rem; margin-bottom: 0.25rem;'>Version 2.0</p>
    <p style='color: #999; font-size: 0.75rem;'>Powered by Genetic Algorithms</p>
    <p style='color: #999; font-size: 0.75rem;'>Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
