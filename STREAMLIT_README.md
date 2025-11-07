# Demand Optimization - Streamlit Web Application

A modern web-based interface for the Demand Optimization system, replacing the tkinter GUI with a Streamlit application.

## Features

### 🎯 Complete Feature Parity with GUI.py

- **File Import**: Upload CSV or Excel files containing power generation data
- **Configuration Management**: Add, edit, delete, and view device configurations
- **Fixed Load Management**: Manage devices with predetermined power consumption
- **Optimization Execution**: Run genetic algorithm optimization with customizable parameters
- **Results Visualization**: Interactive charts showing monthly power profiles and optimized schedules

### ✨ Advantages over the tkinter GUI

- **Web-based**: Access from any browser, no desktop application installation needed
- **Modern UI**: Clean, intuitive interface with better user experience
- **Responsive Design**: Works on different screen sizes
- **Better State Management**: Streamlit's session state provides reliable data persistence
- **Easier Deployment**: Can be deployed to cloud platforms (Streamlit Cloud, Heroku, etc.)
- **Interactive Visualizations**: Better chart interactions with Streamlit's native support

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install streamlit pandas numpy matplotlib openpyxl
   ```

## Running the Application

### Local Development

Run the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Configuration Options

You can customize the Streamlit server with command-line options:

```bash
# Specify a different port
streamlit run streamlit_app.py --server.port 8080

# Run in headless mode (doesn't open browser)
streamlit run streamlit_app.py --server.headless true

# Enable CORS for remote access
streamlit run streamlit_app.py --server.enableCORS true
```

## Usage Guide

### 1. Import Data

1. Navigate to "Import Data" in the sidebar
2. Upload a CSV or Excel file containing hourly power generation data
3. Select the column containing power values
4. Preview the uploaded data

**Supported formats**: `.csv`, `.xlsx`, `.xls`

### 2. Manage Configurations

1. Go to "Manage Configurations"
2. Click "Add New Configuration" to expand the form
3. Fill in the device parameters:
   - **Device Name**: Identifier for the device (e.g., "Water Pumps")
   - **Start Hour**: First hour device can operate (0-23)
   - **End Hour**: Last hour device can operate (1-24)
   - **Power Capacity**: Power consumption per device in kW
   - **Devices per Hour**: Maximum devices that can run simultaneously
   - **Total Device-Hours**: Required device operating hours per day
4. Click "Add Configuration"

**Edit/Delete**: Use the buttons next to each configuration to modify or remove it

### 3. Manage Fixed Loads (Optional)

1. Go to "Manage Fixed Loads"
2. Click "Add New Fixed Load"
3. Enter device name and hourly power consumption for each of the 24 hours
4. Click "Add Fixed Load"

**Visualize**: Click the "Visualize" button to see the consumption profile
**Edit/Delete**: Modify or remove fixed loads as needed

### 4. Run Optimization

1. Navigate to "Run Optimization"
2. Check that prerequisites are met (data uploaded, configurations added)
3. Adjust algorithm parameters using sliders:
   - **Population Size**: Number of schedules per generation (20-500)
   - **Iterations**: Number of generations (20-500)
   - **Sample Runs**: Independent optimization attempts (20-500)
4. Click "Start Optimization"
5. Wait for the optimization to complete (progress bar shows status)

**Note**: Higher parameter values improve solution quality but increase computation time

### 5. View Results

1. Go to "View Results" after optimization completes
2. View two types of visualizations:
   - **Power Profiles**: Monthly view of power supply vs. demand
   - **Optimized Schedules**: Individual device schedules with statistics

**Features**:
- Select different devices from dropdown
- View peak usage, total hours, and power consumption
- Export detailed schedule tables

## Development Mode

To enable faster testing with reduced computation time:

1. Open `streamlit_app.py`
2. Change line 17: `DEVELOPMENT = True`
3. Save and restart the application

This sets default parameters to minimum values (20, 1, 1) instead of production values (200, 200, 200).

## Troubleshooting

### Port Already in Use

If port 8501 is already occupied:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Module Import Errors

Ensure you're running from the correct directory:
```bash
cd path/to/demand-optimisation
streamlit run streamlit_app.py
```

### File Upload Issues

- Check file format is CSV or Excel
- Ensure file has readable column headers
- Verify file is not corrupted

### Optimization Errors

- Verify at least one configuration is added
- Check configuration parameters are valid
- Ensure power data file is properly loaded

## Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy with one click

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t demand-optimization .
docker run -p 8501:8501 demand-optimization
```

## Comparison with Original GUI

| Feature | tkinter GUI | Streamlit App |
|---------|-------------|---------------|
| Platform | Desktop only | Web-based (any device) |
| Installation | Requires tkinter, customtkinter | pip install streamlit |
| Deployment | Standalone executable | Cloud deployment |
| UI Framework | CustomTkinter | Streamlit |
| State Management | Manual dict management | Session state |
| Visualization | Embedded matplotlib | Native Streamlit + matplotlib |
| Responsiveness | Fixed window size | Responsive layout |
| Accessibility | Desktop access only | Remote access via URL |

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the original `CLAUDE.md` for architecture details
3. Refer to [Streamlit documentation](https://docs.streamlit.io)

## License

Same license as the original demand-optimisation project.
