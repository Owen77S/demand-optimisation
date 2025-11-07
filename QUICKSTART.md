# Quick Start Guide - Streamlit Web App

Get started with the Demand Optimization Streamlit app in 5 minutes!

## Installation

```bash
# Install dependencies
pip install streamlit pandas numpy matplotlib openpyxl

# Or use requirements file
pip install -r requirements.txt
```

## Launch the App

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 5-Minute Tutorial

### Step 1: Import Your Data (1 min)
1. Click "Import Data" in the sidebar
2. Upload your power data file (CSV or Excel)
3. Select the column with power values
4. ✅ Data loaded!

**Don't have data?** Use the example file `data/power_fixed.csv` in the project directory.

### Step 2: Add a Configuration (2 min)
1. Go to "Manage Configurations"
2. Click "Add New Configuration"
3. Fill in the form:
   ```
   Device Name: Water Pumps
   Start Hour: 7
   End Hour: 19
   Power Capacity: 12 kW
   Devices per Hour: 6
   Total Device-Hours: 18
   ```
4. Click "Add Configuration"
5. ✅ Configuration added!

**Add more configurations** for different devices (e.g., Grinders, HVAC systems)

### Step 3: (Optional) Add Fixed Loads (1 min)
1. Go to "Manage Fixed Loads"
2. Add any devices that run on fixed schedules
3. Enter hourly power consumption
4. Click "Add Fixed Load"

**Skip this step** if you don't have fixed loads.

### Step 4: Run Optimization (30 sec)
1. Go to "Run Optimization"
2. Check that prerequisites are met ✅
3. Adjust sliders if desired (or keep defaults)
4. Click "Start Optimization"
5. Wait for completion (1-3 minutes depending on parameters)

### Step 5: View Results (30 sec)
1. Go to "View Results"
2. Explore two tabs:
   - **Power Profiles**: See monthly demand vs. supply
   - **Optimized Schedules**: View individual device schedules
3. 🎉 Done!

## Example Workflow

```bash
# 1. Launch app
streamlit run streamlit_app.py

# 2. In browser:
#    - Upload data/power_fixed.csv
#    - Add 2-3 device configurations
#    - Run optimization with default parameters
#    - View results

# 3. Experiment:
#    - Try different parameter values
#    - Add fixed loads
#    - Compare different optimization runs
```

## Tips

- **Start small**: Use 1-2 configurations for your first run
- **Development mode**: Set `DEVELOPMENT = True` for faster testing
- **Save configurations**: Take screenshots or note down successful configs
- **Parameter tuning**: Higher values = better results but slower
- **Remote access**: Deploy to Streamlit Cloud for team access

## Common Issues

**Port already in use?**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Module not found?**
```bash
# Make sure you're in the right directory
cd demand-optimisation
pip install -r requirements.txt
```

**Optimization too slow?**
- Enable development mode
- Reduce parameter values
- Start with fewer configurations

## Next Steps

- Read `STREAMLIT_README.md` for complete documentation
- Check `CLAUDE.md` for architecture details
- Try different power profiles (daily, weekly, yearly)
- Deploy to cloud for production use

## Need Help?

1. Check the troubleshooting section in `STREAMLIT_README.md`
2. Review error messages in the app
3. Verify all prerequisites are met

Happy optimizing! ⚡
