# ESB Smart Meter Insight

Upload your ESB Networks smart meter CSV files and get a clear electricity usage, cost, and supplier-accuracy dashboard.

This is Version 1 of a public-facing Streamlit app for Irish electricity users. It uses manual upload only. It does not ask for ESB login details and does not permanently store uploaded files.

## What The App Does

- Detects ESB HDF CSV file types automatically
- Builds daily and monthly usage summaries
- Estimates electricity costs from your tariff inputs
- Shows daily, monthly, hourly and heatmap charts
- Estimates baseload and highlights likely usage patterns
- Checks for missing intervals, duplicates and obvious data issues
- Compares ESB-derived usage with an optional supplier app figure
- Creates a copy/paste message for supplier support

## Supported ESB Files

The app can use one or more of these ESB Networks HDF CSV exports:

- 30-minute import readings in kWh
- 30-minute import readings in kW
- Daily total import kWh register
- Daily day/night/peak import kWh registers
- Export data, if available

The file names can be confusing. Upload the CSV files you downloaded and the app will inspect the read types inside them.

## How To Get Your ESB Data

1. Go to ESB Networks My Account.
2. Log in or create an account.
3. You may need your MPRN from your electricity bill.
4. Open **My energy consumption**.
5. Go to **Downloads**.
6. Download the available HDF CSV files.
7. Upload them into this app.

## Run Locally

From this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Privacy Notes

- Version 1 uses manual upload.
- The app does not ask for ESB usernames, passwords or login details.
- Uploaded files are read in memory for the current dashboard session.
- If you share screenshots or exported text, you can remove personal identifiers such as MPRN and meter serial number first.

## Demo Mode

If you do not have ESB files handy, choose **Try demo mode** in the app. Demo mode uses synthetic sample data, not a real household.

