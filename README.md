# Irish Energy Insights

Upload your ESB Networks smart meter CSV files and get a plain-English energy health check for your home.

This is a public Streamlit app for Irish electricity users. It uses manual CSV upload only: no ESB login details, usernames or passwords are requested.

## What The App Does

- Detects ESB HDF CSV file types automatically
- Creates an **Energy Health Check** score out of 100
- Explains whether usage looks low, typical or high
- Estimates overnight baseload, always-on watts and baseload cost
- Highlights morning, evening, weekend and overnight behaviour patterns
- Gives cautious appliance clues such as cooking, immersion, EV charging, laundry, storage heating and standby load
- Benchmarks usage against approximate Irish household ranges
- Forecasts monthly and annual bills from tariff inputs
- Ranks practical savings recommendations
- Checks supplier app kWh figures against ESB-derived usage
- Summarises data quality, missing intervals and confidence level

## Supported ESB Files

Upload one or more ESB Networks HDF CSV exports:

- 30-minute import readings in kWh
- 30-minute import readings in kW
- Daily total import kWh register
- Daily day/night/peak import kWh registers
- Export data, if available

The file names can be confusing. That is normal. The app inspects the read types inside the files and uses whatever data is available.

## How To Get Your ESB Data

1. Go to ESB Networks My Account.
2. Log in or create an account.
3. You may need your MPRN from your electricity bill.
4. Open **My energy consumption**.
5. Go to **Downloads**.
6. Download the available HDF CSV files.
7. Upload the CSV files into this app.

## Tariff Inputs

The app asks for:

- supplier name
- plan name
- unit rate in cent/kWh
- standing charge per year
- PSO levy per year
- export rate in cent/kWh, if relevant
- optional supplier app reported kWh

Use rates including VAT where possible. The app estimates bills from usage charge, standing charge, PSO levy and export credit. Real bills can also include discounts, previous balances, credits, VAT handling, estimated reads and corrections.

## Privacy Notes

- No ESB login details are required.
- CSV files are uploaded manually by the user.
- Files are processed for the current Streamlit session.
- The app does not intentionally save uploaded files.
- If you share screenshots or CSV files, consider removing personal identifiers such as MPRN, meter serial number and address details first.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Demo Mode

Choose **Try demo mode** in the app if you do not have ESB files handy. Demo mode uses synthetic sample data, not a real household.

## Important Limitations

The app provides estimates and clues, not certified engineering diagnostics. It cannot know exactly which appliance caused a spike. Appliance clues use cautious wording such as “may suggest”, “could be consistent with” and “worth checking”.
