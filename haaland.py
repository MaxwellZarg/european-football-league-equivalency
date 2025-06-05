# Quick check script
import pandas as pd

# Load your results data
df = pd.read_csv('results/processed/european_combined.csv')

# Search for Haaland
haaland_data = df[df['player'].str.contains('Haaland', case=False, na=False)]
print(f"Haaland records found: {len(haaland_data)}")
print(haaland_data[['player', 'league', 'season', 'team', 'goals', 'minutes']])

# Check transfer labels
if 'will_transfer' in df.columns:
    haaland_transfers = haaland_data[haaland_data['will_transfer'] == 1]
    print(f"Haaland transfers found: {len(haaland_transfers)}")
