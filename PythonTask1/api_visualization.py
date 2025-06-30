import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('WEATHERBIT_API_KEY')

api_url = f"https://api.weatherbit.io/v2.0/current?lat=35.7796&lon=-78.6382&key={api_key}&include=minutely"

response = requests.get(api_url)
data = response.json()

api_response_minutely = data['minutely']

api_response_data = data["data"][0]
city_name = api_response_data['city_name']

df_minutely = pd.DataFrame(api_response_minutely)

df_minutely['timestamp_local'] = pd.to_datetime(df_minutely['timestamp_local'])
df_minutely['timestamp_utc'] = pd.to_datetime(df_minutely['timestamp_utc'])

df_minutely = df_minutely.set_index('timestamp_local')

print("Processed DataFrame Head:")
print(df_minutely.head())

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_minutely, x=df_minutely.index, y='temp')
plt.title(f'Minute-by-Minute Temperature in {city_name}')
plt.xlabel('Time (Local)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

if os.file.exists('visualizationBoard.png'):
    os.remove('visualizationBoard.png')

plt.savefig('visualizationBoard.png')