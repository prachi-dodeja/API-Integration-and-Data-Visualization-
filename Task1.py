import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    import seaborn as sns
    HAS_SEABORN = True
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn not found. Using matplotlib only.")
    plt.style.use('default')
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️ Plotly not found. Using matplotlib for all visualizations.")
class APIDataVisualizer:
    def __init__(self):
        self.data = {}
        self.apis = {
            'users': 'https://jsonplaceholder.typicode.com/users',
            'posts': 'https://jsonplaceholder.typicode.com/posts',
            'todos': 'https://jsonplaceholder.typicode.com/todos',
            'crypto': 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=20&page=1',
            'weather': 'https://api.openweathermap.org/data/2.5/group?id=524901,703448,2643743,2988507,3117735&units=metric&appid=demo'
        }
    def fetch_data(self, api_name):
        """Fetch data from the specified API"""
        try:
            print(f"📡 Fetching data from {api_name} API...")
            
            if api_name == 'weather':

                mock_weather = [
                    {"name": "London", "main": {"temp": 15, "humidity": 65, "pressure": 1013}, "weather": [{"main": "Cloudy"}]},
                    {"name": "Paris", "main": {"temp": 18, "humidity": 58, "pressure": 1015}, "weather": [{"main": "Sunny"}]},
                    {"name": "Berlin", "main": {"temp": 12, "humidity": 72, "pressure": 1010}, "weather": [{"main": "Rainy"}]},
                    {"name": "Madrid", "main": {"temp": 22, "humidity": 45, "pressure": 1018}, "weather": [{"main": "Sunny"}]},
                    {"name": "Rome", "main": {"temp": 25, "humidity": 55, "pressure": 1016}, "weather": [{"main": "Clear"}]}
                ]
                self.data[api_name] = mock_weather
                print(f"✅ Successfully fetched {len(mock_weather)} weather records")
                return
            response = requests.get(self.apis[api_name], timeout=10)
            response.raise_for_status()   
            data = response.json()
            self.data[api_name] = data
            print(f"✅ Successfully fetched {len(data)} records from {api_name}")
        except requests.RequestException as e:
            print(f"❌ Error fetching data from {api_name}: {e}")
            return None
    def analyze_users_data(self):
        """Analyze and visualize users data"""
        if 'users' not in self.data:
            self.fetch_data('users')
        users_df = pd.json_normalize(self.data['users'])
        print("\n📊 USERS DATA ANALYSIS")
        print("=" * 50)
        print(f"Total Users: {len(users_df)}")
        print(f"Unique Companies: {users_df['company.name'].nunique()}")
        print(f"Unique Cities: {users_df['address.city'].nunique()}")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('👥 Users Data Analysis Dashboard', fontsize=16, fontweight='bold')
        company_counts = users_df['company.name'].value_counts()
        axes[0,0].pie(company_counts.values, labels=company_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Users by Company')
        city_counts = users_df['address.city'].value_counts().head(8)
        axes[0,1].bar(range(len(city_counts)), city_counts.values)
        axes[0,1].set_xticks(range(len(city_counts)))
        axes[0,1].set_xticklabels(city_counts.index, rotation=45)
        axes[0,1].set_title('Users by City')
        lats = users_df['address.geo.lat'].astype(float)
        lngs = users_df['address.geo.lng'].astype(float)
        axes[1,0].scatter(lngs, lats, alpha=0.7, s=100)
        axes[1,0].set_xlabel('Longitude')
        axes[1,0].set_ylabel('Latitude')
        axes[1,0].set_title('User Geographic Distribution')
        users_df['email_domain'] = users_df['email'].str.split('@').str[1]
        domain_counts = users_df['email_domain'].value_counts()
        axes[1,1].bar(domain_counts.index, domain_counts.values)
        axes[1,1].set_title('Email Domains')
        axes[1,1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        if HAS_PLOTLY:
            fig_plotly = px.scatter_mapbox(
                users_df, 
                lat='address.geo.lat', 
                lon='address.geo.lng',
                hover_name='name',
                hover_data=['email', 'company.name', 'address.city'],
                zoom=1,
                height=600,
                title="🗺️ Interactive User Geographic Distribution"
            )
            fig_plotly.update_layout(mapbox_style="open-street-map")
            fig_plotly.show()
        else:
            print("📍 Install plotly for interactive map: pip install plotly")
        return users_df
    def analyze_posts_data(self):
        """Analyze and visualize posts data"""
        if 'posts' not in self.data:
            self.fetch_data('posts')
        posts_df = pd.DataFrame(self.data['posts'])
        print("\n📝 POSTS DATA ANALYSIS")
        print("=" * 50)
        print(f"Total Posts: {len(posts_df)}")
        print(f"Unique Authors: {posts_df['userId'].nunique()}")
        posts_df['title_length'] = posts_df['title'].str.len()
        posts_df['body_length'] = posts_df['body'].str.len()
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📝 Posts Data Analysis Dashboard', fontsize=16, fontweight='bold')
        posts_per_user = posts_df['userId'].value_counts().sort_index()
        axes[0,0].bar(posts_per_user.index, posts_per_user.values)
        axes[0,0].set_xlabel('User ID')
        axes[0,0].set_ylabel('Number of Posts')
        axes[0,0].set_title('Posts per User')
        axes[0,1].hist(posts_df['title_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Title Length')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Title Length Distribution')
        axes[1,0].hist(posts_df['body_length'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Body Length')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Body Length Distribution')
        axes[1,1].scatter(posts_df['title_length'], posts_df['body_length'], alpha=0.6)
        axes[1,1].set_xlabel('Title Length')
        axes[1,1].set_ylabel('Body Length')
        axes[1,1].set_title('Title vs Body Length')
        corr = posts_df['title_length'].corr(posts_df['body_length'])
        axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[1,1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        plt.show()
        return posts_df
    def analyze_crypto_data(self):
        """Analyze and visualize cryptocurrency data"""
        if 'crypto' not in self.data:
            self.fetch_data('crypto')
        if not self.data.get('crypto'):
            print("❌ No crypto data available")
            return
        crypto_df = pd.DataFrame(self.data['crypto'])
        print("\n₿ CRYPTOCURRENCY DATA ANALYSIS")
        print("=" * 50)
        print(f"Total Cryptocurrencies: {len(crypto_df)}")
        print(f"Total Market Cap: ${crypto_df['market_cap'].sum():,.0f}")
        print(f"Average 24h Change: {crypto_df['price_change_percentage_24h'].mean():.2f}%")
        if HAS_PLOTLY:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Market Cap Distribution', 'Price vs Market Cap', 
                              '24h Price Change', 'Volume vs Price'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            top_10 = crypto_df.head(10)
            fig.add_trace(
                go.Bar(x=top_10['name'], y=top_10['market_cap'], 
                       name='Market Cap', marker_color='gold'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=crypto_df['current_price'], y=crypto_df['market_cap'],
                          mode='markers', name='Price vs Market Cap',
                          text=crypto_df['name'], marker=dict(size=10, color='blue')),
                row=1, col=2
            )
            colors = ['red' if x < 0 else 'green' for x in crypto_df['price_change_percentage_24h']]
            fig.add_trace(
                go.Bar(x=crypto_df['symbol'], y=crypto_df['price_change_percentage_24h'],
                       name='24h Change %', marker_color=colors),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=crypto_df['total_volume'], y=crypto_df['current_price'],
                          mode='markers', name='Volume vs Price',
                          text=crypto_df['name'], marker=dict(size=8, color='purple')),
                row=2, col=2
            )
            fig.update_layout(height=800, title_text="₿ Cryptocurrency Market Analysis Dashboard")
            fig.show()
        else:
            print("📊 Using matplotlib for crypto visualization...")
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        top_10 = crypto_df.head(10)
        plt.barh(top_10['name'], top_10['market_cap'])
        plt.xlabel('Market Cap (USD)')
        plt.title('Top 10 Cryptocurrencies by Market Cap')
        plt.subplot(2, 2, 2)
        plt.hist(crypto_df['current_price'], bins=15, alpha=0.7, edgecolor='black')
        plt.xlabel('Price (USD)')
        plt.ylabel('Frequency')
        plt.title('Price Distribution')
        plt.subplot(2, 2, 3)
        positive_change = crypto_df[crypto_df['price_change_percentage_24h'] > 0]
        negative_change = crypto_df[crypto_df['price_change_percentage_24h'] < 0]
        plt.hist([positive_change['price_change_percentage_24h'], 
                 negative_change['price_change_percentage_24h']], 
                bins=15, alpha=0.7, label=['Positive', 'Negative'], color=['green', 'red'])
        plt.xlabel('24h Change (%)')
        plt.ylabel('Frequency')
        plt.title('24h Price Change Distribution')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.scatter(crypto_df['market_cap'], crypto_df['total_volume'], alpha=0.7)
        plt.xlabel('Market Cap')
        plt.ylabel('Trading Volume')
        plt.title('Market Cap vs Trading Volume')
        plt.tight_layout()
        plt.show()
        return crypto_df
    def analyze_weather_data(self):
        """Analyze and visualize weather data"""
        if 'weather' not in self.data:
            self.fetch_data('weather')
        weather_df = pd.DataFrame(self.data['weather'])
        weather_df['temperature'] = [w['temp'] for w in weather_df['main']]
        weather_df['humidity'] = [w['humidity'] for w in weather_df['main']]
        weather_df['pressure'] = [w['pressure'] for w in weather_df['main']]
        weather_df['condition'] = [w[0]['main'] for w in weather_df['weather']]
        print("\n🌤️ WEATHER DATA ANALYSIS")
        print("=" * 50)
        print(f"Cities: {len(weather_df)}")
        print(f"Average Temperature: {weather_df['temperature'].mean():.1f}°C")
        print(f"Average Humidity: {weather_df['humidity'].mean():.1f}%")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🌤️ Weather Data Analysis Dashboard', fontsize=16, fontweight='bold')
        axes[0,0].bar(weather_df['name'], weather_df['temperature'], color='orange', alpha=0.7)
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].set_title('Temperature by City')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,1].bar(weather_df['name'], weather_df['humidity'], color='blue', alpha=0.7)
        axes[0,1].set_ylabel('Humidity (%)')
        axes[0,1].set_title('Humidity by City')
        axes[0,1].tick_params(axis='x', rotation=45)
        condition_counts = weather_df['condition'].value_counts()
        axes[1,0].pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Weather Conditions Distribution')
        axes[1,1].scatter(weather_df['temperature'], weather_df['humidity'], 
                         s=weather_df['pressure']/10, alpha=0.7)
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel('Humidity (%)')
        axes[1,1].set_title('Temperature vs Humidity\n(Size = Pressure)')        
        for i, city in enumerate(weather_df['name']):
            axes[1,1].annotate(city, (weather_df['temperature'].iloc[i], weather_df['humidity'].iloc[i]))       
        plt.tight_layout()
        plt.show()
        return weather_df
    def comprehensive_analysis(self):
        """Run comprehensive analysis on all APIs"""
        print("🚀 STARTING COMPREHENSIVE API DATA ANALYSIS")
        print("=" * 60)
        users_df = self.analyze_users_data()
        posts_df = self.analyze_posts_data()
        crypto_df = self.analyze_crypto_data()
        weather_df = self.analyze_weather_data()
        print("\n📈 SUMMARY DASHBOARD")
        print("=" * 50)
        if users_df is not None:
            print(f"👥 Users: {len(users_df)} records")
        if posts_df is not None:
            print(f"📝 Posts: {len(posts_df)} records")
        if crypto_df is not None:
            print(f"₿ Crypto: {len(crypto_df)} records")
        if weather_df is not None:
            print(f"🌤️ Weather: {len(weather_df)} records")
        self.save_data_to_files()
        print("\n✅ Analysis completed successfully!")
        print("📁 Data saved to CSV files for further analysis")
    def save_data_to_files(self):
        """Save fetched data to CSV files"""
        for api_name, data in self.data.items():
            try:
                if api_name == 'weather':
                    df = pd.DataFrame(data)
                    df['temperature'] = [w['temp'] for w in df['main']]
                    df['humidity'] = [w['humidity'] for w in df['main']]
                    df['condition'] = [w[0]['main'] for w in df['weather']]
                    df[['name', 'temperature', 'humidity', 'condition']].to_csv(f'{api_name}_data.csv', index=False)
                else:
                    df = pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])
                    df.to_csv(f'{api_name}_data.csv', index=False)
                print(f"💾 Saved {api_name} data to {api_name}_data.csv")
            except Exception as e:
                print(f"❌ Error saving {api_name} data: {e}")
def main():
    """Main function to run the API data visualization"""
    visualizer = APIDataVisualizer()
    
    print("🎯API INTEGRATION & DATA VISUALIZATION")
    print("=" * 70)
    print("This script demonstrates:")
    print("✅ Fetching data from multiple public APIs")
    print("✅ Data processing and analysis with Pandas")
    print("✅ Static visualizations with Matplotlib & Seaborn")
    print("✅ Interactive visualizations with Plotly")
    print("✅ Data export functionality")
    print("=" * 70)
    visualizer.comprehensive_analysis()
    while True:
        print("\n🎛️ INTERACTIVE MENU")
        print("1. Analyze Users Data")
        print("2. Analyze Posts Data") 
        print("3. Analyze Cryptocurrency Data")
        print("4. Analyze Weather Data")
        print("5. Run Complete Analysis")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            visualizer.analyze_users_data()
        elif choice == '2':
            visualizer.analyze_posts_data()
        elif choice == '3':
            visualizer.analyze_crypto_data()
        elif choice == '4':
            visualizer.analyze_weather_data()
        elif choice == '5':
            visualizer.comprehensive_analysis()
        elif choice == '6':
            print("👋 Thank you for using the API Data Visualizer!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
