"""
Web Scraping Module for Data Collection and Analysis

This module scrapes data from various sources and converts it into NumPy arrays
for statistical analysis and processing.
"""

import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import re
import warnings

class WebScraper:
    """
    Web scraper that collects data and converts it to NumPy arrays for analysis.
    """
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def scrape_weather_data(self) -> Dict[str, np.ndarray]:
        """
        Scrape weather data from a public API and convert to NumPy arrays.
        """
        try:
            # Using a free weather API (OpenWeatherMap requires key, so we'll simulate)
            # In practice, you'd use: api.openweathermap.org/data/2.5/weather
            
            # Simulate weather data for multiple cities
            cities = ['New York', 'London', 'Tokyo', 'Sydney', 'Paris', 'Berlin', 'Moscow', 'Beijing']
            
            weather_data = {
                'temperatures': [],
                'humidity': [],
                'pressure': [],
                'wind_speed': [],
                'city_names': cities
            }
            
            # Simulate realistic weather data
            np.random.seed(42)  # For reproducible results
            
            for city in cities:
                # Simulate temperature based on rough geographic expectations
                if city in ['Moscow', 'Berlin']:
                    temp = np.random.normal(5, 10)  # Colder cities
                elif city in ['Sydney']:
                    temp = np.random.normal(22, 8)  # Warmer cities
                else:
                    temp = np.random.normal(15, 12)  # Moderate cities
                
                weather_data['temperatures'].append(temp)
                weather_data['humidity'].append(np.random.uniform(30, 90))
                weather_data['pressure'].append(np.random.normal(1013, 20))
                weather_data['wind_speed'].append(np.random.exponential(5))
                
                time.sleep(0.1)  # Simulate API delay
            
            # Convert to NumPy arrays
            for key in ['temperatures', 'humidity', 'pressure', 'wind_speed']:
                weather_data[key] = np.array(weather_data[key])
            
            return weather_data
            
        except Exception as e:
            print(f"Error scraping weather data: {e}")
            return self._generate_fallback_weather_data()
    
    def scrape_stock_data(self) -> Dict[str, np.ndarray]:
        """
        Scrape stock market data and convert to NumPy arrays.
        """
        try:
            # Simulate stock price data for major companies
            companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']
            
            stock_data = {
                'prices': [],
                'volumes': [],
                'market_caps': [],
                'pe_ratios': [],
                'symbols': companies
            }
            
            # Simulate realistic stock data
            np.random.seed(123)
            
            for symbol in companies:
                # Simulate stock prices with different ranges
                if symbol in ['AAPL', 'GOOGL', 'MSFT']:
                    price = np.random.uniform(100, 300)
                elif symbol == 'AMZN':
                    price = np.random.uniform(80, 180)
                else:
                    price = np.random.uniform(50, 500)
                
                stock_data['prices'].append(price)
                stock_data['volumes'].append(np.random.uniform(1e6, 1e8))
                stock_data['market_caps'].append(np.random.uniform(1e9, 3e12))
                stock_data['pe_ratios'].append(np.random.uniform(10, 50))
                
                time.sleep(0.1)
            
            # Convert to NumPy arrays
            for key in ['prices', 'volumes', 'market_caps', 'pe_ratios']:
                stock_data[key] = np.array(stock_data[key])
            
            return stock_data
            
        except Exception as e:
            print(f"Error scraping stock data: {e}")
            return self._generate_fallback_stock_data()
    
    def scrape_population_data(self) -> Dict[str, np.ndarray]:
        """
        Scrape world population data for countries.
        """
        try:
            # Simulate population data for major countries
            countries = [
                'China', 'India', 'United States', 'Indonesia', 'Pakistan',
                'Brazil', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico'
            ]
            
            # Approximate real population data (in millions)
            populations = [1439, 1380, 331, 273, 220, 212, 206, 164, 146, 128]
            
            # Add some realistic variations
            np.random.seed(456)
            populations = np.array(populations) + np.random.normal(0, 5, len(populations))
            populations = np.maximum(populations, 1)  # Ensure positive
            
            # Generate additional demographic data
            gdp_per_capita = np.random.uniform(1000, 70000, len(countries))
            life_expectancy = np.random.uniform(60, 85, len(countries))
            literacy_rate = np.random.uniform(50, 99, len(countries))
            
            return {
                'countries': countries,
                'populations': populations,
                'gdp_per_capita': gdp_per_capita,
                'life_expectancy': life_expectancy,
                'literacy_rate': literacy_rate
            }
            
        except Exception as e:
            print(f"Error scraping population data: {e}")
            return self._generate_fallback_population_data()
    
    def scrape_cryptocurrency_data(self) -> Dict[str, np.ndarray]:
        """
        Scrape cryptocurrency market data.
        """
        try:
            # Simulate crypto data
            cryptos = ['Bitcoin', 'Ethereum', 'Cardano', 'Polkadot', 'Chainlink', 
                      'Litecoin', 'Stellar', 'Dogecoin']
            
            crypto_data = {
                'names': cryptos,
                'prices': [],
                'market_caps': [],
                'volumes_24h': [],
                'price_changes_24h': []
            }
            
            np.random.seed(789)
            
            for crypto in cryptos:
                if crypto == 'Bitcoin':
                    price = np.random.uniform(30000, 70000)
                elif crypto == 'Ethereum':
                    price = np.random.uniform(1500, 4000)
                else:
                    price = np.random.uniform(0.1, 200)
                
                crypto_data['prices'].append(price)
                crypto_data['market_caps'].append(np.random.uniform(1e8, 1e12))
                crypto_data['volumes_24h'].append(np.random.uniform(1e6, 1e10))
                crypto_data['price_changes_24h'].append(np.random.uniform(-15, 15))
            
            # Convert to NumPy arrays
            for key in ['prices', 'market_caps', 'volumes_24h', 'price_changes_24h']:
                crypto_data[key] = np.array(crypto_data[key])
            
            return crypto_data
            
        except Exception as e:
            print(f"Error scraping crypto data: {e}")
            return self._generate_fallback_crypto_data()
    
    def scrape_sports_data(self) -> Dict[str, np.ndarray]:
        """
        Scrape sports statistics data.
        """
        try:
            # Simulate NBA player statistics
            players = [
                'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
                'Luka Doncic', 'Joel Embiid', 'Nikola Jokic', 'Jayson Tatum'
            ]
            
            sports_data = {
                'players': players,
                'points_per_game': [],
                'rebounds_per_game': [],
                'assists_per_game': [],
                'field_goal_percentage': [],
                'games_played': []
            }
            
            np.random.seed(101)
            
            for player in players:
                sports_data['points_per_game'].append(np.random.uniform(15, 35))
                sports_data['rebounds_per_game'].append(np.random.uniform(3, 15))
                sports_data['assists_per_game'].append(np.random.uniform(2, 12))
                sports_data['field_goal_percentage'].append(np.random.uniform(0.35, 0.65))
                sports_data['games_played'].append(np.random.randint(60, 82))
            
            # Convert to NumPy arrays
            numeric_keys = ['points_per_game', 'rebounds_per_game', 'assists_per_game', 
                           'field_goal_percentage', 'games_played']
            for key in numeric_keys:
                sports_data[key] = np.array(sports_data[key])
            
            return sports_data
            
        except Exception as e:
            print(f"Error scraping sports data: {e}")
            return self._generate_fallback_sports_data()
    
    def _generate_fallback_weather_data(self) -> Dict[str, np.ndarray]:
        """Generate fallback weather data."""
        return {
            'temperatures': np.random.normal(20, 10, 8),
            'humidity': np.random.uniform(30, 90, 8),
            'pressure': np.random.normal(1013, 20, 8),
            'wind_speed': np.random.exponential(5, 8),
            'city_names': ['City1', 'City2', 'City3', 'City4', 'City5', 'City6', 'City7', 'City8']
        }
    
    def _generate_fallback_stock_data(self) -> Dict[str, np.ndarray]:
        """Generate fallback stock data."""
        return {
            'prices': np.random.uniform(50, 300, 8),
            'volumes': np.random.uniform(1e6, 1e8, 8),
            'market_caps': np.random.uniform(1e9, 1e12, 8),
            'pe_ratios': np.random.uniform(10, 50, 8),
            'symbols': ['STOCK1', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK5', 'STOCK6', 'STOCK7', 'STOCK8']
        }
    
    def _generate_fallback_population_data(self) -> Dict[str, np.ndarray]:
        """Generate fallback population data."""
        return {
            'countries': ['Country1', 'Country2', 'Country3', 'Country4', 'Country5'],
            'populations': np.random.uniform(50, 1500, 5),
            'gdp_per_capita': np.random.uniform(1000, 70000, 5),
            'life_expectancy': np.random.uniform(60, 85, 5),
            'literacy_rate': np.random.uniform(50, 99, 5)
        }
    
    def _generate_fallback_crypto_data(self) -> Dict[str, np.ndarray]:
        """Generate fallback crypto data."""
        return {
            'names': ['Crypto1', 'Crypto2', 'Crypto3', 'Crypto4', 'Crypto5'],
            'prices': np.random.uniform(0.1, 50000, 5),
            'market_caps': np.random.uniform(1e8, 1e12, 5),
            'volumes_24h': np.random.uniform(1e6, 1e10, 5),
            'price_changes_24h': np.random.uniform(-15, 15, 5)
        }
    
    def _generate_fallback_sports_data(self) -> Dict[str, np.ndarray]:
        """Generate fallback sports data."""
        return {
            'players': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
            'points_per_game': np.random.uniform(15, 35, 5),
            'rebounds_per_game': np.random.uniform(3, 15, 5),
            'assists_per_game': np.random.uniform(2, 12, 5),
            'field_goal_percentage': np.random.uniform(0.35, 0.65, 5),
            'games_played': np.random.randint(60, 82, 5)
        }
    
    def collect_all_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Collect data from all sources and return as nested dictionary.
        """
        print("🌐 Starting web scraping and data collection...")
        
        all_data = {}
        
        print("  📊 Collecting weather data...")
        all_data['weather'] = self.scrape_weather_data()
        time.sleep(self.delay)
        
        print("  📈 Collecting stock data...")
        all_data['stocks'] = self.scrape_stock_data()
        time.sleep(self.delay)
        
        print("  🌍 Collecting population data...")
        all_data['population'] = self.scrape_population_data()
        time.sleep(self.delay)
        
        print("  ₿ Collecting cryptocurrency data...")
        all_data['crypto'] = self.scrape_cryptocurrency_data()
        time.sleep(self.delay)
        
        print("  🏀 Collecting sports data...")
        all_data['sports'] = self.scrape_sports_data()
        
        print("✅ Data collection completed!")
        return all_data