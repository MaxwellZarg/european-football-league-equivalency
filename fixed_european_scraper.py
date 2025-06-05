"""
Fixed European Football Scraper for all 7 top leagues (2017-2024)
Properly handles comma-separated numbers and all data formatting issues.
Based on the working England scraper structure.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import logging
from datetime import datetime
import random
from typing import Dict, List, Optional, Tuple
import re

class FixedEuropeanScraper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different ChromeDriver paths
        possible_paths = [
            '/opt/homebrew/bin/chromedriver',  # Mac Homebrew
            '/usr/local/bin/chromedriver',     # Linux/Mac
            'C:/chromedriver/chromedriver.exe', # Windows
            'chromedriver.exe',                # Current directory
            'chromedriver'                     # PATH
        ]
        
        self.chrome_driver_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.chrome_driver_path = path
                break
        
        if not self.chrome_driver_path:
            print("Chrome driver not found. Using default 'chromedriver'")
            self.chrome_driver_path = 'chromedriver'
        
        # Data directory structure
        self.data_dir = os.path.join(self.base_dir, 'data', 'raw', 'european_leagues')
        self.setup_logging()
        
        # All 7 top European leagues
        self.leagues = {
            'Premier-League': {'id': '9', 'country': 'England'},
            'La-Liga': {'id': '12', 'country': 'Spain'}, 
            'Serie-A': {'id': '11', 'country': 'Italy'},
            'Bundesliga': {'id': '20', 'country': 'Germany'},
            'Ligue-1': {'id': '13', 'country': 'France'},
            'Primeira-Liga': {'id': '32', 'country': 'Portugal'},
            'Eredivisie': {'id': '23', 'country': 'Netherlands'}
        }
        
        # 7 seasons (2017-2024)
        self.seasons = [
            '2017-2018', '2018-2019', '2019-2020', '2020-2021', 
            '2021-2022', '2022-2023', '2023-2024'
        ]
        
        # Define stat types with expected columns (based on your working scraper)
        self.stat_types = {
            'standard': {
                'url_part': 'stats',
                'possible_table_ids': ['stats_standard', 'stats_standard_9', 'stats_standard_10', 
                                     'stats_standard_11', 'stats_standard_12', 'stats_standard_13',
                                     'stats_standard_20', 'stats_standard_23', 'stats_standard_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year',
                    'games', 'games_starts', 'minutes', 'minutes_90s', 'goals', 'assists',
                    'goals_assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow',
                    'cards_red', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'progressive_carries',
                    'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90',
                    'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90',
                    'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90',
                    'npxg_xg_assist_per90'
                ]
            },
            'shooting': {
                'url_part': 'shooting',
                'possible_table_ids': ['stats_shooting', 'stats_shooting_9', 'stats_shooting_10', 
                                     'stats_shooting_11', 'stats_shooting_12', 'stats_shooting_13',
                                     'stats_shooting_20', 'stats_shooting_23', 'stats_shooting_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year', 'minutes_90s',
                    'goals', 'shots', 'shots_on_target', 'shots_on_target_pct', 'shots_per90',
                    'shots_on_target_per90', 'goals_per_shot', 'goals_per_shot_on_target',
                    'average_shot_distance', 'shots_free_kicks', 'pens_made', 'pens_att',
                    'xg', 'npxg', 'npxg_per_shot', 'xg_net', 'npxg_net'
                ]
            },
            'passing': {
                'url_part': 'passing',
                'possible_table_ids': ['stats_passing', 'stats_passing_9', 'stats_passing_10', 
                                     'stats_passing_11', 'stats_passing_12', 'stats_passing_13',
                                     'stats_passing_20', 'stats_passing_23', 'stats_passing_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year', 'minutes_90s',
                    'passes_completed', 'passes', 'passes_pct', 'passes_total_distance',
                    'passes_progressive_distance', 'passes_completed_short', 'passes_short',
                    'passes_pct_short', 'passes_completed_medium', 'passes_medium',
                    'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long',
                    'assists', 'xg_assist', 'pass_xa', 'xg_assist_net', 'assisted_shots',
                    'passes_into_final_third', 'passes_into_penalty_area',
                    'crosses_into_penalty_area', 'progressive_passes'
                ]
            },
            'gca': {
                'url_part': 'gca',
                'possible_table_ids': ['stats_gca', 'stats_gca_9', 'stats_gca_10', 
                                     'stats_gca_11', 'stats_gca_12', 'stats_gca_13',
                                     'stats_gca_20', 'stats_gca_23', 'stats_gca_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year', 'minutes_90s',
                    'sca', 'sca_per90', 'sca_passes_live', 'sca_passes_dead', 'sca_take_ons',
                    'sca_shots', 'sca_fouled', 'sca_defense', 'gca', 'gca_per90',
                    'gca_passes_live', 'gca_passes_dead', 'gca_take_ons', 'gca_shots',
                    'gca_fouled', 'gca_defense'
                ]
            },
            'defense': {
                'url_part': 'defense',
                'possible_table_ids': ['stats_defense', 'stats_defense_9', 'stats_defense_10', 
                                     'stats_defense_11', 'stats_defense_12', 'stats_defense_13',
                                     'stats_defense_20', 'stats_defense_23', 'stats_defense_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year', 'minutes_90s',
                    'tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd',
                    'challenge_tackles', 'challenges', 'challenge_tackles_pct', 'challenges_lost',
                    'blocks', 'blocked_shots', 'blocked_passes', 'interceptions',
                    'tackles_interceptions', 'clearances', 'errors'
                ]
            },
            'playingtime': {
                'url_part': 'playingtime',
                'possible_table_ids': ['stats_playing_time', 'stats_playing_time_9', 'stats_playing_time_10', 
                                     'stats_playing_time_11', 'stats_playing_time_12', 'stats_playing_time_13',
                                     'stats_playing_time_20', 'stats_playing_time_23', 'stats_playing_time_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year',
                    'games', 'minutes', 'minutes_per_game', 'minutes_pct', 'minutes_90s',
                    'games_starts', 'minutes_per_start', 'games_complete', 'games_subs',
                    'minutes_per_sub', 'unused_subs', 'points_per_game', 'on_goals_for',
                    'on_goals_against', 'plus_minus', 'plus_minus_per90', 'plus_minus_wowy',
                    'on_xg_for', 'on_xg_against', 'xg_plus_minus', 'xg_plus_minus_per90',
                    'xg_plus_minus_wowy'
                ]
            },
            'misc': {
                'url_part': 'misc',
                'possible_table_ids': ['stats_misc', 'stats_misc_9', 'stats_misc_10', 
                                     'stats_misc_11', 'stats_misc_12', 'stats_misc_13',
                                     'stats_misc_20', 'stats_misc_23', 'stats_misc_32'],
                'expected_columns': [
                    'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year', 'minutes_90s',
                    'cards_yellow', 'cards_red', 'cards_yellow_red', 'fouls', 'fouled', 'offsides',
                    'crosses', 'interceptions', 'tackles_won', 'pens_won', 'pens_conceded',
                    'own_goals', 'ball_recoveries', 'aerials_won', 'aerials_lost', 'aerials_won_pct'
                ]
            }
        }

    def setup_logging(self):
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'fixed_european_scraper_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FixedEuropeanScraper')

    def get_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            service = Service(self.chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            self.logger.error(f"Failed to start Chrome with service: {e}")
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except Exception as e2:
                self.logger.error(f"Failed to start Chrome at all: {e2}")
                raise e2
        
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.set_page_load_timeout(30)
        return driver

    def find_table(self, soup: BeautifulSoup, stat_type: str) -> Optional[BeautifulSoup]:
        """Find the correct table using multiple methods"""
        # Try all possible table IDs for this stat type
        for table_id in self.stat_types[stat_type]['possible_table_ids']:
            table = soup.find('table', {'id': table_id})
            if table:
                self.logger.debug(f"Found table with ID: {table_id}")
                return table
        
        # Fallback: find any table with expected structure
        tables = soup.find_all('table')
        for table in tables:
            if table.find('th', {'data-stat': 'player'}):
                self.logger.debug("Found table using fallback method")
                return table
        
        return None

    def wait_for_table(self, driver: webdriver.Chrome, stat_type: str, timeout: int = 10) -> bool:
        """Wait for table to load with multiple attempts"""
        try:
            for table_id in self.stat_types[stat_type]['possible_table_ids']:
                try:
                    WebDriverWait(driver, timeout).until(
                        EC.presence_of_element_located((By.ID, table_id))
                    )
                    return True
                except TimeoutException:
                    continue
            return False
        except Exception as e:
            self.logger.error(f"Error waiting for table: {str(e)}")
            return False

    def clean_value(self, value: str) -> str:
        """Clean and standardize data values - PROPERLY HANDLES COMMAS"""
        if not value or value == '-':
            return ''
        
        value = value.strip()
        
        # Remove commas from numbers (CRITICAL FIX)
        value = value.replace(',', '')
        
        # Remove percentage signs but keep the number
        value = re.sub(r'%$', '', value)
        
        # Handle ratio format (e.g., "5-2") to decimal
        if '-' in value and len(value.split('-')) == 2:
            try:
                parts = value.split('-')
                if len(parts) == 2 and all(p.replace('.', '').replace('-', '').isdigit() for p in parts):
                    num, den = map(float, parts)
                    return str(num/den if den != 0 else 0)
            except:
                pass
        
        return value

    def extract_player_data(self, row: BeautifulSoup, stat_type: str) -> Optional[Dict]:
        """Extract player data with expected columns and proper cleaning"""
        try:
            stats = {}
            
            # Get player ID first
            player_td = row.find('td', {'data-stat': 'player'})
            if player_td and 'data-append-csv' in player_td.attrs:
                stats['player_id'] = player_td['data-append-csv']
            
            # Extract all expected columns
            for col in self.stat_types[stat_type]['expected_columns']:
                element = row.find(['th', 'td'], {'data-stat': col})
                if element:
                    if col == 'player' and element.find('a'):
                        stats[col] = element.find('a').text.strip()
                    elif col == 'nationality':
                        value = element.text.strip()
                        stats[col] = value[:3] if value else ''
                    else:
                        # Apply proper cleaning (including comma removal)
                        stats[col] = self.clean_value(element.text)
                else:
                    stats[col] = ''
            
            # Only return if we have a player name
            if stats.get('player'):
                return stats
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting player data: {str(e)}")
            return None

    def scrape_table(self, url: str, stat_type: str, league_name: str, season: str, retry_count: int = 3) -> pd.DataFrame:
        """Scrape a specific stats table with retries"""
        driver = None
        for attempt in range(retry_count):
            try:
                if driver:
                    driver.quit()
                driver = self.get_driver()
                
                self.logger.info(f"Scraping {stat_type} from {league_name} {season} (Attempt {attempt + 1}/{retry_count})")
                
                driver.get(url)
                
                # Wait for table to load
                if not self.wait_for_table(driver, stat_type):
                    self.logger.warning(f"Table not found after waiting for {league_name} {season} {stat_type}")
                
                # Add random delay
                time.sleep(random.uniform(3, 7))
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = self.find_table(soup, stat_type)
                
                if not table:
                    self.logger.error(f"Table not found for {stat_type} in {league_name} {season}")
                    continue
                
                tbody = table.find('tbody')
                if not tbody:
                    self.logger.error(f"No tbody found in {stat_type} table for {league_name} {season}")
                    continue
                
                data = []
                for row in tbody.find_all('tr', {'class': lambda x: x is None or 'thead' not in x}):
                    player_data = self.extract_player_data(row, stat_type)
                    if player_data:
                        data.append(player_data)
                
                if not data:
                    self.logger.error(f"No data rows found in {stat_type} table for {league_name} {season}")
                    continue
                
                df = pd.DataFrame(data)
                
                # Verify expected columns and add missing ones
                expected_cols = set(self.stat_types[stat_type]['expected_columns'])
                missing_cols = expected_cols - set(df.columns)
                if missing_cols:
                    self.logger.warning(f"Missing columns in {stat_type}: {missing_cols}")
                    for col in missing_cols:
                        df[col] = ''
                
                self.logger.info(f"Successfully scraped {len(df)} rows for {stat_type} from {league_name} {season}")
                return df
                
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1} for {league_name} {season} {stat_type}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(5, 10))
                continue
            finally:
                if driver:
                    driver.quit()
        
        self.logger.error(f"Failed to scrape {stat_type} for {league_name} {season} after {retry_count} attempts")
        return pd.DataFrame()

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data while preserving important info"""
        non_numeric_cols = ['player', 'nationality', 'position', 'team', 'player_id']
        
        for col in df.columns:
            if col not in non_numeric_cols:
                try:
                    # Convert to numeric, coerce errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    continue
        
        return df

    def scrape_league_season(self, season: str, league_name: str, league_info: dict):
        """Scrape all stats for a specific league and season"""
        league_id = league_info['id']
        country = league_info['country']
        
        base_url = f'https://fbref.com/en/comps/{league_id}/{season}'
        season_dir = os.path.join(self.data_dir, country, f'{league_name}_{season}')
        os.makedirs(season_dir, exist_ok=True)
        
        results = {}
        
        for stat_type in self.stat_types:
            try:
                url = f"{base_url}/{self.stat_types[stat_type]['url_part']}/{season}-{league_name}-Stats"
                
                df = self.scrape_table(url, stat_type, league_name, season)
                
                if not df.empty:
                    # Clean numerical data
                    df = self.clean_numeric_data(df)
                    
                    # Save to CSV
                    filename = f'{stat_type}_stats.csv'
                    filepath = os.path.join(season_dir, filename)
                    df.to_csv(filepath, index=False, encoding='utf-8-sig')
                    self.logger.info(f"Saved {len(df)} rows of {stat_type} data to {filepath}")
                    results[stat_type] = len(df)
                else:
                    self.logger.error(f"Failed to scrape {stat_type} data for {league_name} {season}")
                    results[stat_type] = 0
                
                # Random delay between requests
                time.sleep(random.uniform(3, 7))
                
            except Exception as e:
                self.logger.error(f"Error processing {stat_type} for {league_name} {season}: {str(e)}")
                results[stat_type] = 0
                continue
        
        return results

    def run_full_scrape(self, test_mode: bool = False):
        """Run the complete scraping process for all 7 leagues"""
        if test_mode:
            # Test with just one league and season
            test_leagues = {'Premier-League': self.leagues['Premier-League']}
            test_seasons = ['2023-2024']
            leagues_to_use = test_leagues
            seasons_to_use = test_seasons
            self.logger.info("Running in TEST MODE - Premier League 2023-24 only")
        else:
            leagues_to_use = self.leagues
            seasons_to_use = self.seasons
            self.logger.info("Running FULL SCRAPE - All 7 European leagues, 2017-2024")
        
        # Calculate total tasks
        total_tasks = len(leagues_to_use) * len(seasons_to_use)
        completed_tasks = 0
        
        # Summary tracking
        summary = {}
        
        self.logger.info(f"Starting fixed European league scraper for {len(leagues_to_use)} leagues over {len(seasons_to_use)} seasons")
        self.logger.info(f"Total combinations to scrape: {total_tasks}")
        self.logger.info(f"Leagues: {', '.join(leagues_to_use.keys())}")
        
        for league_name, league_info in leagues_to_use.items():
            league_summary = {}
            country = league_info['country']
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting {league_name} ({country})")
            self.logger.info(f"{'='*60}")
            
            for season in seasons_to_use:
                try:
                    completed_tasks += 1
                    
                    self.logger.info(f"\n{'-'*40}")
                    self.logger.info(f"Progress: {completed_tasks}/{total_tasks} "
                                   f"({(completed_tasks/total_tasks)*100:.1f}%)")
                    self.logger.info(f"Processing: {league_name} {season}")
                    self.logger.info(f"{'-'*40}")
                    
                    season_results = self.scrape_league_season(season, league_name, league_info)
                    league_summary[season] = season_results
                    
                    # Log season summary
                    total_rows = sum(season_results.values())
                    self.logger.info(f"Completed {league_name} {season}: {total_rows} total rows across all stat types")
                    
                    # Add delay between seasons
                    if season != seasons_to_use[-1]:
                        wait_time = random.uniform(10, 15)
                        self.logger.info(f"Waiting {wait_time:.1f} seconds before next season...")
                        time.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {league_name} {season}: {str(e)}")
                    self.logger.info("Waiting 45 seconds before continuing...")
                    time.sleep(45)
                    league_summary[season] = {}
                    continue
            
            summary[league_name] = league_summary
            
            # Add extra delay between leagues
            if league_name != list(leagues_to_use.keys())[-1]:
                wait_time = random.uniform(30, 45)
                self.logger.info(f"\nCompleted all seasons for {league_name}. "
                               f"Waiting {wait_time:.1f} seconds before next league...")
                time.sleep(wait_time)
        
        # Final summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("SCRAPING COMPLETED!")
        self.logger.info(f"{'='*60}")
        
        for league_name, league_data in summary.items():
            total_league_rows = sum(sum(s.values()) for s in league_data.values())
            self.logger.info(f"{league_name}: {total_league_rows} total rows")
            
            # Show any failed seasons
            failed_seasons = [s for s, data in league_data.items() if sum(data.values()) == 0]
            if failed_seasons:
                self.logger.warning(f"  Failed seasons: {', '.join(failed_seasons)}")
        
        self.logger.info(f"\nData saved to: {self.data_dir}")
        return summary


def main():
    """Main function to run the fixed European league scraper"""
    print("FIXED European League Scraper - All 7 Top Leagues")
    print("Properly handles comma-separated numbers and all data formatting")
    print("Leagues: England, Spain, Italy, Germany, France, Portugal, Netherlands")
    print("Seasons: 2017-2024 (7 years)")
    print("=" * 80)
    
    scraper = FixedEuropeanScraper()
    
    # Ask user if they want test mode
    test_mode = input("Run in test mode? (y/n) [tests with Premier League 2023-24 only]: ").lower().strip()
    test_mode = test_mode in ['y', 'yes', '1', 'true']
    
    try:
        summary = scraper.run_full_scrape(test_mode=test_mode)
        
        print(f"\n{'='*80}")
        print("SCRAPING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        # Print summary
        total_files = 0
        for league_name, league_data in summary.items():
            league_files = sum(len([s for s in season_data.values() if s > 0]) for season_data in league_data.values())
            total_files += league_files
            print(f"{league_name}: {league_files} successful stat files")
        
        print(f"\nTotal successful files: {total_files}")
        print(f"Data location: {scraper.data_dir}")
        
        if test_mode:
            print("\nTest completed! Run again without test mode for full scrape of all 7 leagues.")
        else:
            print("\nYou now have properly cleaned data for all 7 top European leagues!")
            print("All comma-separated numbers properly handled!")
            print("Ready for accurate equivalency analysis!")
        
        return summary
        
    except KeyboardInterrupt:
        scraper.logger.info("Scraping interrupted by user")
        print("\nScraping interrupted by user")
        return None
    except Exception as e:
        scraper.logger.error(f"Unexpected error during scraping: {str(e)}")
        print(f"\nError during scraping: {str(e)}")
        print("Check the logs for detailed error messages")
        return None


if __name__ == "__main__":
    main()
