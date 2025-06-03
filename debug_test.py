import sys
print("Python version:", sys.version)

try:
    response = input("Test input (type anything): ")
    print(f"You entered: {response}")
    print("Input working correctly!")
except Exception as e:
    print(f"Input error: {e}")

try:
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    
    options = Options()
    options.add_argument("--headless")
    
    driver_path = ChromeDriverManager().install()
    print(f"ChromeDriver installed at: {driver_path}")
    
    driver = webdriver.Chrome(service=Service(driver_path), options=options)
    driver.get("https://www.google.com")
    print("Selenium working!")
    driver.quit()
    
except Exception as e:
    print(f"Selenium error: {e}")
