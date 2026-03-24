import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--window-size=1920,5000') # Large height to capture everything
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

try:
    print("Starting browser...")
    driver = webdriver.Chrome(options=options)
    
    print("Loading dashboard...")
    driver.get('http://127.0.0.1:8000/')
    driver.implicitly_wait(10)
    
    print("Inputting AAPL and searching...")
    search_box = driver.find_element(By.NAME, 'stock_number')
    search_box.clear()
    search_box.send_keys('AAPL')
    
    days_box = driver.find_element(By.NAME, 'days')
    days_box.clear()
    days_box.send_keys('120')
    
    btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    btn.click()
    
    # Wait for the main chart to render
    print("Waiting for data to load and charts to render (max 30s)...")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, 'kline-chart'))
    )
    
    # Give it some extra time for apexcharts to animate and render completely
    time.sleep(15)
    
    print("1. Taking full page screenshot (Dashboard)...")
    driver.save_screenshot('e:/Infinity/mydjango/demo/aapl_dashboard_full.png')
    
    # Test News
    print("2. Testing News & Sentiment...")
    driver.get('http://127.0.0.1:8000/news/?news_query=AAPL&news_days=30')
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'card'))
    )
    time.sleep(5)
    driver.save_screenshot('e:/Infinity/mydjango/demo/aapl_news.png')
    
    # Test Institutional
    print("3. Testing Institutional...")
    driver.get('http://127.0.0.1:8000/institutional/')
    # In the Institutional page the inputs might be different. 
    # Usually the input name is 'ticker' or 'stock_id' but I'll check with a generic method.
    try:
        search_box = driver.find_element(By.NAME, 'ticker')
    except:
        search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
    search_box.clear()
    search_box.send_keys('AAPL')
    btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    btn.click()
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, 'institutional-chart'))
    )
    time.sleep(5)
    driver.save_screenshot('e:/Infinity/mydjango/demo/aapl_institutional.png')
    
    # Test Fair Value
    print("4. Testing Fair Value...")
    driver.get('http://127.0.0.1:8000/valuation/')
    try:
        search_box = driver.find_element(By.NAME, 'ticker')
    except:
        search_box = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
    search_box.clear()
    search_box.send_keys('AAPL')
    btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    btn.click()
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'card'))
    )
    time.sleep(5)
    driver.save_screenshot('e:/Infinity/mydjango/demo/aapl_valuation.png')

    print("SUCCESS: All tests completed.")
except Exception as e:
    print(f"ERROR: {e}")
    driver.save_screenshot('e:/Infinity/mydjango/demo/aapl_error.png')
finally:
    driver.quit()
