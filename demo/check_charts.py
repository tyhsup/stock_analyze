import time, os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--window-size=1920,3000') # Tall window to see everything
driver = webdriver.Chrome(options=options)
try:
    driver.get('http://127.0.0.1:8000/')
    wait = WebDriverWait(driver, 10)
    
    # Enter NVDA and 150 days
    ticker_input = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Ticker (e.g., AAPL)"]')
    ticker_input.clear()
    ticker_input.send_keys('NVDA')
    days_input = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Days"]')
    days_input.clear()
    days_input.send_keys('150')
    driver.find_element(By.XPATH, '//button[contains(text(), "Analyze")]').click()
    
    # Wait for the main chart to load
    wait.until(EC.presence_of_element_located((By.ID, 'kline-main')))
    time.sleep(12) # wait for API data fetch
    
    # Open indicators dropdown
    driver.find_element(By.XPATH, '//button[contains(., "Indicators")]').click()
    wait.until(EC.element_to_be_clickable((By.ID, 'ind-adx')))
    time.sleep(1)
    
    # Click indicators
    driver.find_element(By.ID, 'ind-adx').click()
    time.sleep(1)
    driver.find_element(By.ID, 'ind-aroon').click()
    time.sleep(1)
    driver.find_element(By.ID, 'ind-mfi').click()
    time.sleep(1)
    
    # Close dropdown by clicking header
    driver.find_element(By.XPATH, '//span[contains(text(), "Historical Price")]').click()
    
    # Wait for rendering
    time.sleep(2)
    
    # Check if they exist in DOM
    print("ADX displayed:", driver.find_element(By.ID, 'chart-adx-wrap').is_displayed())
    print("AROON displayed:", driver.find_element(By.ID, 'chart-aroon-wrap').is_displayed())
    print("MFI displayed:", driver.find_element(By.ID, 'chart-mfi-wrap').is_displayed())
    
    # Screenshot
    ss_path = r'C:\\Users\\許廷宇\\.gemini\\antigravity\\brain\\b7fe8e81-6e77-44ef-bf9a-cb6d7dedf934\\selenium_momentum_check.png'
    driver.save_screenshot(ss_path)
    print(f'Screenshot saved to {ss_path}')
    
    # Logs
    for entry in driver.get_log('browser'):
        print(f'Console: {entry}')
finally:
    driver.quit()
