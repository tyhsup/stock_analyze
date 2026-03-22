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
    
    # Enter AAPL and 150 days
    ticker_input = driver.find_element(By.ID, 'ticker')
    ticker_input.clear()
    ticker_input.send_keys('AAPL')
    days_input = driver.find_element(By.ID, 'days')
    days_input.clear()
    days_input.send_keys('150')
    driver.find_element(By.XPATH, '//button[contains(text(), "Analyze")]').click()
    
    # Wait for the main chart to load
    wait.until(EC.presence_of_element_located((By.ID, 'kline-main')))
    time.sleep(15) # wait for API data fetch
    
    # Scroll to the investor chart area
    driver.execute_script('document.getElementById("investor-chart").scrollIntoView();')
    time.sleep(4)
    
    # Screenshot
    ss_path = r'C:\\Users\\許廷宇\\.gemini\\antigravity\\brain\\b7fe8e81-6e77-44ef-bf9a-cb6d7dedf934\\us_investors_check.png'
    driver.save_screenshot(ss_path)
    print(f'Screenshot saved to {ss_path}')
finally:
    driver.quit()
