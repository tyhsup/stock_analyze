import talib
import numpy as np
import pandas as pd
import math

def debug_indicators():
    # Create 200 days of dummy data
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 200),
        'High': np.random.uniform(200, 300, 200),
        'Low': np.random.uniform(50, 100, 200),
        'Close': np.random.uniform(100, 200, 200),
        'Volume': np.random.uniform(1000, 5000, 200)
    })
    
    O = data['Open'].values.astype(float)
    H = data['High'].values.astype(float)
    L = data['Low'].values.astype(float)
    C = data['Close'].values.astype(float)
    V = data['Volume'].values.astype(float)

    groups = talib.get_function_groups()
    failures = []
    success_counts = {}

    for group_name, functions in groups.items():
        success_counts[group_name] = 0
        for func_name in functions:
            try:
                if func_name == 'MAVP': continue
                func = getattr(talib, func_name)
                
                output = None
                if group_name == 'Pattern Recognition':
                    output = func(O, H, L, C)
                elif group_name == 'Volume Indicators':
                    if func_name in ['AD', 'ADOSC', 'OBV']:
                        output = func(H, L, C, V) if func_name != 'OBV' else func(C, V)
                    else:
                        output = func(H, L, C, V)
                elif group_name in ['Volatility Indicators', 'Cycle Indicators']:
                    try:
                        output = func(H, L, C)
                    except:
                        output = func(C)
                elif group_name == 'Price Transform':
                    if any(x in func_name for x in ['AVG', 'TYP', 'WCL']):
                        output = func(O, H, L, C)
                    elif 'MED' in func_name:
                        output = func(H, L)
                    else:
                        output = func(C)
                else:
                    try:
                        output = func(C)
                    except:
                        try:
                            output = func(H, L, C)
                        except:
                            failures.append((func_name, "Input Mismatch"))
                            continue

                if output is not None:
                    if isinstance(output, tuple):
                        # Check if all are NaNs
                        for i, out in enumerate(output):
                            if np.isnan(out).all():
                                failures.append((f"{func_name}_part_{i}", "All NaNs"))
                            else:
                                success_counts[group_name] += 1
                    else:
                        if np.isnan(output).all():
                            failures.append((func_name, "All NaNs"))
                        else:
                            success_counts[group_name] += 1
                else:
                    failures.append((func_name, "Returned None"))

            except Exception as e:
                failures.append((func_name, str(e)))

    print("\n--- TA-Lib Diagnosis Results ---")
    print(f"Total Groups: {len(groups)}")
    for group, count in success_counts.items():
        print(f"Group '{group}': {count} successful indicators")
    
    print("\n--- Problematic Indicators ---")
    for func, reason in failures:
        print(f"FAILED: {func} | Reason: {reason}")

if __name__ == "__main__":
    debug_indicators()
