import os

file_path = r'e:\Infinity\mydjango\demo\templates\home.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove all my recent attempt endif tags to start fresh in that region
# We'll look for the block starting with {% if kline_json %}
start_token = '{% if kline_json %}'
end_token = '{% if number %}'

if start_token in content and end_token in content:
    start_idx = content.find(start_token)
    end_idx = content.find(end_token)
    
    # Let's extract everything from start_token to end_token
    # and reconstruct it properly.
    
    prefix = content[:start_idx]
    suffix = content[end_idx:]
    
    # The block we are replacing covers:
    # 1. Financial Overview
    # 2. OHLCV K-line Charts
    # 3. Specific Ticker Investor Trend
    
    # I will rebuild the core structure:
    new_inner = """{% if kline_json %}
<!-- Feature: Financial Metrics Overview -->
<div class="row mt-4">
    <div class="col-12">
        <div class="card border-0 shadow-sm bg-light">
            <div class="card-header bg-white border-0 py-3 fw-bold d-flex justify-content-between align-items-center">
                <span><i class="bi bi-bar-chart-line-fill me-2 text-primary"></i> Financial Overview: {{ number }}</span>
                <span class="badge bg-secondary">Data from yfinance</span>
            </div>
            <div class="card-body p-3">
                {% if financial_summary %}
                <div class="row g-3">
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4 border-primary">
                            <div class="text-muted small mb-1">P/E Ratio</div>
                            <div class="h5 mb-0 fw-bold text-dark">{{ financial_summary.pe|default:"--" }}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4 border-info">
                            <div class="text-muted small mb-1">P/B Ratio</div>
                            <div class="h5 mb-0 fw-bold text-dark">{{ financial_summary.pb|default:"--" }}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4 border-success">
                            <div class="text-muted small mb-1">EPS (Trailing)</div>
                            <div class="h5 mb-0 fw-bold text-success">{{ financial_summary.eps|default:"--" }}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4 border-warning">
                            <div class="text-muted small mb-1">Market Cap</div>
                            <div class="h5 mb-0 fw-bold text-dark">{{ financial_summary.marketCap|default:"--" }}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4 border-danger">
                            <div class="text-muted small mb-1">ROE</div>
                            <div class="h5 mb-0 fw-bold {% if financial_summary.roe > 0 %}text-success{% elif financial_summary.roe < 0 %}text-danger{% endif %}">
                                {% if financial_summary.roe %}{{ financial_summary.roe|floatformat:2 }}%{% else %}--{% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4" style="border-color: #6f42c1 !important;">
                            <div class="text-muted small mb-1">Gross Margin</div>
                            <div class="h5 mb-0 fw-bold">
                                {% if financial_summary.gross_margin %}{{ financial_summary.gross_margin|floatformat:2 }}%{% else %}--{% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4" style="border-color: #fd7e14 !important;">
                            <div class="text-muted small mb-1">Revenue Growth</div>
                            <div class="h5 mb-0 fw-bold {% if financial_summary.revenue_growth > 0 %}text-success{% elif financial_summary.revenue_growth < 0 %}text-danger{% endif %}">
                                {% if financial_summary.revenue_growth %}{{ financial_summary.revenue_growth|floatformat:2 }}%{% else %}--{% endif %}
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="p-3 bg-white rounded shadow-xs border-start border-4" style="border-color: #20c997 !important;">
                            <div class="text-muted small mb-1">52W High</div>
                            <div class="h5 mb-0 fw-bold text-dark">{{ financial_summary.fiftyTwoWeekHigh|default:"--" }}</div>
                        </div>
                    </div>
                </div>
                {% else %}
                <div class="text-center py-4 text-muted">
                    <i class="bi bi-info-circle me-2"></i> No fundamental data available from yfinance for this ticker.
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-lg-12">
        <div class="card shadow-sm border-0">
            <div class="card-header bg-dark text-white">
                <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                    <span><i class="bi bi-graph-up-arrow me-2"></i> Historical Price &amp; Volume (OHLCV)</span>
                    <div class="d-flex align-items-center gap-3 flex-wrap">
                        <div class="btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-outline-light active" id="btn-day" onclick="setPeriod('D')">Day</button>
                            <button type="button" class="btn btn-outline-light" id="btn-week" onclick="setPeriod('W')">Week</button>
                            <button type="button" class="btn btn-outline-light" id="btn-month" onclick="setPeriod('M')">Month</button>
                        </div>
                        <button class="btn btn-outline-light btn-sm" onclick="resetBrushZoom()"><i class="bi bi-arrow-counterclockwise me-1"></i> Reset Brush</button>
                        <div class="dropdown">
                            <button class="btn btn-outline-light btn-sm dropdown-toggle" type="button" data-bs-toggle="dropdown" data-bs-auto-close="outside">
                                <i class="bi bi-sliders me-1"></i> Indicators
                            </button>
                            <ul class="dropdown-menu dropdown-menu-end p-3" id="indicators-dropdown" style="min-width:320px;max-height:600px;overflow-y:auto;z-index:1050;">
                                <li class="text-center p-3 text-muted">Loading indicators...</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            <div class="card-body p-0">
                <div id="kline-main" style="min-height: 450px;"></div>
                <div id="kline-volume" style="min-height: 160px;"></div>
                <div id="kline-rsi-wrap" style="display:none;">
                    <div class="d-flex justify-content-between align-items-center px-3 pt-2">
                        <strong class="small text-muted">RSI (14)</strong>
                        <button class="btn btn-outline-secondary btn-sm py-0" onclick="resetOscillatorZoom('rsi')">↺ Reset Zoom</button>
                    </div>
                    <div id="kline-rsi" style="min-height: 220px;"></div>
                </div>
                <div id="kline-macd-wrap" style="display:none;">
                    <div class="d-flex justify-content-between align-items-center px-3 pt-2">
                        <strong class="small text-muted">MACD (12,26,9)</strong>
                        <button class="btn btn-outline-secondary btn-sm py-0" onclick="resetOscillatorZoom('macd')">↺ Reset Zoom</button>
                    </div>
                    <div id="kline-macd" style="min-height: 220px;"></div>
                </div>
                <!-- Other oscillator panels placeholder -->
                <div id="kline-stoch-wrap" style="display:none;"><div id="kline-stoch"></div></div>
                <div id="kline-cci-wrap" style="display:none;"><div id="kline-cci"></div></div>
                <div id="kline-williamsr-wrap" style="display:none;"><div id="kline-williamsr"></div></div>
                <div id="kline-atr-wrap" style="display:none;"><div id="kline-atr"></div></div>
                <div id="kline-obv-wrap" style="display:none;"><div id="kline-obv"></div></div>
                <div id="kline-ad-wrap" style="display:none;"><div id="kline-ad"></div></div>
                <div id="kline-cycle-wrap" style="display:none;"><div id="kline-cycle"></div></div>
                <div id="kline-stats-wrap" style="display:none;"><div id="kline-stats"></div></div>
                <div id="kline-math-wrap" style="display:none;"><div id="kline-math"></div></div>
            </div>
        </div>
    </div>
</div>

{% if investor_json %}
<div class="row mt-4">
    <div class="col-lg-12">
        <div class="card">
            <div class="card-header fw-bold d-flex justify-content-between align-items-center">
                <span><i class="bi bi-graph-up me-2"></i> Specific Ticker Investor Trend</span>
                <div class="d-flex gap-2">
                    <button class="btn btn-outline-secondary btn-sm" onclick="resetOscillatorZoom('investor')">↺ Reset</button>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary active" id="inv-btn-stacked" onclick="toggleInvestorChartMode('stacked')">Stacked</button>
                        <button class="btn btn-outline-primary" id="inv-btn-grouped" onclick="toggleInvestorChartMode('grouped')">Grouped</button>
                    </div>
                </div>
            </div>
            <div class="card-body p-3">
                <div id="investor-chart" style="min-height: 400px;"></div>
            </div>
        </div>
    </div>
</div>
{% endif %}
{% endif %}

"""
    
    # We need to make sure we don't accidentally leave hanging blocks.
    # The end_token is {% if number %}.
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(prefix + new_inner + suffix)
    print("Successfully rebuilt kline_json block structure.")
else:
    print("Could not find start/end tokens.")
