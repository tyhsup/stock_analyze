$code = Get-Content e:/Infinity/mydjango/demo/stock_Django/stock_cost_AI.py -Raw
$prompt = Get-Content e:/Infinity/mydjango/reasoning_prompt_ai.txt -Raw
$full_query = $prompt + "`n`n目前的 `stock_cost_AI.py` 原始碼如下：`n" + $code

# Write to temp file to avoid command line length limits
$full_query | Out-File -Encoding utf8 -FilePath e:/Infinity/mydjango/temp_query.txt

# Run the Python script with the file content
python .agents/helpers/gemma_reasoner.py "$(Get-Content e:/Infinity/mydjango/temp_query.txt -Raw)"
