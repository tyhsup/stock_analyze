import re

with open("e:/Infinity/mydjango/Gemini_task/app/static/style.css", "r", encoding="utf-8") as f:
    content = f.read()

# 找出所有包含 table, jobs-table, th, td 的 CSS 規則
rules = re.findall(r'([^{]+?\{[^}]+?\})', content)
target_rules = []
for rule in rules:
    if any(kwd in rule for kwd in ["table", "jobs", "th", "td", "nth-", "display", "visibility"]):
        target_rules.append(rule.strip())

print(f"Total target rules: {len(target_rules)}")
for r in target_rules[:50]:
    print("-" * 40)
    print(r)
