import requests
from bs4 import BeautifulSoup
import csv
import re

def extract_after_last_product_catalog(title):
    keyword = "Product Catalog"
    idx = title.rfind(keyword)
    if idx != -1:
        return title[idx + len(keyword):].strip()
    return title.strip()  # if not found, return original (trimmed)

url = "https://www.shl.com/solutions/products/product-catalog/view/net-framework-4-5/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

text = soup.get_text(separator=" ", strip=True)

# Strip irrelevant nav text
nav_keywords = ["Services", "Resources", "Careers", "About", "Partners", "Customers", "Practice", "Support"]
for keyword in nav_keywords:
    text = re.sub(rf'\b{keyword}\b.*?\b{keyword}\b', '', text)

# Reapply title and description regex with more precision
title_match = re.search(r'Product Catalog\s+([^\n]+?)\s+Description', text, re.DOTALL)
if title_match:
    title = title_match.group(1).strip()
else:
    title = ""

title = extract_after_last_product_catalog(title)    

desc_match = re.search(r'Description\s+(.*?)\s+Job levels', text, re.DOTALL)
description = desc_match.group(1).strip() if desc_match else ""

# The rest remains the same
job_levels = re.search(r'Job levels\s+(.*?)\s+Languages', text)
language = re.search(r'Languages\s+(.*?)\s+Assessment length', text)
assessment_length = re.search(r'Assessment length\s+(.*?)\s+Test Type', text)
test_type = re.search(r'Test Type:\s*(.*?)\s+Remote Testing', text)

# Clean values
get_value = lambda match: match.group(1).strip() if match else ""

# Remote Testing
remote_testing = "Yes" if "Remote Testing" in text else "No"

# Save to CSV
with open("shl_net_framework_cleaned.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Topic", "Description", "Job Levels", "Language", "Assessment Length", "Test Type", "Remote Testing"])
    writer.writerow([
        title,
        description,
        get_value(job_levels),
        get_value(language),
        get_value(assessment_length),
        get_value(test_type),
        remote_testing
    ])

print("✅ Cleaned CSV saved: 'shl_net_framework_cleaned.csv'")
