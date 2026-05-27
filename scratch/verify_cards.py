import re

# Read index.html
with open('index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Extract all href attributes from note-cards
card_hrefs = re.findall(r'class="note-card"\s+href="([^"]+)"', html)
print(f"Found {len(card_hrefs)} note-cards in index.html.")

# Read notes-data.js
with open('notes-data.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Find all keys in notes-data.js
# Keys are defined like: "key_name.md": {
keys = re.findall(r'"([^"]+\.md)":\s*\{', js)
print(f"Found {len(keys)} database entries in notes-data.js.")

# Verify each card
mismatch_count = 0
for href in card_hrefs:
    # Extract file param
    if 'file=' in href:
        file_param = href.split('file=')[1]
        if file_param not in keys:
            print(f"MISMATCH: Card links to '{file_param}' but it is not in notes-data.js!")
            mismatch_count += 1
    else:
        print(f"INVALID HREF: Card has href '{href}' without 'file=' parameter!")
        mismatch_count += 1

if mismatch_count == 0:
    print("SUCCESS: All card hrefs successfully map to database keys!")
else:
    print(f"WARNING: Found {mismatch_count} mismatches.")
