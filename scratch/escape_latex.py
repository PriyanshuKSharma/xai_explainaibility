# Escape LaTeX backslashes in notes-data.js
with open('notes-data.js', 'r') as f:
    content = f.read()

# First replace existing double backslashes to single backslash
content = content.replace('\\\\', '\\')
# Then replace all single backslashes with double backslashes
content = content.replace('\\', '\\\\')

with open('notes-data.js', 'w') as f:
    f.write(content)

print("Backslashes successfully escaped in notes-data.js!")
