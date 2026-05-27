# Escape inner backticks in notes-data.js
with open('notes-data.js', 'r') as f:
    text = f.read()

out = []
in_content = False
i = 0
n = len(text)

while i < n:
    # Check if we are starting a content block: "content: `"
    if not in_content and i + 10 < n and text[i:i+10] == "content: `":
        in_content = True
        out.append(text[i:i+10])
        i += 10
        continue
    
    # Check if we are ending a content block: "`" followed by closure
    if in_content and text[i] == '`':
        # If the backtick is followed by the closing tags in notesData object
        # A closing backtick in our file is always followed by:
        # \n  }, or \n  }
        is_closing = False
        lookahead = text[i+1:i+15]
        if lookahead.strip().startswith('}'):
            is_closing = True
            
        if is_closing:
            in_content = False
            out.append('`')
            i += 1
            continue
        else:
            # It's an inner backtick! Let's escape it as \`
            out.append('\\`')
            i += 1
            continue
            
    out.append(text[i])
    i += 1

with open('notes-data.js', 'w') as f:
    f.write("".join(out))

print("Inner backticks successfully escaped in notes-data.js!")
