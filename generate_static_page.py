import os
import markdown
from jinja2 import Environment, FileSystemLoader

def generate_static_page():
    # Read the content of assignment.md
    with open('assignment.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('.'))
    template = env.from_string("""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>观月见心：从"坐标系平移"重探心学本体论</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        .indent {
            text-indent: 2em;
        }
        .reference {
            margin-top: 20px;
            border-top: 1px solid #ccc;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    {{ content | safe }}
</body>
</html>
    """)

    # Render the template with the HTML content
    rendered_html = template.render(content=html_content)

    # Ensure the docs directory exists
    os.makedirs('docs', exist_ok=True)

    # Write the rendered HTML to index.html in the docs directory
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(rendered_html)

if __name__ == "__main__":
    generate_static_page()
    print("Static page generated successfully in docs/index.html")
