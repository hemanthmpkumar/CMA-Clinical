import re

def convert():
    path = "latex/jbi/main.tex"
    with open(path, "r") as f:
        content = f.read()
    
    # 1. Documentclass
    content = re.sub(r'\\documentclass\[.*?\]\{article\}', r'\\documentclass[review]{elsarticle}', content)
    
    # 2. Remove natbib, geometry, setstretch
    content = re.sub(r'\\usepackage\[.*?\]\{natbib\}\n?', '', content)
    content = re.sub(r'\\usepackage\{geometry\}\n?', '', content)
    content = re.sub(r'\\usepackage\{setstretch\}\n?', '', content)
    content = re.sub(r'\\geometry\{.*?\}\n?', '', content)
    content = re.sub(r'\\setstretch\{.*?\}\n?', '', content)
    
    # Add journal command
    content = re.sub(r'(\\begin\{document\})', r'\\journal{Journal of Biomedical Informatics}\n\n\1', content)
    
    # Extract title, author, date, abstract
    title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    author_match = re.search(r'\\author\{(.*?)\}', content, re.DOTALL)
    date_match = re.search(r'\\date\{(.*?)\}\n', content, re.DOTALL)
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    
    # Remove original title, author, date, maketitle, abstract
    content = re.sub(r'\\title\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\author\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\date\{.*?\}\n', '', content, flags=re.DOTALL)
    content = re.sub(r'\\maketitle\n?', '', content)
    content = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', content, flags=re.DOTALL)
    
    # Build frontmatter
    title_text = title_match.group(1) if title_match else ""
    # For author, we'll extract the two authors
    author_text = author_match.group(1).strip() if author_match else ""
    authors = [a.strip() for a in author_text.split('\\\\')]
    
    frontmatter = "\\begin{frontmatter}\n\n"
    frontmatter += f"\\title{{{title_text}}}\n\n"
    
    for author in authors:
        if author:
            frontmatter += f"\\author{{{author}}}\n"
            
    frontmatter += "\n\\begin{abstract}\n"
    if abstract_match:
        frontmatter += abstract_match.group(1).strip() + "\n"
    frontmatter += "\\end{abstract}\n\n"
    frontmatter += "\\end{frontmatter}\n"
    
    # Insert frontmatter after \begin{document}
    # Let's find \begin{document} and any trailing whitespace
    content = re.sub(r'(\\begin\{document\}\s*)', r'\1' + frontmatter.replace('\\', '\\\\') + '\n\n', content, count=1)
    
    # Clean up multiple empty lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Bibliography style
    content = re.sub(r'\\bibliographystyle\{.*?\}', r'\\bibliographystyle{elsarticle-num}', content)
    
    with open(path, "w") as f:
        f.write(content)
    
    print("Done converting main.tex")

if __name__ == "__main__":
    convert()
