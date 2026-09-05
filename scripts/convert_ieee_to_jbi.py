import re
import os

def convert():
    input_path = "latex/ieee/main.tex"
    output_path = "latex/jbi/main.tex"
    
    with open(input_path, "r") as f:
        content = f.read()
    
    # Replace documentclass
    content = re.sub(r'\\documentclass\[.*?\]\{IEEEtran\}', r'\\documentclass[review]{elsarticle}', content)
    
    # Remove cite package
    content = re.sub(r'\\usepackage\{cite\}\n?', '', content)
    
    # Add journal command
    content = re.sub(r'(\\begin\{document\})', r'\\journal{Journal of Biomedical Informatics}\n\n\1', content)
    
    # Extract blocks
    title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    keywords_match = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', content, re.DOTALL)
    
    # Remove original blocks
    content = re.sub(r'\\title\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\author\{.*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\maketitle\n?', '', content)
    content = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}', '', content, flags=re.DOTALL)
    
    # Build frontmatter
    title_text = title_match.group(1).strip() if title_match else ""
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""
    keywords_text = keywords_match.group(1).strip() if keywords_match else ""
    
    frontmatter = "\\begin{frontmatter}\n\n"
    frontmatter += f"\\title{{{title_text}}}\n\n"
    frontmatter += "\\author{Hemanth Manchabale Papachappa}\n"
    frontmatter += "\\author{Aniriuddha Ganguly}\n\n"
    
    frontmatter += "\\begin{abstract}\n"
    frontmatter += abstract_text + "\n"
    frontmatter += "\\end{abstract}\n\n"
    
    if keywords_text:
        frontmatter += "\\begin{keyword}\n"
        frontmatter += keywords_text + "\n"
        frontmatter += "\\end{keyword}\n\n"
        
    frontmatter += "\\end{frontmatter}\n"
    
    # Insert frontmatter after \begin{document}
    content = re.sub(r'(\\begin\{document\}\s*)', r'\1' + frontmatter.replace('\\', '\\\\') + '\n\n', content, count=1)
    
    # Clean up empty lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Bibliography
    content = re.sub(r'\\bibliographystyle\{IEEEtran\}', r'\\bibliographystyle{elsarticle-num}', content)
    content = re.sub(r'\\bibliography\{\.\./references\}', r'\\bibliography{references}', content)
    
    with open(output_path, "w") as f:
        f.write(content)
        
    print(f"Done converting {input_path} to {output_path}")

if __name__ == "__main__":
    convert()
