from pdf2image import convert_from_path

# Specify poppler path directly
pages = convert_from_path(
    r'C:\path\to\your.pdf',
    first_page=1,
    last_page=1,
    poppler_path=r'C:\poppler-25.12.0\bin'  # Add this line
)
print(f"Converted {len(pages)} page(s)")