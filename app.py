import tkinter as tk
from tkinter import filedialog, messagebox
import PyPDF2
import docx
import re
import json
import os
from langdetect import detect

# تابع تمیز کردن فایل PDF
def clean_pdf(file_path, regex_patterns):
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            output_text = ""
            for page in range(len(reader.pages)):
                output_text += reader.pages[page].extract_text()

            # تشخیص زبان
            language = detect(output_text)

            # اعمال الگوهای regex برای تمیز کردن
            cleaned_text = apply_regex(output_text, regex_patterns)

            # ایجاد خروجی JSON شامل نام فایل و متن تمیز شده
            save_cleaned_output(file_path, cleaned_text, language)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع تمیز کردن فایل Word
def clean_word(file_path, regex_patterns):
    try:
        doc = docx.Document(file_path)
        output_text = "\n".join([para.text for para in doc.paragraphs])

        # تشخیص زبان
        language = detect(output_text)

        # اعمال الگوهای regex برای تمیز کردن
        cleaned_text = apply_regex(output_text, regex_patterns)

        # ایجاد خروجی JSON شامل نام فایل و متن تمیز شده
        save_cleaned_output(file_path, cleaned_text, language)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع اعمال الگوهای regex
def apply_regex(text, regex_patterns):
    cleaned_text = text
    for pattern, replacement in regex_patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    return cleaned_text

# تابع ذخیره خروجی تمیز شده در قالب JSON
def save_cleaned_output(file_path, cleaned_text, language):
    output_data = {
        "file_name": os.path.basename(file_path),
        "language": language,
        "cleaned_text": cleaned_text
    }

    # ذخیره فایل JSON
    json_file_path = f"{os.path.splitext(file_path)[0]}_cleaned.json"
    with open(json_file_path, "w", encoding='utf-8') as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)

    messagebox.showinfo("موفقیت", f"فایل تمیز شد و به صورت JSON در {json_file_path} ذخیره شد.")

# تابع برای انتخاب فایل و نوع فایل
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Word Files", "*.docx")])
    if file_path:
        regex_patterns = [
            (r'[;:\[\]\|\_\?!■]', ''),
            (r'(-{2,})', '-'),
            (r'(/{2,})', '/'),
            (r'(\.{2,})', '.'),
            (r'(n\\{3,})', '\n'),
            
            # اینجا می‌توانید الگوهای بیشتری اضافه کنید
        ]
        if file_path.endswith(".pdf"):
            clean_pdf(file_path, regex_patterns)
        elif file_path.endswith(".docx"):
            clean_word(file_path, regex_patterns)

# ساخت رابط کاربری (GUI)
root = tk.Tk()
root.title("برنامه تمیزکننده فایل‌ها")
root.geometry("300x150")

label = tk.Label(root, text="فایل PDF یا Word خود را انتخاب کنید:")
label.pack(pady=10)

browse_button = tk.Button(root, text="انتخاب فایل", command=browse_file)
browse_button.pack(pady=10)

# اجرای برنامه
root.mainloop()
