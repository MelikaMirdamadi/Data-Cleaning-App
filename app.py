import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import PyPDF2
import docx
import re
import json
import os
from langdetect import detect

# تابع تمیز کردن فایل PDF
def clean_pdf(file_path, regex_patterns, progress_bar):
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            output_text = ""
            num_pages = len(reader.pages)

            # پردازش هر صفحه و به‌روزرسانی نوار پیشرفت
            for page in range(num_pages):
                output_text += reader.pages[page].extract_text()
                progress_bar['value'] = (page + 1) / num_pages * 100  # به‌روزرسانی نوار پیشرفت
                root.update_idletasks()  # به‌روزرسانی رابط کاربری

            # تشخیص زبان
            language = detect(output_text)

            # اعمال الگوهای regex برای تمیز کردن
            cleaned_text = apply_regex(output_text, regex_patterns)

            # ایجاد خروجی JSON شامل نام فایل و متن تمیز شده
            save_cleaned_output(file_path, cleaned_text, language)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع تمیز کردن فایل Word
def clean_word(file_path, regex_patterns, progress_bar):
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
    if file_path:  # بررسی اینکه آیا کاربر فایلی انتخاب کرده است
        regex_patterns = [
            (r'[;:\[\]\|\_\?!■]', ''),  # حذف علائم اضافی
            (r'(-{2,})', '-'),  # کاهش تعداد خط تیره‌های متوالی
            (r'(/{2,})', '/'),  # کاهش تعداد اسلش‌های متوالی
            (r'(\.{2,})', '.'),  # کاهش تعداد نقطه‌های متوالی
            (r'(\n{3,})', '\n\n'),  # کاهش تعداد خط جدید متوالی
            (r'\s+', ' ')  # حذف فضاهای خالی اضافی
        ]
        progress_bar['value'] = 0  # تنظیم نوار پیشرفت به صفر
        if file_path.endswith(".pdf"):
            clean_pdf(file_path, regex_patterns, progress_bar)
        elif file_path.endswith(".docx"):
            clean_word(file_path, regex_patterns, progress_bar)

# ساخت رابط کاربری (GUI)
root = tk.Tk()
root.title("برنامه پاكسازي داده ")
root.geometry("400x200")

label = tk.Label(root, text="please select PDF or Word file:")
label.pack(pady=10)

browse_button = tk.Button(root, text="انتخاب فایل", command=browse_file)
browse_button.pack(pady=10)

# افزودن نوار پیشرفت
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate" )
progress_bar.pack(pady=10)

# اجرای برنامه
root.mainloop()
