import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import PyPDF2
import docx
import re
import json
import os
import pandas as pd
from langdetect import detect
import functools
from transformers import LongformerTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cache the tokenizer to avoid reloading it multiple times
@functools.lru_cache()
def get_tokenizer(model_name):
    return LongformerTokenizer.from_pretrained(model_name)

# تابع چانک کردن متن
def chunk_text(in_txt: str, max_chunk_size: int = 728, chunk_overlap: int = 20) -> list:
    model_name = "allenai/longformer-base-4096"
    tokenizer = get_tokenizer(model_name)

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(in_txt)
    return chunks

# تابع تمیز کردن فایل PDF
def clean_pdf(file_path, regex_patterns, progress_bar):
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            output_text = ""
            num_pages = len(reader.pages)

            for page in range(num_pages):
                output_text += reader.pages[page].extract_text()
                progress_bar['value'] = (page + 1) / num_pages * 100
                root.update_idletasks()

        # تشخیص زبان
        language = detect(output_text)

        # اعمال الگوهای regex برای تمیز کردن
        cleaned_text = apply_regex(output_text, regex_patterns)

        # چانک کردن متن تمیز شده
        chunks = chunk_text(cleaned_text)

        # ایجاد خروجی JSON شامل نام فایل و چانک‌ها
        save_cleaned_output(file_path, chunks, language)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع تمیز کردن فایل Word
def clean_word(file_path, regex_patterns, progress_bar):
    try:
        doc = docx.Document(file_path)
        output_text = "\n".join([para.text for para in doc.paragraphs])
        
        # به‌روزرسانی نوار پیشرفت
        num_paragraphs = len(doc.paragraphs)
        progress_bar['maximum'] = num_paragraphs
        
        for i, para in enumerate(doc.paragraphs):
            progress_bar['value'] = i + 1  # به‌روزرسانی نوار پیشرفت
            root.update_idletasks()  # به‌روزرسانی رابط کاربری

        # تشخیص زبان
        language = detect(output_text)

        # اعمال الگوهای regex برای تمیز کردن
        cleaned_text = apply_regex(output_text, regex_patterns)

        # چانک کردن متن تمیز شده
        chunks = chunk_text(cleaned_text)

        # ایجاد خروجی JSON شامل نام فایل و چانک‌ها
        save_cleaned_output(file_path, chunks, language)

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع تمیز کردن فایل Excel
def clean_excel(file_path, regex_patterns, progress_bar):
    try:
        # بارگذاری فایل Excel
        df = pd.read_excel(file_path, engine="openpyxl")

        # آماده‌سازی الگوهای خروجی
        template = {"file_name": os.path.basename(file_path), "languages": "eng+fas", "Chunk": None}
        state = []
        chunk_raw_size = 6
        perv_faq = ""

        # پردازش سطرهای DataFrame
        for index_raw, (index, row) in enumerate(df.iterrows()):
            question = row["Unnamed: 0"]  #Unnamed: 0
            answer = row["پاسخ"]

            perv_faq += f"\nQuestion : {question} -> Answer : {answer}" + "-" * 15
            if ((index_raw + 1) % chunk_raw_size) == 0:
                state.append(perv_faq.strip())
                perv_faq = ""

        template["Chunk"] = list(set(state))
        json_data = json.dumps(template, indent=4, ensure_ascii=False)

        # ذخیره‌سازی به فایل JSON
        json_file_path = f"{os.path.splitext(file_path)[0]}_faqs.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(json_data)

        messagebox.showinfo("موفقیت", f"فایل تمیز شد و به صورت JSON در {json_file_path} ذخیره شد.")

    except Exception as e:
        messagebox.showerror("خطا", f"مشکلی پیش آمده: {str(e)}")

# تابع ذخیره خروجی تمیز شده در قالب JSON
def save_cleaned_output(file_path, chunks, language):
    output_data = {
        "file_name": os.path.basename(file_path),
        "language": language,
        "chunks": chunks  # چانک‌ها
    }

    json_file_path = f"{os.path.splitext(file_path)[0]}_cleaned.json"
    with open(json_file_path, "w", encoding='utf-8') as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)

    messagebox.showinfo("موفقیت", f"فایل تمیز شد و به صورت JSON در {json_file_path} ذخیره شد.")

# تابع اعمال الگوهای regex
def apply_regex(text, regex_patterns):
    cleaned_text = text
    for pattern, replacement in regex_patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    return cleaned_text

# تابع انتخاب فایل
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Word Files", "*.docx"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
    if file_path:
        regex_patterns = [
            (r'[;:\[\]\|\_\?!■]', ''),  
            (r'(-{2,})', '-'),
            (r'(/{2,})', '/'),
            (r'(\.{2,})', '.'),
            (r'(\n{3,})', '\n\n'), 
            (r'\s+', ' '),  
            (r'\.' , ''), 
            (r"\…{2,}" , ''),
            # (r"\–{2,}" , '–'),
        ]
        progress_bar['value'] = 0  # تنظیم مقدار اولیه نوار پیشرفت
        if file_path.endswith(".pdf"):
            clean_pdf(file_path, regex_patterns, progress_bar)
        elif file_path.endswith(".docx"):
            clean_word(file_path, regex_patterns, progress_bar)
        elif file_path.endswith(".xlsx"):
            clean_excel(file_path, regex_patterns, progress_bar)

# ساخت رابط کاربری
root = tk.Tk()
root.title("برنامه پاکسازی داده ها")
root.geometry("500x300")
root.configure(bg="#f0f0f0")

label = tk.Label(root, text="لطفا فايل خود را انتخاب كنيد", font=("Helvetica", 13, "bold"), bg="#f0f0f0")
label.pack(pady=10)

browse_button = tk.Button(root, text="انتخاب فایل", command=browse_file, bg="blue", fg="white")
browse_button.pack(pady=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

# اضافه کردن فریم تنظیمات
settings_frame = tk.Frame(root, bg="#f0f0f0")
settings_frame.pack(pady=10)

root.mainloop()
