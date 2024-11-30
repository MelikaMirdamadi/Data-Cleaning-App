import openai
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import PyPDF2
import docx
import re
from pathlib import Path
from hazm import Normalizer, word_tokenize, sent_tokenize
import threading
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import langdetect
import functools
from transformers import LongformerTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class PersianTextCleaner:
    def __init__(self):
        self.normalizer = Normalizer()
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=self.get_persian_stop_words())
        self.similarity_threshold = 0.8
        self.min_chunk_length = 100
        self.stored_chunks = set()
        # Initialize tokenizer for chunking
        self.model_name = "allenai/longformer-base-4096"
        self.tokenizer = self.get_tokenizer(self.model_name)
    
    #Chunking function
    @functools.lru_cache()
    def get_tokenizer(self, model_name):
        return LongformerTokenizer.from_pretrained(model_name)
    
    def chunk_text(self, text, max_chunk_size=728, chunk_overlap=20, min_chunk_size=25):
        """Split text into chunks using Longformer tokenizer."""
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        return self.fix_token_wise_length(chunks, self.tokenizer, max_chunk_size, min_chunk_size)
    
    def fix_token_wise_length(self, strs, tokenizer, max_size=728, min_size=25):
        """Adjust chunk lengths based on token count."""
        out_list = []
        tmp = ""
        s_size = 0
        last_s_size = 0
        for i, s in enumerate(strs):
            cleansed_s = s.lstrip().rstrip()
            tmp = tmp + cleansed_s
            tokens = tokenizer.tokenize(cleansed_s)
            s_size += len(tokens)
            if s_size > max_size:
                out_list.append(tmp[:tmp.rfind(" ")])
                tmp = tmp[tmp.rfind(" ") + 1:] + "\n"
                s_size = len(tokenizer.tokenize(tmp))
            elif s_size < min_size:
                if i > 0 and (last_s_size + s_size <= max_size):
                    out_list[-1] += "\n" + tmp
                    last_s_size += s_size
                    tmp = ""
                    s_size = 0
                else:
                    tmp += "\n"
            else:
                out_list.append(tmp)
                last_s_size = s_size
                tmp = ""
                s_size = 0
        if tmp:
            out_list.append(tmp)
        return out_list
    
    #duplicate 
    def find_duplicate_chunks(self, text, chunk_size=200, overlap=100):
        chunks = []
        words = word_tokenize(text)
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) >= self.min_chunk_length:
                chunks.append(chunk)
        
        duplicate_ranges = []
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i + 1:], i + 1):
                similarity = self._calculate_similarity(chunk1, chunk2)
                if similarity > self.similarity_threshold:
                    start_idx = i * (chunk_size - overlap)
                    end_idx = min((i + 1) * chunk_size, len(words))
                    duplicate_ranges.append((start_idx, end_idx))
        
        return duplicate_ranges

    def _calculate_similarity(self, text1, text2):
        """محاسبه میزان شباهت بین دو قطعه متن"""
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = (vectors * vectors.T).A[0, 1]
            return similarity
        except:
            return 0.0

    def remove_duplicate_content(self, text):
        """حذف بخش‌های تکراری از متن"""
        if not text.strip():
            return text
            
        paragraphs = text.split('\n')
        unique_paragraphs = []
        
        for para in paragraphs:
            if len(para.strip()) < self.min_chunk_length:
                unique_paragraphs.append(para)
                continue
                
            is_duplicate = False
            normalized_para = self.normalizer.normalize(para)
            
            for stored_para in self.stored_chunks:
                similarity = self._calculate_similarity(normalized_para, stored_para)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_paragraphs.append(para)
                self.stored_chunks.add(normalized_para)
        
        return '\n'.join(unique_paragraphs)
    
    
    
    def get_persian_stop_words(self):
        """Return a set of Persian stop words."""
        return {
            'و', 'در', 'به', 'از', 'که', 'می', 'این', 'است', 'را', 'با', 'های', 'برای',
            'آن', 'یک', 'شود', 'شد', 'ها', 'کرد', 'تا', 'کند', 'بر', 'بود', 'گفت',
            'نیز', 'وی', 'هم', 'کنند', 'دارد', 'ما', 'کرده', 'یا', 'اما', 'باید', 'دو',
            'اند', 'هر', 'خود', 'اگر', 'همه', 'پس', 'نمی', 'بی', 'شده', 'هیچ', 'چون',
        }

    def detect_language(self, text):
        """Detect the language of the text."""
        try:
            lang = langdetect.detect(text)
            if lang == 'fa':
                return 'Persian'
            elif lang == 'ar':
                return 'Arabic'
            elif lang == 'en':
                return 'English'
            else:
                return lang.upper()
        except:
            return 'Unknown'

    def remove_table_of_contents(self, text):
        """Remove table of contents and similar listing sections from the text."""
        lines = text.split('\n')
        cleaned_lines = []
        in_toc = False
        consecutive_toc_lines = 0
        
        toc_patterns = [
            r'^(?:\d+[-.])*\d+\s+.*?\s+\d+\s*$',
            r'^.*?\.{2,}.*?\d+\s*$',
            r'^.*?…+.*?\d+\s*$',
            r'^\s*(?:فصل|بخش|قسمت)\s+\d+\s*[:\.]\s*.*?\s+\d+\s*$',
            r'^\s*(?:فهرست|مندرجات|محتوا|محتوی|محتویات)\s*$',
            r'^\s*(?:table\s+of\s+contents|contents|index)\s*$',
            r'^.*?(?:\s*…\s*){2,}.*?$',
            r'^.*?(?:\.{3,}|…{2,}).*?$',
            r'^.*?\s*\.\s*\.\s*\.*\s*.*?$',
       
        ]
       
        

        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
                
            if re.search(r'(?:\s*…\s*){2,}', line):
                continue
                
            is_toc_line = any(re.match(pattern, line, re.IGNORECASE) for pattern in toc_patterns)
            has_number_prefix = re.match(r'^\d+[-.)]\s+.*?(?:\s+\d+)?$', line)
            
            if is_toc_line or has_number_prefix:
                consecutive_toc_lines += 1
                if consecutive_toc_lines >= 3:
                    in_toc = True
                continue
            else:
                consecutive_toc_lines = 0
                
                if in_toc:
                    if not line or len(line) < 5:
                        continue
                    else:
                        words = word_tokenize(line)
                        if len(words) > 5:
                            in_toc = False
                
                if not in_toc and line:
                    if not re.search(r'(?:\s*…\s*){2,}', line):
                        cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'^.*?(?:\s*…\s*){2,}.*?$', '', text, flags=re.MULTILINE)
        
        return text

    def remove_headers_footers(self, text):
        """Remove headers, footers, and book metadata with enhanced pattern matching."""
        lines = text.split('\n')
        cleaned_lines = []
        
        metadata_section_active = True
        consecutive_content_lines = 0
        
        
        metadata_patterns = {
            'author': [
                r'^.*?(?:نویسنده|مؤلف|تالیف|نگارنده|گردآورنده|تدوین|پدیدآور|به\s*قلم)[:\s]+.*$',
                r'^.*?(?:نوشته\s*(?:شده)?\s*(?:توسط|به\s*وسیله))[:\s]+.*$',
                r'^.*?(?:author|writer|written\s+by)[:\s]+.*$'
            ],
            'translator': [
                r'^.*?(?:مترجم|ترجمه|برگردان)[:\s]+.*$',
                r'^.*?(?:translated\s+by|translator)[:\s]+.*$'
            ],
            'publisher': [
                r'^.*?(?:ناشر|انتشارات|نشر|موسسه\s*انتشارات)[:\s]+.*$',
                r'^.*?(?:publisher|publishing|published\s+by)[:\s]+.*$'
            ],
            'isbn': [
                r'^.*?(?:شابک|isbn)[:\s]+.*$',
                r'^\s*\d{1,3}[-‐]\d{1,5}[-‐]\d{1,5}[-‐]\d{1,3}[-‐][0-9xX]'
            ],
            'price': [
                r'^.*?(?:قیمت|بها|price)[:\s]+.*?(?:\d+|رایگان).*$',
                r'^.*?(?:تومان|ریال|[£$€])\s*\d+.*$'
            ],
            'rights': [
                r'^.*?(?:کلیه\s*حقوق|حق\s*(?:چاپ|نشر|تکثیر)|copyright|\(c\)|\©).*$',
                r'^.*?(?:all\s*rights\s*reserved).*$'
            ],
            'contact': [
                r'^.*?(?:تلفن|فکس|ایمیل|پست\s*الکترونیک|تماس|نشانی|آدرس)[:\s]+.*$',
                r'^.*?(?:www\.|http|@|telephone|fax|email|address)[:\s]+.*$',
                r'^.*?(?:\+\d{10,}|\d{8,}).*$'  # شماره‌های تلفن
            ],
            'date': [
                r'^.*?(?:تاریخ|سال)\s*(?:نشر|چاپ|انتشار)[:\s]+.*$',
                r'^.*?(?:چاپ|نوبت\s*چاپ)[:\s]+.*$',
                r'^\s*\d{4}/\d{1,2}/\d{1,2}\s*$',
                r'^\s*\d{1,2}/\d{1,2}/\d{4}\s*$'
            ],
            'other_metadata': [
                r'^.*?(?:شمارگان|تیراژ|نوبت\s*چاپ|لیتوگرافی|صفحه\s*آرایی|طراح\s*جلد)[:\s]+.*$',
                r'^.*?(?:ویراستار|ویرایش|صفحه|قطع|سرشناسه|فیپا)[:\s]+.*$',
                r'^.*?(?:کد\s*کتاب|شماره\s*نشر|شماره\s*ثبت)[:\s]+.*$'
            ]
        }
        
        # الگوهای پایان بخش متادیتا
        end_metadata_patterns = [
            r'^(?:فصل|بخش|مقدمه|پیشگفتار|فهرست|درباره|چکیده)\s*(?:\d+)?[:]*\s*$',
            r'^\s*[-_=*]{3,}\s*$',  # خط جداکننده
            r'^\s*\d+\s*$',  # شماره صفحه تنها
            r'^.{200,}$'
              
        ]
        
        def is_metadata_line(line):
            """Check if a line matches any metadata pattern."""
            for patterns in metadata_patterns.values():
                if any(re.match(pattern, line, re.IGNORECASE | re.UNICODE) for pattern in patterns):
                    return True
            return False
        
        def is_end_of_metadata(line):
            """Check if a line indicates the end of metadata section."""
            return any(re.match(pattern, line, re.IGNORECASE | re.UNICODE) for pattern in end_metadata_patterns)
        
        def is_content_line(line):
            """Check if a line appears to be actual content."""
            words = len(line.split())
            return len(line) > 50 and words > 10 and not is_metadata_line(line)
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            if metadata_section_active:
                if is_end_of_metadata(line) or is_content_line(line):
                    consecutive_content_lines += 1
                    if consecutive_content_lines >= 2:
                        metadata_section_active = False
                elif is_metadata_line(line):
                    consecutive_content_lines = 0
                    continue
                else:
                    
                    if len(line) < 40 or not re.search(r'[.!؟،]', line):
                        continue
            
            if not metadata_section_active:
                if not re.match(r'^\s*\d+\s*$', line):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def clean_symbols(self, text):
        """Clean symbols while preserving mathematical expressions."""
        segments = re.split(r'(\s+)', text)
        cleaned_segments = []
        
        for segment in segments:
            if self.is_mathematical_expression(segment):
                cleaned_segments.append(segment)
            else:
                cleaned = re.sub(r'[!؟\\/|،,;:\[\](){}\"\'`](?:\s*[!؟\\/|،,;:\[\](){}\"\'`])*', ' ', segment)
                cleaned = re.sub(r'(?<!\d)\.+(?!\d)', ' ', cleaned)
                cleaned = re.sub(r'•', ' ', cleaned)
                cleaned_segments.append(cleaned)
        
        return ''.join(cleaned_segments)

    def is_mathematical_expression(self, text):
        """Check if text segment contains mathematical expressions."""
        math_patterns = [
            r'\d+[\+\-\*/÷×=≠≈<>≤≥]\d+',
            r'[\+\-]?\d*\.?\d+[eE][\+\-]?\d+',
            r'\d+/\d+',
            r'\[\d+\]',
            r'\(\d+\)',
            r'√\d+',
            r'\d+²|³',
            r'∑|∏|∫|∂|∇|∆',
            r'α|β|γ|θ|π|μ|σ|ω',
            r'sin|cos|tan|log|ln',
        ]
        return any(re.search(pattern, text) for pattern in math_patterns)

    def remove_extra_whitespace(self, text):
        """Remove extra whitespace and normalize space characters."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalize_persian_characters(self, text):
        """Normalize Persian characters and numbers."""
        replacements = {
            'ك': 'ک',
            'ي': 'ی',
            'ۀ': 'ه‌ی',
            'ة': 'ه',
            'ؤ': 'و',
            'إ': 'ا',
            'أ': 'ا',
            'ئ': 'ی',
            'ء': '',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        persian_numbers = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        for persian, english in persian_numbers.items():
            text = text.replace(persian, english)
            
        return text

    def extract_main_content(self, text):
        """Extract main content using TF-IDF scores."""
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return text
            
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            
            mean_score = np.mean(sentence_scores)
            selected_sentences = [
                sent for sent, score in zip(sentences, sentence_scores)
                if score > mean_score * 0.8
            ]
            
            return ' '.join(selected_sentences)
        except:
            return text

    def clean_text(self, text):
        """Main cleaning function with chunking capability"""
        if not text.strip():
            return "", []

        # Initial normalization
        text = self.normalizer.normalize(text)
        
        # Remove table of contents
        text = self.remove_table_of_contents(text)
        
        # Remove headers and footers
        text = self.remove_headers_footers(text)
        
        # Remove duplicate content
        text = self.remove_duplicate_content(text)
        
        # Extract main content
        text = self.extract_main_content(text)
        
        # Normalize characters
        text = self.normalize_persian_characters(text)
        
        # Clean symbols
        text = self.clean_symbols(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Fix half-space issues
        text = re.sub(r'‌+', '‌', text)
        text = re.sub(r'\s*‌\s*', '‌', text)
        
        # Create chunks from cleaned text
        chunks = self.chunk_text(text)
        
        return text, chunks

    

class PersianDocumentProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Data Cleaning App")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        self.files = []
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="آماده برای پردازش")
        self.cleaner = PersianTextCleaner()
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = ttk.Label(main_frame, text="برنامه پاكسازي داده", font=('Tahoma', 14, 'bold'))
        title_label.pack(pady=10)
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="انتخاب فایل", command=self.select_files, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="انتخاب پوشه", command=self.select_directory).pack(side=tk.LEFT)
        list_frame = ttk.LabelFrame(main_frame, text="فایل‌های انتخاب شده", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.files_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=10, font=('Tahoma', 9))
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.configure(yscrollcommand=scrollbar.set)
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, font=('Tahoma', 9))
        self.status_label.pack(pady=5)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="شروع پردازش", command=self.process_files, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="پاک کردن لیست", command=self.clear_list).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="خروج", command=self.root.quit).pack(side=tk.RIGHT)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="انتخاب فایل‌ها",
            filetypes=[
                ("همه فایل‌های پشتیبانی شده", "*.pdf;*.docx;*.doc;*.txt;*.xlsx"),
                ("PDF", "*.pdf"),
                ("Word", "*.docx;*.doc"),
                ("Text", "*.txt"),
                ("Excel", "*.xlsx")
            ]
        )
        self.add_files(files)

    def select_directory(self):
        directory = filedialog.askdirectory(title="انتخاب پوشه")
        if directory:
            path = Path(directory)
            files = []
            for ext in ['.pdf', '.docx', '.doc', '.txt', '.xlsx']:
                files.extend(path.glob(f'**/*{ext}'))
            self.add_files(files)

    def add_files(self, files):
        for file in files:
            if str(file) not in self.files:
                self.files.append(str(file))
                self.files_listbox.insert(tk.END, Path(file).name)

    def clear_list(self):
        self.files = []
        self.files_listbox.delete(0, tk.END)
        self.progress_var.set(0)
        self.status_var.set("آماده برای پردازش")

    def extract_text(self, file_path):
        """Extract text from PDF, Word, or Text file."""
        file_path = Path(file_path)
        text = ""

        if file_path.suffix.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

        elif file_path.suffix.lower() in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

        return text

    def extract_excel_data(self, file_path):
        """Extract data from Excel file and convert it to JSON."""
        df = pd.read_excel(file_path, engine="openpyxl")
        template = {"file_name": Path(file_path).stem, "languages": "eng+fas", "Chunk": None}
        state = []
        chunk_raw_size = 6
        perv_faq = ""

        for index_raw, (_, row) in enumerate(df.iterrows()):
            question = row.get("سوال", "")
            answer = row.get("پاسخ", "")
            perv_faq += f"\nQuestion : {question} -> Answer : {answer}" + "-" * 15
            if ((index_raw + 1) % chunk_raw_size) == 0:
                state.append(perv_faq.strip())
                perv_faq = ""

        template["Chunk"] = list(set(state))
        return template

    def process_files(self):
        if not self.files:
            messagebox.showwarning("هشدار", "لطفاً ابتدا فایل‌ها را انتخاب کنید.")
            return

        def process_thread():
            total_files = len(self.files)
            for i, file_path in enumerate(self.files):
                try:
                    self.status_var.set(f"در حال پردازش: {Path(file_path).name}")

                    if file_path.endswith(".xlsx"):
                        # Process Excel file
                        output_data = self.extract_excel_data(file_path)
                    else:
                        # Process text-based files
                        text = self.extract_text(file_path)
                        cleaned_text, chunks = self.cleaner.clean_text(text)
                        language = self.cleaner.detect_language(cleaned_text)
                        output_data = {
                            "filename": Path(file_path).name,
                            "language": language,
                            "chunks": chunks
                        }

                    # Save to JSON file
                    output_path = Path(file_path).with_stem(f"{Path(file_path).stem}_processed").with_suffix('.json')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)

                    # Update progress
                    progress = ((i + 1) / total_files) * 100
                    self.progress_var.set(progress)

                except Exception as e:
                    messagebox.showerror("خطا", f"خطا در پردازش {Path(file_path).name}:\n{str(e)}")

            self.status_var.set("پردازش تکمیل شد")
            messagebox.showinfo("اتمام", "پردازش همه فایل‌ها با موفقیت انجام شد.")

        threading.Thread(target=process_thread, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PersianDocumentProcessorGUI()
    app.run()