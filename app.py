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

class PersianTextCleaner:
    def __init__(self):
        self.normalizer = Normalizer()
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=self.get_persian_stop_words())
         # تنظیمات تشخیص محتوای تکراری
        self.similarity_threshold = 0.8
        self.min_chunk_length = 100
        self.stored_chunks = set()
    
    
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
            
        # تقسیم متن به پاراگراف‌ها
        paragraphs = text.split('\n')
        unique_paragraphs = []
        
        for para in paragraphs:
            if len(para.strip()) < self.min_chunk_length:
                unique_paragraphs.append(para)
                continue
                
            # بررسی شباهت با پاراگراف‌های قبلی
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
        
        # شمارنده برای تشخیص بخش متادیتا
        metadata_section_active = True
        consecutive_content_lines = 0
        
        # الگوهای متادیتا با دقت بالاتر
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
            r'^.{200,}$'  # خط طولانی (احتمالاً متن اصلی)
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
            # خط باید طولانی باشد و شامل چندین کلمه
            words = len(line.split())
            return len(line) > 50 and words > 10 and not is_metadata_line(line)
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # بررسی پایان بخش متادیتا
            if metadata_section_active:
                if is_end_of_metadata(line) or is_content_line(line):
                    consecutive_content_lines += 1
                    if consecutive_content_lines >= 2:
                        metadata_section_active = False
                elif is_metadata_line(line):
                    consecutive_content_lines = 0
                    continue
                else:
                    # خطوط کوتاه یا مشکوک در بخش متادیتا را حذف می‌کنیم
                    if len(line) < 40 or not re.search(r'[.!؟،]', line):
                        continue
            
            # اگر از بخش متادیتا خارج شدیم، خط را نگه می‌داریم
            if not metadata_section_active:
                # حذف خطوط تکی عدد (شماره صفحه)
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
        """تابع اصلی پاکسازی با قابلیت حذف محتوای تکراری"""
        if not text.strip():
            return ""

        # نرمال‌سازی اولیه
        text = self.normalizer.normalize(text)
        
        # حذف فهرست مطالب
        text = self.remove_table_of_contents(text)
        
        # حذف سرصفحه و پاصفحه
        text = self.remove_headers_footers(text)
        
        # حذف محتوای تکراری
        text = self.remove_duplicate_content(text)
        
        # استخراج محتوای اصلی
        text = self.extract_main_content(text)
        
        # نرمال‌سازی کاراکترها
        text = self.normalize_persian_characters(text)
        
        # پاکسازی نمادها
        text = self.clean_symbols(text)
        
        # پاکسازی فضاهای خالی
        text = self.remove_extra_whitespace(text)
        
        # رفع مشکل نیم‌فاصله
        text = re.sub(r'‌+', '‌', text)
        text = re.sub(r'\s*‌\s*', '‌', text)
        
        return text

class PersianDocumentProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("برنامه پاکسازی متن فارسی")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # تنظیم فونت برای نمایش صحیح فارسی
        default_font = ('Tahoma', 10)
        self.root.option_add('*Font', default_font)
        
        # متغیرهای مورد نیاز
        self.files = []
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="آماده برای پردازش")
        
        # Initialize processor components
        self.cleaner = PersianTextCleaner()
        
        self.create_widgets()
        
    def create_widgets(self):
        # ایجاد فریم اصلی
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # عنوان برنامه
        title_label = ttk.Label(
            main_frame, 
            text="پردازشگر متون فارسی", 
            font=('Tahoma', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # دکمه‌های انتخاب فایل
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            file_frame, 
            text="انتخاب فایل", 
            command=self.select_files,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            file_frame, 
            text="انتخاب پوشه", 
            command=self.select_directory
        ).pack(side=tk.LEFT)
        
        # لیست فایل‌ها
        list_frame = ttk.LabelFrame(main_frame, text="فایل‌های انتخاب شده", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.files_listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.EXTENDED, 
            height=10,
            font=('Tahoma', 9)
        )
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.configure(yscrollcommand=scrollbar.set)
        
        # نوار پیشرفت
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X)
        
        # برچسب وضعیت
        self.status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            font=('Tahoma', 9)
        )
        self.status_label.pack(pady=5)
        
        # دکمه‌های عملیات
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame, 
            text="شروع پردازش", 
            command=self.process_files,
            style='Accent.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="پاک کردن لیست", 
            command=self.clear_list
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame, 
            text="خروج", 
            command=self.root.quit
        ).pack(side=tk.RIGHT)

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="انتخاب فایل‌ها",
            filetypes=[
                ("همه فایل‌های پشتیبانی شده", "*.pdf;*.docx;*.doc;*.txt"),
                ("PDF", "*.pdf"),
                ("Word", "*.docx;*.doc"),
                ("Text", "*.txt")
            ]
        )
        self.add_files(files)

    def select_directory(self):
        directory = filedialog.askdirectory(title="انتخاب پوشه")
        if directory:
            path = Path(directory)
            files = []
            for ext in ['.pdf', '.docx', '.doc', '.txt']:
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
        """Extract text from PDF or Word file."""
        file_path = Path(file_path)
        text = ""
        
        try:
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
        except Exception as e:
            raise Exception(f"خطا در خواندن فایل: {str(e)}")

    def process_files(self):
        if not self.files:
            messagebox.showwarning("هشدار", "لطفاً ابتدا فایل‌ها را انتخاب کنید.")
            return
            
        def process_thread():
            total_files = len(self.files)
            for i, file_path in enumerate(self.files):
                try:
                    self.status_var.set(f"در حال پردازش: {Path(file_path).name}")
                    
                    # استخراج متن
                    text = self.extract_text(file_path)
                    
                    # تمیز کردن متن
                    cleaned_text = self.cleaner.clean_text(text)
                    
                    # ذخیره در همان مسیر با پسوند _cleaned
                    output_path = Path(file_path)
                    output_path = output_path.with_stem(f"{output_path.stem}_cleaned")
                    output_path = output_path.with_suffix('.json')
                    
                    # ذخیره نتیجه
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    
                    # بروزرسانی پیشرفت
                    progress = ((i + 1) / total_files) * 100
                    self.progress_var.set(progress)
                    
                except Exception as e:
                    messagebox.showerror("خطا", f"خطا در پردازش {Path(file_path).name}:\n{str(e)}")
            
            self.status_var.set("پردازش تکمیل شد")
            messagebox.showinfo("اتمام", "پردازش همه فایل‌ها با موفقیت انجام شد.")
        
        # شروع پردازش در thread جداگانه
        threading.Thread(target=process_thread, daemon=True).start()

    def run(self):
        # تنظیم استایل دکمه‌ها
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Tahoma', 10, 'bold'))
        self.root .mainloop()
        
if __name__ == "__main__":
    app = PersianDocumentProcessorGUI()
    app.run()
    