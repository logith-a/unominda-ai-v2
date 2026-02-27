from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import base64
from groq import Groq
import time
import re
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'
CORS(app)

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 🔑 Get API key from environment variable ONLY - NEVER hardcode!
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("="*70)
    print("❌ CRITICAL ERROR: GROQ_API_KEY environment variable not set!")
    print("="*70)
    print("Please set it in your environment:")
    print("  - Local PowerShell: $env:GROQ_API_KEY='your-key'")
    print("  - Render Dashboard: Add to Environment variables")
    print("="*70)
    # No fallback - will fail gracefully

# Initialize Groq client
try:
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    else:
        client = None
        print("❌ Groq client not initialized - API key missing")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    client = None

class WebPDFChatbot:
    def __init__(self):
        self.pdf_folder = "pdfs"
        self.pages_data = []
        self.documents = ""
        self.pdf_files = []
        self.extracted_images = []
        self.image_analysis_cache = {}
        self.conversation_history = []
        self.model_stats = {
            'llama-3.3-70b-versatile': {'success': 0, 'failure': 0},
            'llama-3.1-8b-instant': {'success': 0, 'failure': 0},
            'mixtral-8x7b-instruct-v0.1': {'success': 0, 'failure': 0},
            'gemma2-9b-it': {'success': 0, 'failure': 0}
        }
        self.last_image_query = ""
        self.last_topic = ""
        self.load_pdfs()
    
    # ===== READ PDFs =====
    def extract_text_with_ocr(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_content = ""
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if len(text.strip()) < 50:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                try:
                    ocr_text = pytesseract.image_to_string(img)
                    text_content += ocr_text + "\n"
                    pages.append({
                        "page": page_num + 1,
                        "text": ocr_text,
                        "method": "OCR"
                    })
                except:
                    text_content += text + "\n"
                    pages.append({
                        "page": page_num + 1,
                        "text": text,
                        "method": "Normal (fallback)"
                    })
            else:
                text_content += text + "\n"
                pages.append({
                    "page": page_num + 1,
                    "text": text,
                    "method": "Normal"
                })
        
        return text_content, pages
    
    # ===== EXTRACT IMAGES =====
    def extract_actual_images(self, pdf_path, file_name):
        doc = fitz.open(pdf_path)
        images_found = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    images_found.append({
                        "file": file_name,
                        "page": page_num + 1,
                        "index": img_index + 1,
                        "data_base64": img_base64,
                        "mime_type": "image/png",
                        "width": pix.width,
                        "height": pix.height
                    })
                
                pix = None
        
        return images_found
    
    # ===== LOAD PDFs =====
    def load_pdfs(self):
        print("📚 Loading PDFs...")
        
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)
            print(f"📁 Created '{self.pdf_folder}' folder")
        
        self.pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        
        if not self.pdf_files:
            print("⚠️ No PDF files found")
            return
        
        for file in self.pdf_files:
            path = os.path.join(self.pdf_folder, file)
            print(f"📖 Processing: {file}")
            file_text, pages = self.extract_text_with_ocr(path)
            
            self.documents += f"\n\n----- {file} -----\n\n"
            self.documents += file_text
            
            for page in pages:
                self.pages_data.append({
                    "file": file,
                    "page": page["page"],
                    "text": page["text"],
                    "method": page["method"]
                })
        
        print("🔍 Extracting images...")
        for file in self.pdf_files:
            path = os.path.join(self.pdf_folder, file)
            images = self.extract_actual_images(path, file)
            self.extracted_images.extend(images)
            print(f"📸 {file}: {len(images)} images found")
        
        print(f"✅ Loaded {len(self.pdf_files)} PDFs, {len(self.pages_data)} pages, {len(self.extracted_images)} images")
    
    # ===== UNDERSTAND INTENT =====
    def is_greeting(self, text):
        greetings = ['hi', 'hello', 'hey', 'வணக்கம்', 'vanakkam', 'नमस्ते']
        return text.lower().strip() in greetings
    
    def is_thanks(self, text):
        thanks = ['thank', 'thanks', 'thx', 'நன்றி', 'dhanyavaad']
        return any(word in text.lower() for word in thanks)
    
    def is_how_are_you(self, text):
        phrases = ['how are you', 'how r u', 'how do you do']
        return any(phrase in text.lower() for phrase in phrases)
    
    def is_image_request(self, text):
        image_words = ['pic', 'picture', 'image', 'photo', 'show', 'படம்', 'காட்டு']
        return any(word in text.lower() for word in image_words)
    
    # ===== FIND IMAGES =====
    def find_relevant_images(self, query, max_images=4):
        if not self.extracted_images:
            return []
        
        query_lower = query.lower()
        scored = []
        
        search_query = query_lower
        if len(query_lower.split()) <= 2 and self.last_image_query:
            search_query = self.last_image_query
        elif len(query_lower.split()) <= 2 and self.last_topic:
            search_query = self.last_topic
        
        for img in self.extracted_images[:50]:
            score = 0
            img_name = img['file'].lower()
            
            if 'solder' in search_query and 'solder' in img_name:
                score += 3
            if 'bridge' in search_query and 'bridge' in img_name:
                score += 3
            if 'nwo' in search_query and 'nwo' in img_name:
                score += 3
            if 'pcb' in search_query and 'pcb' in img_name:
                score += 3
            
            if score > 0:
                scored.append((score, img))
        
        if not scored and self.extracted_images:
            return self.extracted_images[:max_images]
        
        scored.sort(reverse=True)
        return [img for score, img in scored[:max_images]]
    
    def get_image_response(self, images, query):
        if images:
            self.last_image_query = query
            img_list = [{
                "file": img['file'],
                "page": img['page'],
                "data_base64": img['data_base64'],
                "mime_type": "image/png",
                "caption": f"Page {img['page']}"
            } for img in images]
            
            if 'bridge' in query.lower():
                return f"📸 Here are {len(images)} bridging images:", img_list
            elif 'solder' in query.lower():
                return f"📸 Here are {len(images)} soldering images:", img_list
            elif 'nwo' in query.lower():
                return f"📸 Here are {len(images)} NWO images:", img_list
            else:
                return f"📸 Here are {len(images)} images:", img_list
        return "Sorry, no images found.", []
    
    # ===== SIMPLE RESPONSES =====
    def get_greeting_response(self):
        return "Hi! 👋 How can I help you today?"
    
    def get_how_are_you_response(self):
        return "I'm doing great, thanks for asking! 😊 How can I help you with SMT questions today?"
    
    def get_thanks_response(self):
        return "You're welcome! 😊 Happy to help!"
    
    # ===== EXTRACT TOPIC =====
    def extract_topic(self, text):
        text_lower = text.lower()
        if 'nwo' in text_lower:
            return 'nwo'
        elif 'solder' in text_lower:
            return 'soldering'
        elif 'bridge' in text_lower:
            return 'bridging'
        elif 'gasket' in text_lower:
            return 'gasketing'
        elif 'pcb' in text_lower:
            return 'pcb'
        elif 'wave' in text_lower:
            return 'wave soldering'
        return ''
    
    # ===== ANSWER QUESTION =====
    def ask(self, question):
        try:
            self.conversation_history.append({"role": "user", "content": question})
            print(f"📝 Question: {question}")
            
            # Check if Groq client is initialized
            if client is None:
                error_msg = "❌ Groq client not initialized. "
                if not GROQ_API_KEY:
                    error_msg += "API key is missing. Please set GROQ_API_KEY environment variable."
                else:
                    error_msg += "Check your API key."
                return {"answer": error_msg, "images": []}
            
            topic = self.extract_topic(question)
            if topic:
                self.last_topic = topic
            print(f"📌 Topic: {self.last_topic}")
            
            # Handle greetings
            if self.is_greeting(question):
                return {"answer": self.get_greeting_response(), "images": []}
            
            if self.is_how_are_you(question):
                return {"answer": self.get_how_are_you_response(), "images": []}
            
            if self.is_thanks(question):
                return {"answer": self.get_thanks_response(), "images": []}
            
            # Handle image requests
            if self.is_image_request(question):
                images = self.find_relevant_images(question, max_images=4)
                msg, img_list = self.get_image_response(images, question)
                return {"answer": msg, "images": img_list}
            
            # Prepare PDF context
            context = "Here is information from UNOMINDA manuals:\n\n"
            for i, page in enumerate(self.pages_data[:5]):
                context += f"[Page {page['page']} from {page['file']}]\n"
                context += page['text'][:500] + "\n\n"
            
            # Conversation context
            conv_context = ""
            if len(self.conversation_history) > 2:
                conv_context = "Recent chat:\n"
                for entry in self.conversation_history[-4:]:
                    role = "User" if entry["role"] == "user" else "Assistant"
                    content = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
                    conv_context += f"{role}: {content}\n"
            
            # Topic context
            topic_context = ""
            if self.last_topic and len(question.split()) <= 3:
                topic_context = f"They're asking about {self.last_topic}. Answer about that.\n"
            
            # Language detection
            lang = 'en'
            if re.search(r'[\u0B80-\u0BFF]', question):
                lang = 'ta'
            elif re.search(r'[\u0900-\u097F]', question):
                lang = 'hi'
            
            prompt = f"""{context}

{conv_context}

{topic_context}

Question: "{question}"

RULES:
1. Use SIMPLE words - like talking to a friend
2. Be FRIENDLY and use emojis 😊
3. If info is in the PDFs above, use it
4. If not in PDFs, use your knowledge to help
5. Keep answers short and sweet
6. Answer in {'Tamil' if lang == 'ta' else 'Hindi' if lang == 'hi' else 'English'}

Answer:"""
            
            # ✅ CURRENT Groq models (as of 2026)
            models_to_try = [
                'llama-3.3-70b-versatile',    # Best quality, latest Llama
                'llama-3.1-8b-instant',       # Fast, good for simple queries
                'mixtral-8x7b-instruct-v0.1', # Alternative good model
                'gemma2-9b-it'                 # Google's model on Groq
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"🎯 Trying Groq model: {model_name}")
                    
                    # Groq API call
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful SMT assistant. Use simple, friendly language with emojis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )
                    
                    if response and response.choices and response.choices[0].message.content:
                        answer = response.choices[0].message.content
                        self.model_stats[model_name]['success'] += 1
                        self.conversation_history.append({"role": "assistant", "content": answer})
                        return {"answer": answer, "images": []}
                    else:
                        print(f"⚠️ Empty response from {model_name}")
                        continue
                        
                except Exception as e:
                    self.model_stats[model_name]['failure'] += 1
                    error_str = str(e).lower()
                    print(f"❌ Error with {model_name}: {str(e)}")
                    
                    if "quota" in error_str or "429" in error_str:
                        print(f"⚠️ Rate limit for {model_name}, trying next...")
                        continue
                    elif "auth" in error_str or "api_key" in error_str:
                        return {"answer": "❌ API Key error. Please check your Groq API key.", "images": []}
                    else:
                        continue
            
            # If all models fail
            return {"answer": "I'm here to help! 😊 Please try asking again.", "images": []}
            
        except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            return {"answer": f"I'm here to help! 😊 Error: {str(e)[:100]}", "images": []}

# Initialize chatbot
chatbot = WebPDFChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.ask(message)
        
        return jsonify({
            'response': response['answer'],
            'images': response.get('images', []),
            'sources': [],
            'pdfs': chatbot.pdf_files
        })
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pdfs', methods=['GET'])
def get_pdfs():
    return jsonify({
        'pdfs': chatbot.pdf_files,
        'total_pages': len(chatbot.pages_data),
        'total_pdfs': len(chatbot.pdf_files),
        'total_images': len(chatbot.extracted_images)
    })

if __name__ == '__main__':
    print("="*70)
    print("🤖 UNOMINDA AI - GROQ VERSION")
    print("="*70)
    print(f"📊 PDFs: {len(chatbot.pdf_files)} | Pages: {len(chatbot.pages_data)} | Images: {len(chatbot.extracted_images)}")
    print("="*70)
    print("✅ CURRENT GROQ MODELS:")
    print("   • llama-3.3-70b-versatile (best quality)")
    print("   • llama-3.1-8b-instant (fast)")
    print("   • mixtral-8x7b-instruct-v0.1")
    print("   • gemma2-9b-it")
    print("="*70)
    if GROQ_API_KEY:
        print("✅ GROQ_API_KEY is set in environment")
    else:
        print("❌ GROQ_API_KEY is NOT set in environment!")
    print("="*70)
    print("🚀 Server: http://localhost:5000")
    print("="*70)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)