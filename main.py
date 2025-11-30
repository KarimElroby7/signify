import torch
import cv2
import numpy as np
import time
import Levenshtein
from ultralytics import YOLO
from bidi.algorithm import get_display
import arabic_reshaper
from PIL import ImageFont, ImageDraw, Image

class ArabicSignLanguageRecognizer:
    def __init__(self, model_path, font_path="arial.ttf"):
        self.font = ImageFont.truetype(font_path, 40)

        # تحقق من وجود GPU واستخدامه إذا كان متاحًا
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # تحميل النموذج
        self.yolo_model = YOLO(model_path).to(self.device)

        self.franco_to_arabic = {
            'ain': 'ع', 'al': 'ال', 'aleff': 'ا', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ', 'dhad': 'ض',
            'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح', 'jeem': 'ج', 'kaaf': 'ك',
            'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ra': 'ر', 'saad': 'ص',
            'seen': 'س', 'sheen': 'ش', 'ta': 'ت', 'taa': 'ط', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة',
            'waw': 'و', 'ya': 'ي', 'yaa':  'ي', 'zay': 'ز'
        }

        self.arabic_words = [
            "مرحبا", "كيف", "حالك", "بيت", "سلام", "شكرا", "نعم", "لا", "حسنا",
            "انا", "اسمي", "كريم", "اذهب", "الي", "اكادمية", "السلاب",
            "فارس", "احمد", "دكتور", "نهلة", "مهندس", "مهندسة", "عزة", "اسلام",
            "جامعة", "كلية", "يشرح", "يتكلم", "اميرة", "مها", "ندي",
            "يأكل", "يشرب", "يكتب", "يقرأ", "يفكر", "يجلس", "يقوم", "يمشي", "يلعب", "يسمع"
        ]

        self.collected_letters = []
        self.final_word = ""
        self.min_letter_delay = 1.0
        self.last_detected_time = time.time()

    def correct_word(self, word):
        if word in self.arabic_words:
            return word
        closest_match = min(self.arabic_words, key=lambda w: Levenshtein.distance(word, w))
        return closest_match if Levenshtein.distance(word, closest_match) <= 2 else word

    def put_arabic_text(self, img, text, position, color=(0, 255, 0)):
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, bidi_text, font=self.font, fill=color)
        return np.array(img_pil)

    def process_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Failed to open camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # (Optional) Resize frame for faster processing
            # frame = cv2.resize(frame, (640, 480))

            results = self.yolo_model(frame)
            current_time = time.time()
            detected_letters = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if 0 <= class_id < len(self.franco_to_arabic):
                        letter = list(self.franco_to_arabic.keys())[class_id]
                        arabic_letter = self.franco_to_arabic.get(letter, "")

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        frame = self.put_arabic_text(frame, arabic_letter, (x1, y1 - 40), (0, 255, 0))

                        if arabic_letter and current_time - self.last_detected_time > self.min_letter_delay:
                            detected_letters.append(arabic_letter)
                            self.last_detected_time = current_time

            if detected_letters:
                self.collected_letters.extend(detected_letters)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('\r') or key == 13:  # ENTER key
                self.final_word = self.correct_word("".join(self.collected_letters))
                self.collected_letters = []

            if key == ord('\b') or key == 8:  # BACKSPACE key
                if self.collected_letters:
                    self.collected_letters.pop()

            if key == ord(' '):  # SPACE key
                self.collected_letters.append(" ")

            # Display collected letters and final word
            word_display = "".join(self.collected_letters)
            frame = self.put_arabic_text(frame, word_display, (50, 50), (255, 0, 0))
            frame = self.put_arabic_text(frame, self.final_word, (50, 100), (0, 255, 255))

            cv2.imshow("Real-Time Sign Language Recognition", frame)

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ====== Run the recognizer ======

recognizer = ArabicSignLanguageRecognizer(
    model_path=r"C:\Users\karimelroby\Desktop\signify-core\projet.pt",
    font_path=r"C:\Users\karimelroby\Desktop\signify-core\arial.ttf"
)

recognizer.process_webcam(camera_index=0)
