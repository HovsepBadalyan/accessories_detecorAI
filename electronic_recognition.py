import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


print("Загрузка модели MobileNetV2...")
model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
print("Модель успешно загружена!")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру.")
    exit()

print("Нажмите 'q' для выхода из программы.")

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр.")
        break

    
    
    resized_frame = cv2.resize(frame, (224, 224))
    
    
    img_array = np.expand_dims(resized_frame, axis=0)
    
    
    img_ready = preprocess_input(img_array)

    
    predictions = model.predict(img_ready, verbose=0)
    
    
    
    decoded_results = decode_predictions(predictions, top=1)[0]
    class_id, label, confidence = decoded_results[0]

    
    
    text = f"{label} ({confidence * 100:.1f}%)"
    
    
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    
    cv2.imshow("Electronic Equipment Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
