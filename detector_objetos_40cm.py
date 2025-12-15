import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time

class ObjectDistanceDetector:
    def __init__(self, model_name='yolov8n.pt'):
        print("🚀 Inicializando Detector de Objetos a 40cm...")
        
        # Carrega o modelo YOLO
        self.model = YOLO(model_name)
        print(f"✅ Modelo {model_name} carregado!")
        
        # Configura síntese de voz
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        print("✅ Síntese de voz configurada")
        
        # Distância alvo em cm
        self.target_distance = 40
        
        # Larguras conhecidas dos objetos específicos (em cm)
        self.known_widths = {
            'cell phone': 7.5,      # Celular
            'scissors': 15.0,       # Tesoura
            'glasses': 14.0,        # Óculos
            'wrist watch': 5.0,     # Relógio de pulso
            # Objetos adicionais para melhor detecção
            'person': 40,
            'bottle': 8,
            'cup': 8,
            'book': 15,
            'laptop': 30,
            'mouse': 8,
            'keyboard': 35,
            'chair': 45,
            'remote': 15,
        }
        
        # Mapeamento de nomes em português
        self.portuguese_names = {
            'cell phone': 'celular',
            'scissors': 'tesoura', 
            'glasses': 'óculos',
            'wrist watch': 'relógio',
            'person': 'pessoa',
            'bottle': 'garrafa',
            'cup': 'copo',
            'book': 'livro',
            'laptop': 'notebook',
            'mouse': 'mouse',
            'keyboard': 'teclado',
            'chair': 'cadeira',
            'remote': 'controle'
        }
        
        # Focal length (ajuste conforme sua câmera)
        self.focal_length = 700
        
        # Controle de anúncios
        self.last_announcement = {}
        self.cooldown = 3  # segundos
        
        print(f"🎯 Configurado para detectar objetos a {self.target_distance}cm")
        print("📱 Objetos principais: celular, tesoura, óculos, relógio")
    
    def get_portuguese_name(self, english_name):
        """Retorna o nome em português do objeto"""
        return self.portuguese_names.get(english_name, english_name)
    
    def speak(self, text):
        """Faz o sistema falar o texto"""
        print(f"🔊 {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def calculate_distance(self, pixel_width, object_name):
        """Calcula distância baseada na largura do objeto em pixels"""
        if object_name in self.known_widths:
            known_width = self.known_widths[object_name]
            if pixel_width > 0:
                distance = (known_width * self.focal_length) / pixel_width
                return distance
        return None
    
    def is_at_target_distance(self, distance):
        """Verifica se está na distância alvo (40cm) com margem de erro"""
        if distance is None:
            return False
        return abs(distance - self.target_distance) <= 8  # Margem de ±8cm
    
    def process_frame(self, frame):
        """Processa o frame e detecta objetos"""
        # Executa detecção
        results = self.model(frame, conf=0.5, verbose=False)
        
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Foca apenas nos objetos de interesse quando estão com boa confiança
                    if confidence > 0.6:
                        # Coordenadas da bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calcula largura em pixels
                        pixel_width = x2 - x1
                        
                        # Calcula distância
                        distance = self.calculate_distance(pixel_width, class_name)
                        
                        # Verifica se está na distância alvo
                        at_target_distance = self.is_at_target_distance(distance)
                        
                        detection = {
                            'name': class_name,
                            'portuguese_name': self.get_portuguese_name(class_name),
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'distance': distance,
                            'at_target_distance': at_target_distance,
                            'pixel_width': pixel_width
                        }
                        
                        detections.append(detection)
                        
                        # Desenha na imagem
                        self.draw_detection(frame, detection)
        
        return detections, frame
    
    def draw_detection(self, frame, detection):
        """Desenha a detecção no frame com informações de distância"""
        name = detection['name']
        portuguese_name = detection['portuguese_name']
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['bbox']
        distance = detection['distance']
        at_target = detection['at_target_distance']
        
        # Cor baseada na distância (verde se estiver na distância alvo)
        if at_target:
            color = (0, 255, 0)  # Verde - na distância correta
        elif distance and distance < self.target_distance:
            color = (0, 255, 255)  # Amarelo - muito perto
        elif distance and distance > self.target_distance:
            color = (0, 165, 255)  # Laranja - muito longe
        else:
            color = (255, 0, 0)  # Vermelho - distância desconhecida
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Texto com informações
        if distance is not None:
            distance_text = f"{distance:.1f}cm"
            if at_target:
                status = "🎯 NA DISTÂNCIA!"
            elif distance < self.target_distance:
                status = f"↔️ AFASTE {self.target_distance - distance:.1f}cm"
            else:
                status = f"🔍 APROXIME {distance - self.target_distance:.1f}cm"
        else:
            status = "📏 Distância desconhecida"
        
        label = f"{portuguese_name} {confidence:.2f}"
        distance_label = f"{status}"
        
        # Fundo para o texto principal
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 25), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Fundo para a distância
        dist_size = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(frame, (x1, y1 - dist_size[1] - 5), 
                     (x1 + dist_size[0], y1 - label_size[1] - 20), color, -1)
        
        # Texto principal
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Texto da distância
        cv2.putText(frame, distance_label, (x1, y1 - label_size[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def announce_detections(self, detections):
        """Anuncia objetos que estão na distância alvo"""
        current_time = time.time()
        
        for detection in detections:
            if detection['at_target_distance'] and detection['confidence'] > 0.7:
                portuguese_name = detection['portuguese_name']
                distance = detection['distance']
                
                # Verifica cooldown
                if (portuguese_name not in self.last_announcement or 
                    current_time - self.last_announcement[portuguese_name] > self.cooldown):
                    
                    announcement = f"{portuguese_name} detectado a {distance:.1f} centímetros"
                    self.speak(announcement)
                    self.last_announcement[portuguese_name] = current_time
    
    def add_info_panel(self, frame, detections_count, target_detections):
        """Adiciona painel de informações na imagem"""
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Informações
        info_lines = [
            f"Objetos detectados: {detections_count}",
            f"Na distância {self.target_distance}cm: {target_detections}",
            "🎯 OBJETOS PRINCIPAIS:",
            "  • Celular • Tesoura • Óculos • Relógio",
            "Pressione 'Q' para sair | 'C' para calibrar",
            "ESPAÇO: Ativar/desativar voz"
        ]
        
        colors = [
            (255, 255, 255),  # Branco
            (0, 255, 0) if target_detections > 0 else (255, 255, 255),  # Verde se tem objetos
            (255, 255, 0),    # Amarelo para título
            (255, 255, 0),    # Amarelo para objetos
            (255, 255, 255),  # Branco
            (255, 255, 255)   # Branco
        ]
        
        for i, (line, color) in enumerate(zip(info_lines, colors)):
            cv2.putText(frame, line, (10, 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def calibrate_for_object(self, object_name, real_width_cm):
        """Calibra para um objeto específico"""
        print(f"\n🎯 Calibrando para: {object_name}")
        print(f"📏 Largura real: {real_width_cm}cm")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Câmera não disponível")
            return
        
        print(f"Posicione o {object_name} a {self.target_distance}cm da câmera")
        print("Pressione 'S' quando estiver pronto...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecta objetos
            results = self.model(frame, conf=0.6, verbose=False)
            
            temp_frame = frame.copy()
            cv2.putText(temp_frame, f"CALIBRACAO: {object_name}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(temp_frame, f"Posicione a {self.target_distance}cm - Largura: {real_width_cm}cm", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(temp_frame, "Pressione 'S' para salvar, 'ESC' para cancelar", 
                       (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Mostra detecções atuais
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        
                        if cls_name == object_name and conf > 0.5:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            pixel_width = x2 - x1
                            
                            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(temp_frame, f"{cls_name} {pixel_width}px", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow('Calibracao', temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Atualiza a largura conhecida
                self.known_widths[object_name] = real_width_cm
                print(f"✅ {object_name} calibrado com largura {real_width_cm}cm")
                break
            elif key == 27:  # ESC
                print("❌ Calibração cancelada")
                break
        
        cap.release()
        cv2.destroyWindow('Calibracao')
    
    def run(self):
        """Executa o detector principal"""
        print("🎥 Iniciando câmera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Erro: Não foi possível acessar a câmera!")
            return
        
        # Configurações da câmera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Câmera inicializada!")
        print(f"\n🎯 OBJETIVO: Posicione objetos a {self.target_distance}cm da câmera")
        print("📱 Objetos principais: celular, tesoura, óculos, relógio")
        print("\n🎮 CONTROLES:")
        print("   - Q: Sair")
        print("   - C: Calibrar objetos")
        print("   - ESPAÇO: Ativar/desativar voz")
        
        voice_enabled = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erro ao capturar frame")
                    break
                
                # Processa detecções
                detections, processed_frame = self.process_frame(frame)
                
                # Conta objetos na distância alvo
                target_detections = sum(1 for d in detections if d['at_target_distance'])
                
                # Adiciona informações
                self.add_info_panel(processed_frame, len(detections), target_detections)
                
                # Anuncia detecções (se voz ativada)
                if voice_enabled and target_detections > 0:
                    self.announce_detections(detections)
                
                # Mostra frame
                cv2.imshow(f'Detector - Objetos a {self.target_distance}cm', processed_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("\n🔧 MENU DE CALIBRAÇÃO:")
                    print("1. Celular (largura padrão: 7.5cm)")
                    print("2. Tesoura (largura padrão: 15cm)")
                    print("3. Óculos (largura padrão: 14cm)")
                    print("4. Relógio (largura padrão: 5cm)")
                    
                    choice = input("Escolha o objeto para calibrar (1-4) ou Enter para cancelar: ")
                    if choice == '1':
                        self.calibrate_for_object('cell phone', 7.5)
                    elif choice == '2':
                        self.calibrate_for_object('scissors', 15.0)
                    elif choice == '3':
                        self.calibrate_for_object('glasses', 14.0)
                    elif choice == '4':
                        self.calibrate_for_object('wrist watch', 5.0)
                
                elif key == ord(' '):
                    voice_enabled = not voice_enabled
                    status = "ATIVADA" if voice_enabled else "DESATIVADA"
                    print(f"🔊 Voz {status}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrompido pelo usuário")
        except Exception as e:
            print(f"❌ Erro: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Programa finalizado")

def main():
    print("=" * 70)
    print("           🎯 DETECTOR DE OBJETOS A 40cm")
    print("=" * 70)
    print("🔍 Detecta objetos específicos e anuncia quando estão a ~40cm")
    print("📱 Objetos: Celular, Tesoura, Óculos, Relógio")
    print("🗣️  Anuncia em português o nome do objeto e distância")
    
    detector = ObjectDistanceDetector('yolov8n.pt')
    detector.run()

if __name__ == "__main__":
    main()