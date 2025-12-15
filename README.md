🎯 Detector de Objetos a 40cm com YOLOv8, OpenCV e TTS

Este projeto foi desenvolvido como parte do Desafio Final da DIO – Sensores Inteligentes, com o objetivo de aplicar conceitos de Visão Computacional, Inteligência Artificial e interação homem-máquina em um sistema funcional e didático.

Utilizando YOLOv8, o sistema realiza a detecção de objetos em tempo real por meio da câmera, calcula a distância aproximada do objeto até a lente e fornece feedback visual e sonoro em português quando o objeto está posicionado a aproximadamente 40 cm da câmera.


🔍 Funcionalidades principais

Detecção de objetos em tempo real

Estimativa de distância baseada em largura conhecida do objeto

Destaque visual por cores conforme a distância

Anúncio por voz (Text-to-Speech) em português

Painel informativo na tela

Sistema de calibração de objetos

Controle por teclado (voz, calibração e encerramento)


🧠 Tecnologias utilizadas

Python 3.9

YOLOv8 (Ultralytics)

OpenCV 4.5.4.58

NumPy 1.12.6

pyttsx3 (Text-to-Speech)

Ultralytics # Para YOLO V8


# Criar ambiente virtual (recomendado)
python -m venv yolo_env

yolo_env\Scripts\activate  # Windows

# Instalar as versões compatíveis

pip install numpy==1.21.6

pip install opencv-python==4.5.4.58

pip install torch torchvision torchaudio

pip install ultralytics  # Para YOLO v8

🎯 Objetivo do projeto

Criar uma solução prática que simule o uso de sensores inteligentes baseados em visão computacional, demonstrando como uma simples câmera pode atuar como sensor de distância e reconhecimento de objetos.

Este repositório faz parte da construção do meu portfólio técnico na DIO, reforçando conhecimentos em IA aplicada, IoT e visão computacional.


##🚀 Fique à vontade para explorar, testar, sugerir melhorias ou fazer um fork!
