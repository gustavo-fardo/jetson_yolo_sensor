# jetson_yolo_sensor

## Instalação
### Imagem Jetson Nano Ubuntu 20
-   Obter um cartão SD de, ao menos, 32 GB para armazenar o sistema operacional 
-   Fazer o download da imagem JetsonNanoUb20_3b.img.xz (8.7 GByte!) do [Sync](https://ln5.sync.com/dl/403a73c60/bqppm39m-mh4qippt-u5mhyyfi-nnma8c4t/view/default/14418794280004)
-   Fazer o flash da imagem no cartão SD com o [balenaEtcher](https://etcher.balena.io/)

  ***OBS: fazer o flash com a imagem compactada, não descompactar***
- Inserir o cartão SD na Jetson Nano
- Acessar com a senha *jetson*. Não é necessário configurar usuários, idiomas, etc.



### Instalar este pacote
***OBS: Ao testar fora da Jetson, utilizar o Python 3.8.10, padrão do Ubuntu 20.04***

- Abrir o terminal e digitar:
```
git clone https://github.com/gustavo-fardo/jetson_yolo_sensor.git
```
- Instalar as dependências:
```
cd jetson_yolo_sensor/
python3 -m pip install -r requirements.txt
sudo apt install libgtk2.0-dev
```
## Execução
- Com parâmetros padrão:
```
cd jetson_yolo_sensor/
python3 yolov8_serial.py
```
- Com parâmetros personalizados (todos opcionais):
    - Detalhamento de cada parâmetro:
      ```
      --model_path <caminho do modelo>
      --serial-port <caminho da porta serial>
      --baudrate <valor do baudrate>
      --capture-index <caminho do vídeo para teste | 'csi' para csi-camera | 0 para câmera usb)>
      --show-detection <True ou False, para mostrar ou não detecção em uma janela>
      ```
    - Rodar com parâmetros escolhidos
      ```
      cd jetson_yolo_sensor/
      python3 yolov8_serial.py --<nome parâmetro 1> <valor parâmetro 1> --<nome parâmetro 2> <valor parâmetro 2> ...
      ```
