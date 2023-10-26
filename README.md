# jetson_yolo_sensor

## Instalação
### Imagem Jetson Nano Ubuntu 20
-   Obter um cartão SD de, ao menos, 32 GB para armazenar o sistema operacional 
-   Fazer o download da imagem JetsonNanoUb20_3b.img.xz (8.7 GByte!) do [Sync](https://ln5.sync.com/dl/403a73c60/bqppm39m-mh4qippt-u5mhyyfi-nnma8c4t/view/default/14418794280004)
-   Fazer o flash da imagem no cartão SD com o [balenaEtcher](https://etcher.balena.io/)

  **OBS: fazer o flash com a imagem compactada, não descompactar**
- Inserir o cartão SD na Jetson Nano
- Acessar com a senha *jetson*. Não é necessário configurar usuários, idiomas, etc.



### Instalar este pacote
OBS: Ao testar fora da Jetson, utilizar o Python 3.8.10, padrão do Ubuntu 20.04

- Abrir o terminal e digitar:
```
git clone https://github.com/gustavo-fardo/jetson_yolo_sensor.git
```
- Instalar as dependências:
```
cd jetson_yolo_sensor/
pip install -r requirements.txt
sudo apt install libgtk2.0-dev
```
## Execução
- Com parâmetros padrão:
```
cd jetson_yolo_sensor/
python yolov8_serial.py
```
- Com parâmetros personalizados (todos opcionais):
```
cd jetson_yolo_sensor/
python yolov8_serial.py --model_path <caminho do modelo> --serial-port <caminho da porta serial> --baudrate <valor do baudrate> --capture-index <caminho do video para teste (ou 0 para camera)> --show-detection <True ou False, para mostrar ou não deteccao em uma janela>
```
