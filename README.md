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
- Abrir o terminal e digitar:
```
git clone https://github.com/gustavo-fardo/jetson_yolo_sensor.git
```
- Instalar as dependências:
```
cd jetson_yolo_sensor/
pip install -r requirements.txt
```
## Execução
- No terminal:
```
cd jetson_yolo_sensor/
python yolov8_serial.py
```
