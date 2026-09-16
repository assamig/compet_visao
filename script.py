import cv2                       
import numpy as np                  
import matplotlib.pyplot as plt  

def mostrar(imagem, titulo=""):
    if imagem.ndim == 3:
        # imagem.ndim == 3 significa imagem colorida: (altura, largura, 3 canais)
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  ## OpenCV usa BGR; o matplotlib espera RGB. Convertem BGR para RGB
        plt.imshow(rgb)                                
    else:
        # imagem de 1 canal (cinza):
        plt.imshow(imagem, cmap="gray")                
    plt.title(titulo)                                  
    plt.axis("off")                                    
    plt.show()



fps = 20                            # quadros por segundo do vídeo
largura_v, altura_v = 320, 240      # tamanho de cada quadro (largura, altura)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

escritor = cv2.VideoWriter(
    "video_atividade.mp4",            # arquivo de saída
    fourcc,                         # codec definido acima
    fps,                            # taxa de quadros por segundo
    (largura_v, altura_v),          # frameSize
)

n_quadros = 60                     # quadros a gerar
for i in range(n_quadros):          
    quadro = np.full((altura_v, largura_v, 3), 255, dtype=np.uint8)   # fundo branco
    # x avança um pouco a cada quadro -> dá a sensação de movimento
    x = int(30 + (largura_v - 60) * i / n_quadros)
    cv2.circle(
        quadro,
        (x, altura_v // 2),         # center: (x que avança, meio da altura)
        20,                       
        (255, 0, 0),              # color BGR: azul
        thickness=-1,               # preenchido
    )
    escritor.write(quadro)          # grava este quadro no arquivo de vídeo

escritor.release()                  # fecha o arquivo
print("Vídeo criado com", n_quadros, "quadros.")


cap = cv2.VideoCapture(
    "video_atividade.mp4",            
)
print("Vídeo aberto?", cap.isOpened())  

# Cada constante CAP_PROP_* identifica um metadado do vídeo:
fps_lido = cap.get(cv2.CAP_PROP_FPS)            # quadros por segundo
total    = cap.get(cv2.CAP_PROP_FRAME_COUNT)    # total de quadros (pode ser aproximado)
larg     = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # largura do quadro em pixels
alt      = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   # altura do quadro em pixels
print(f"fps={fps_lido} | total de quadros={total} | tamanho={larg}x{alt}")


quadros = []                        # lista para guardar cada quadro lido
while True:                         # repete até o vídeo terminar
    ret, quadro = cap.read()        # ret: True se leu um quadro; quadro: a imagem (ou None)
    if not ret:                     # ret == False -> não há mais quadros
        break                       # encerra o laço
    quadros.append(quadro)          # guarda o quadro lido
cap.release()                       # libera o arquivo de vídeo

total_lidos = len(quadros)          # quantos quadros realmente lemos
print("Quadros lidos:", total_lidos)

indice_meio = total_lidos // 2      # // divisão inteira -> índice do quadro central
cv2.imwrite(
    "frame_meio.jpg",               # arquivo de saída
    quadros[indice_meio],           # o quadro do meio da lista
)
print("Salvo frame_meio.jpg (quadro", indice_meio, "de", total_lidos, ")")
mostrar(quadros[indice_meio], "Quadro do meio do vídeo")

##COMENTÁRIOS SOBRE O SCRIPT:##

#O vídeo é representado por quadros definidos, 
#na qual os pixels desses quadros são feitos em um loop que vai servir para criar a forma de círculo 
#e a movimentação dele no vídeo e no final com o quadro central do vídeo sendo captado por uma função que
#exibe esse mesmo quadro como uma imagem que foi feito por 3 canais de captura para uma imagem colorida.