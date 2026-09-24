import cv2                        # OpenCV
import numpy as np                  # matrizes
import matplotlib.pyplot as plt     # exibir no notebook

def mostrar(imagens, titulos, largura=15):
    # Exibe várias imagens lado a lado (cinza ou BGR).
    n = len(imagens)
    fig, eixos = plt.subplots(1, n, figsize=(largura, largura / n * 0.9))
    eixos = [eixos] if n == 1 else eixos
    for ax, im, t in zip(eixos, imagens, titulos):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)          # 1 canal
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))        # BGR -> RGB
        ax.set_title(t); ax.axis("off")
    plt.show()




try:
    from skimage import data
    moedas = data.coins()                              # 24 moedas, tons de cinza
    foto = data.astronaut()[:, :, ::-1].copy()         # RGB -> BGR (padrão do OpenCV)
except ImportError:
    # Plano B sem scikit-image: moedas sintéticas e uma imagem neutra
    moedas = np.full((300, 380), 70, np.uint8)
    for x in range(60, 380, 90):
        for y in range(60, 300, 90):
            cv2.circle(moedas, (x, y), 32, 180, -1)
    foto = np.full((300, 300, 3), 200, np.uint8)
print("moedas:", moedas.shape, "| foto:", foto.shape)
mostrar([moedas, foto], ["Moedas", "Foto"], largura=11)




suave = cv2.GaussianBlur(moedas, (5, 5), 0)        # tira ruído antes de binarizar

# Otsu: um limiar para a imagem toda
_, bin_otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptativo: um limiar por região (lida melhor com a moeda escura)
bin_adapt = cv2.adaptiveThreshold(
    suave,                            # imagem em cinza
    255,                              # valor para o objeto
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # média gaussiana da vizinhança
    cv2.THRESH_BINARY,                # objeto claro -> branco
    71,                               # blockSize: vizinhança grande (moedas são grandes)
    -5,                               # C negativo: exige pixel um pouco ACIMA da média
)

# Abertura morfológica: remove pontinhos brancos soltos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # janela redonda 5x5
binaria = cv2.morphologyEx(bin_adapt, cv2.MORPH_OPEN, kernel)

mostrar([bin_otsu, bin_adapt, binaria], ["Otsu", "Adaptativo", "Adaptativo + abertura"])

contornos, hierarquia = cv2.findContours(
    binaria,                  # imagem binária (objeto branco)
    cv2.RETR_EXTERNAL,        # mode: só os contornos externos
    cv2.CHAIN_APPROX_SIMPLE,  # method: guarda só os pontos de canto
)
print("contornos encontrados:", len(contornos))

moedas_c = [c for c in contornos if cv2.contourArea(c) > 400]   # descarta ruído pequeno
print("moedas (área > 400):", len(moedas_c))

saida = cv2.cvtColor(moedas, cv2.COLOR_GRAY2BGR)   # cópia colorida para desenhar
cv2.drawContours(
    saida,           # imagem onde desenhar
    moedas_c,        # lista de contornos
    -1,              # contourIdx: -1 = todos
    (0, 140, 255),   # cor BGR (laranja)
    2,               # espessura
)
mostrar([saida], [f"{len(moedas_c)} moedas"], largura=8)

forma = np.zeros((220, 460), np.uint8)
cv2.circle(forma, (110, 110), 80, 255, -1); cv2.circle(forma, (110, 110), 40, 0, -1)   # anel
cv2.rectangle(forma, (250, 40), (420, 180), 255, -1); cv2.rectangle(forma, (300, 80), (370, 140), 0, -1)

for nome, modo in [("RETR_EXTERNAL", cv2.RETR_EXTERNAL), ("RETR_TREE", cv2.RETR_TREE)]:
    cs, hier = cv2.findContours(forma, modo, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{nome}: {len(cs)} contornos")
# hierarquia (RETR_TREE): cada linha = [próximo, anterior, primeiro filho, pai]
print(hier[0])

quad = np.zeros((200, 200), np.uint8); cv2.rectangle(quad, (40, 40), (160, 160), 255, -1)
for nome, metodo in [("NONE", cv2.CHAIN_APPROX_NONE), ("SIMPLE", cv2.CHAIN_APPROX_SIMPLE)]:
    cs, _ = cv2.findContours(quad, cv2.RETR_EXTERNAL, metodo)
    print(f"CHAIN_APPROX_{nome}: {len(cs[0])} pontos")      # SIMPLE -> 4 cantos



c = max(moedas_c, key=cv2.contourArea)          # a maior moeda

area = cv2.contourArea(c)                        # área em px²
perim = cv2.arcLength(c, True)                   # perímetro; True = contorno fechado
x, y, w, h = cv2.boundingRect(c)                 # caixa envolvente
(cx, cy), r = cv2.minEnclosingCircle(c)          # menor círculo que contém o contorno
M = cv2.moments(c)                               # momentos da região
centro = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))   # centroide
circ = 4 * np.pi * area / perim ** 2             # circularidade (1 = círculo)

print(f"área={area:.0f}  perímetro={perim:.0f}  caixa={w}x{h}  raio={r:.1f}  circularidade={circ:.2f}")

v = cv2.cvtColor(moedas, cv2.COLOR_GRAY2BGR)
cv2.drawContours(v, [c], -1, (0, 140, 255), 2)                    # contorno
cv2.rectangle(v, (x, y), (x + w, y + h), (80, 200, 0), 2)         # caixa (verde)
cv2.circle(v, (int(cx), int(cy)), int(r), (255, 120, 0), 2)      # círculo (azul)
cv2.circle(v, centro, 4, (0, 0, 255), -1)                         # centroide (vermelho)
mostrar([v[y - 20:y + h + 20, x - 20:x + w + 20]], ["Medidas da maior moeda"], largura=5)


areas = [cv2.contourArea(c) for c in moedas_c]
circs = [4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2 for c in moedas_c]
print("área média:", round(np.mean(areas)), "| circularidade média:", round(np.mean(circs), 2))

plt.figure(figsize=(9, 3.2))
plt.hist(areas, bins=10, color="#1F4E5F")         # distribuição dos tamanhos
plt.xlabel("área (px²)"); plt.ylabel("nº de moedas"); plt.title("Tamanhos das moedas"); plt.show()

ruidosa = binaria.copy()
rng = np.random.default_rng(3)
for _ in range(60):                               # joga 60 pontinhos brancos (ruído)
    px, py = rng.integers(0, ruidosa.shape[1]), rng.integers(0, ruidosa.shape[0])
    cv2.circle(ruidosa, (int(px), int(py)), 2, 255, -1)

todos, _ = cv2.findContours(ruidosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
grandes = [c for c in todos if cv2.contourArea(c) > 400]    # limite de área
print(f"todos: {len(todos)} | área > 400: {len(grandes)}")

base = np.zeros((160, 220), np.uint8)
cv2.putText(base, "VC", (18, 125), cv2.FONT_HERSHEY_SIMPLEX, 4.2, 255, 14)   # texto branco grosso
m = rng.random(base.shape)
suja = base.copy(); suja[m > 0.97] = 255; suja[(m < 0.05) & (base == 255)] = 0  # ruído fora e furos dentro

K = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,   # shape: ELLIPSE, RECT ou CROSS
    (5, 5),              # ksize: tamanho do elemento estruturante
)
mostrar([suja,
         cv2.erode(suja, K),                              # encolhe o branco
         cv2.dilate(suja, K),                             # engorda o branco
         cv2.morphologyEx(suja, cv2.MORPH_OPEN, K),       # erosão -> dilatação
         cv2.morphologyEx(suja, cv2.MORPH_CLOSE, K)],     # dilatação -> erosão
        ["Entrada", "Erosão", "Dilatação", "Abertura", "Fechamento"], largura=17)


import math
sh = np.zeros((260, 700), np.uint8)
cv2.fillPoly(sh, [np.array([[70, 210], [150, 50], [230, 210]])], 255)          # triângulo
cv2.rectangle(sh, (280, 70), (420, 210), 255, -1)                               # quadrado
cv2.circle(sh, (570, 140), 75, 255, -1)                                         # círculo

cs, _ = cv2.findContours(sh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
v = cv2.cvtColor(sh // 4, cv2.COLOR_GRAY2BGR)
for c in sorted(cs, key=lambda c: cv2.boundingRect(c)[0]):                       # esquerda -> direita
    aprox = cv2.approxPolyDP(
        c,                                   # contorno original
        0.02 * cv2.arcLength(c, True),       # epsilon: 2% do perímetro
        True,                                # closed: contorno fechado
    )
    n = len(aprox)
    nome = {3: "triângulo", 4: "quadrado"}.get(n, "círculo" if n > 6 else f"{n} lados")
    print(f"{n} vértices -> {nome}")
    cv2.drawContours(v, [aprox], -1, (0, 140, 255), 3)
mostrar([v], ["approxPolyDP"], largura=11)

x, y, w, h = cv2.boundingRect(moedas_c[5])
modelo = moedas[y:y + h, x:x + w]                   # recorte de uma moeda = o modelo

mapa = cv2.matchTemplate(
    moedas,                  # imagem onde procurar
    modelo,                  # o que procurar
    cv2.TM_CCOEFF_NORMED,    # método: correlação normalizada (-1 a 1)
)
ys, xs = np.where(mapa >= 0.6)                     # posições com semelhança >= 0.6

v = cv2.cvtColor(moedas, cv2.COLOR_GRAY2BGR); achados = []
for (px, py) in zip(xs, ys):                       # evita marcar a mesma moeda várias vezes
    if all(abs(px - ax) > w // 2 or abs(py - ay) > h // 2 for ax, ay in achados):
        achados.append((px, py)); cv2.rectangle(v, (px, py), (px + w, py + h), (0, 140, 255), 2)
print("encontradas:", len(achados))
mostrar([modelo, v], ["Modelo", "Encontradas"], largura=10)



arq = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"   # modelo pronto do OpenCV
detector = cv2.CascadeClassifier(arq)
cinza = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)     # Haar trabalha em tons de cinza

rostos = detector.detectMultiScale(
    cinza,              # imagem em cinza
    scaleFactor=1.1,    # reduz a imagem 10% a cada escala
    minNeighbors=5,     # detecções vizinhas exigidas para confirmar
    minSize=(40, 40),   # menor rosto aceito, em pixels
)
print("rostos:", len(rostos))
v = foto.copy()
for (x, y, w, h) in rostos:                        # uma caixa por rosto
    cv2.rectangle(v, (x, y), (x + w, y + h), (0, 200, 60), 3)
mostrar([v], ["Rostos detectados"], largura=6)

def iou(a, b):
    # a e b no formato (x, y, largura, altura)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])                         # canto sup. esq. da interseção
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)                          # área da interseção
    uniao = a[2] * a[3] + b[2] * b[3] - inter                          # área da união
    return inter / uniao

real = (20, 20, 100, 100)
for prev in [(30, 30, 100, 100), (65, 30, 100, 100), (130, 30, 100, 100)]:
    print(prev, "-> IoU =", round(iou(real, prev), 2))



CAMINHO = "fotos_chave.jpg"                     # <- troque pelo nome do seu arquivo
minha = cv2.imread(CAMINHO, cv2.IMREAD_GRAYSCALE)
if minha is None:
    print("Imagem não encontrada; usando as moedas como exemplo.")
    minha = moedas.copy()

# 1) Binarize (use o melhor método da parte B; troque BINARY por BINARY_INV se o objeto for escuro)
sv = cv2.GaussianBlur(minha, (5, 5), 0)
_, b = cv2.threshold(sv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# 2) Contornos, filtro por área e contagem
cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
objs = [c for c in cs if cv2.contourArea(c) > 400]          # ajuste o limite à sua imagem
print("objetos:", len(objs))

# 3) Caixa e medidas do maior objeto
v = cv2.cvtColor(minha, cv2.COLOR_GRAY2BGR)
cv2.drawContours(v, objs, -1, (0, 140, 255), 2)
if objs:
    g = max(objs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(g)
    cv2.rectangle(v, (x, y), (x + w, y + h), (80, 200, 0), 3)
    A, P = cv2.contourArea(g), cv2.arcLength(g, True)
    print(f"maior objeto: área={A:.0f}  perímetro={P:.0f}  circularidade={4*np.pi*A/P**2:.2f}")
mostrar([b, v], ["Binária", "Objetos"], largura=12)

# 4) Rostos (use uma foto com rosto; se não houver, usa a foto de exemplo)
cor = cv2.imread(CAMINHO)
cor = foto if cor is None else cor
r = detector.detectMultiScale(cv2.cvtColor(cor, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(40, 40))
print("rostos:", len(r))