import cv2                        # OpenCV: processamento de imagens
import numpy as np                  # NumPy: imagens são matrizes
import matplotlib.pyplot as plt     # matplotlib: exibir no notebook
import skimage as skimage

def mostrar(imagens, titulos, largura=15):
    # Exibe várias imagens lado a lado.
    # imagens: lista de matrizes (cinza ou BGR); titulos: lista de textos
    n = len(imagens)                                   # quantas imagens
    fig, eixos = plt.subplots(1, n, figsize=(largura, largura / n * 0.9))
    if n == 1:
        eixos = [eixos]                                # garante lista mesmo com 1 imagem
    for ax, im, t in zip(eixos, imagens, titulos):
        if im.ndim == 2:                               # 1 canal -> mapa de cinza
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:                                          # 3 canais -> converte BGR para RGB
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t)                                # título de cada painel
        ax.axis("off")                                 # sem eixos
    plt.show()



try:
    from skimage import data             # banco de imagens de exemplo do scikit-image
    moedas = data.coins()                # moedas, tons de cinza (uint8)
    pagina = data.page()                 # página de livro com sombra (uint8)
except ImportError:
    # Plano B (sem scikit-image): cria imagens sintéticas equivalentes
    moedas = np.full((300, 380), 60, np.uint8)                   # fundo escuro
    for x in range(60, 380, 90):
        for y in range(60, 300, 90):
            cv2.circle(moedas, (x, y), 30, 170, -1)              # "moedas" claras
    pagina = np.tile(np.linspace(90, 230, 380).astype(np.uint8), (190, 1))  # sombra
    cv2.putText(pagina, "Texto de exemplo", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)

print("moedas:", moedas.shape, moedas.dtype, "| página:", pagina.shape)
mostrar([moedas, pagina], ["Moedas", "Página com sombra"], largura=12)


T, binaria = cv2.threshold(
    moedas,              # src: imagem de entrada em tons de cinza
    127,                 # thresh: o limiar T
    255,                 # maxval: valor dado aos pixels acima de T
    cv2.THRESH_BINARY,   # type: acima de T -> maxval; abaixo -> 0
)                        # retorna (T usado, imagem binária)
print("T usado:", T)
print("valores na imagem binária:", np.unique(binaria))   # só 0 e 255
mostrar([moedas, binaria], ["Entrada", "Binária (T = 127)"], largura=12)




resultados, titulos = [], []
for T in [60, 127, 190]:                              # três limiares para comparar
    _, b = cv2.threshold(moedas, T, 255, cv2.THRESH_BINARY)   # _ descarta o T devolvido
    resultados.append(b)
    titulos.append(f"T = {T}")
mostrar(resultados, titulos)



hist = cv2.calcHist(
    [moedas],    # images: lista de imagens
    [0],         # channels: canal 0 (único, cinza)
    None,        # mask: None = imagem inteira
    [256],       # histSize: 256 caixas (uma por intensidade)
    [0, 256],    # ranges: faixa de valores
)

T_otsu, otsu = cv2.threshold(
    moedas,
    0,                                   # thresh: ignorado quando usamos OTSU
    255,                                 # maxval
    cv2.THRESH_BINARY + cv2.THRESH_OTSU, # type: BINARY combinado com OTSU
)
print("Otsu escolheu T =", T_otsu)

plt.figure(figsize=(10, 3.5))
plt.plot(hist)                                          # curva do histograma
plt.axvline(127, ls="--", color="gray", label="T manual = 127")
plt.axvline(T_otsu, color="orange", label=f"T de Otsu = {int(T_otsu)}")
plt.xlabel("intensidade"); plt.ylabel("nº de pixels"); plt.legend(); plt.show()

mostrar([binaria, otsu], ["Manual (127)", f"Otsu ({int(T_otsu)})"], largura=12)


gradiente = np.tile(np.linspace(0, 255, 256).astype(np.uint8), (40, 1))  # 0 -> 255
tipos = {
    "BINARY": cv2.THRESH_BINARY,          # acima -> 255 ; abaixo -> 0
    "BINARY_INV": cv2.THRESH_BINARY_INV,  # acima -> 0   ; abaixo -> 255
    "TRUNC": cv2.THRESH_TRUNC,            # acima -> T   ; abaixo -> mantém
    "TOZERO": cv2.THRESH_TOZERO,          # acima -> mantém ; abaixo -> 0
}
imgs, tits = [gradiente], ["Entrada"]
for nome, flag in tipos.items():
    _, r = cv2.threshold(gradiente, 127, 255, flag)   # mesmo T = 127 para todos
    imgs.append(r); tits.append(nome)
mostrar(imgs, tits, largura=18)



_, global_ = cv2.threshold(pagina, 127, 255, cv2.THRESH_BINARY)          # um T só
_, otsu_pg = cv2.threshold(pagina, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptativo = cv2.adaptiveThreshold(
    pagina,                           # src: imagem em cinza
    255,                              # maxValue: valor dos pixels que passam na regra
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # adaptiveMethod: média gaussiana da vizinhança
    cv2.THRESH_BINARY,                # thresholdType: BINARY ou BINARY_INV
    21,                               # blockSize: tamanho da vizinhança (ímpar)
    10,                               # C: constante subtraída da média local
)
mostrar([pagina, global_, otsu_pg, adaptativo],
        ["Entrada", "Global (127)", "Otsu", "Adaptativo"], largura=18)



rng = np.random.default_rng(0)                     # gerador aleatório com semente fixa

ruido_gauss = np.clip(
    moedas.astype(float) + rng.normal(0, 25, moedas.shape),  # soma ruído (média 0, desvio 25)
    0, 255).astype(np.uint8)                                 # volta para 0..255 e uint8

sal_pimenta = moedas.copy()
sorteio = rng.random(moedas.shape)                  # número entre 0 e 1 por pixel
sal_pimenta[sorteio < 0.04] = 0                     # 4% dos pixels viram preto (pimenta)
sal_pimenta[sorteio > 0.96] = 255                   # 4% viram branco (sal)

mostrar([moedas, ruido_gauss, sal_pimenta], ["Original", "Ruído gaussiano", "Sal e pimenta"])



M = np.array([[10, 10, 10, 80, 80, 80]] * 5, dtype=np.float32)   # 5 linhas iguais

kernel = np.ones((3, 3), np.float32) / 9            # kernel da média: 9 pesos de 1/9

# Conta manual na janela centrada na linha 2, coluna 3
janela = M[1:4, 2:5]                                 # linhas 1..3, colunas 2..4
print("janela:\n", janela)
print("média manual:", (janela * kernel).sum())      # multiplica e soma -> ~56.67

# O mesmo com o OpenCV
suave = cv2.filter2D(
    M,          # src: matriz de entrada
    -1,         # ddepth: -1 = mesmo tipo da entrada
    kernel,     # kernel: a janela de pesos
)
print("linha 2 depois da média:", np.round(suave[2]).astype(int))   # 10 -> 33 -> 57 -> 80



media = cv2.blur(
    sal_pimenta,     # src
    (5, 5),          # ksize: (largura, altura) da janela
)
gauss = cv2.GaussianBlur(
    sal_pimenta,     # src
    (5, 5),          # ksize: tamanho da janela (ímpar)
    0,               # sigmaX: espalhamento; 0 = calculado a partir do ksize
)
mediana = cv2.medianBlur(
    sal_pimenta,     # src
    5,               # ksize: um inteiro ímpar > 1 (janela 5x5)
)
mostrar([sal_pimenta, media, gauss, mediana], ["Sal e pimenta", "Média", "Gaussiano", "Mediana"], largura=18)


imgs = [cv2.GaussianBlur(moedas, (k, k), 0) for k in (3, 9, 21)]   # kernels cada vez maiores
mostrar(imgs, ["3x3", "9x9", "21x21"])

img = np.full((120, 400), 40, np.uint8)     # fundo escuro
img[:, 130:270] = 200                       # faixa clara no meio
img = cv2.GaussianBlur(img, (9, 9), 0)      # suaviza um pouco a transição

linha = img[60, :].astype(float)            # intensidades ao longo da linha 60
variacao = np.abs(np.diff(linha))           # diferença entre vizinhos (derivada)

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(linha); ax[0].set_title("Intensidade ao longo da linha (degrau)")
ax[1].plot(variacao, color="orange"); ax[1].set_title("Variação: picos nas bordas")
plt.show()


suave = cv2.GaussianBlur(moedas, (5, 5), 0)          # suaviza antes de derivar

gx = cv2.Sobel(
    suave,          # src
    cv2.CV_64F,     # ddepth: float, pois a variação pode ser negativa
    1,              # dx: ordem da derivada em x
    0,              # dy: ordem da derivada em y
    ksize=3,        # tamanho do kernel (1, 3, 5 ou 7)
)
gy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=3)       # variação em y
magnitude = np.sqrt(gx ** 2 + gy ** 2)                 # força da borda

def para_uint8(a):
    # normaliza valores quaisquer para 0..255, só para exibir
    return cv2.normalize(np.abs(a), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

mostrar([para_uint8(gx), para_uint8(gy), para_uint8(magnitude)],
        ["Sobel X", "Sobel Y", "Magnitude"])


bordas = cv2.Canny(
    suave,      # image: de preferência já suavizada
    50,         # threshold1: limiar baixo da histerese
    150,        # threshold2: limiar alto da histerese
)
mostrar([moedas, suave, bordas], ["Cinza", "Suavizada", "Canny (50, 150)"])

pares = [(10, 30), (50, 150), (150, 300)]                # (baixo, alto)
mostrar([cv2.Canny(suave, b, a) for b, a in pares], [f"Canny{p}" for p in pares])

sem = cv2.Canny(ruido_gauss, 50, 150)                          # direto na imagem ruidosa
com = cv2.Canny(cv2.GaussianBlur(ruido_gauss, (5, 5), 0), 50, 150)  # suaviza antes
mostrar([sem, com], ["Sem suavizar", "Com suavização"], largura=12)



CAMINHO = "fotos_chave.jpg"                    # <- troque pelo nome da sua imagem
minha = cv2.imread(CAMINHO, cv2.IMREAD_GRAYSCALE) # lê já em tons de cinza
if minha is None:                                 # se não encontrou o arquivo...
    print("Imagem não encontrada; usando as moedas como exemplo.")
    minha = moedas.copy()

# 1) Limiar manual x Otsu
_, m_manual = cv2.threshold(minha, 127, 255, cv2.THRESH_BINARY)
T, m_otsu = cv2.threshold(minha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("T de Otsu:", T)

# 2) Adaptativo
m_adapt = cv2.adaptiveThreshold(minha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 10)
mostrar([minha, m_manual, m_otsu, m_adapt], ["Entrada", "Manual", f"Otsu ({int(T)})", "Adaptativo"], largura=18)

# 3) Ruído + filtros
rp = minha.copy(); s = np.random.default_rng(1).random(minha.shape)
rp[s < 0.04] = 0; rp[s > 0.96] = 255
mostrar([rp, cv2.blur(rp, (5, 5)), cv2.GaussianBlur(rp, (5, 5), 0), cv2.medianBlur(rp, 5)],
        ["Com ruído", "Média", "Gaussiano", "Mediana"], largura=18)

# 4) Canny sem e com suavização, dois pares de limiares
sv = cv2.GaussianBlur(minha, (5, 5), 0)
mostrar([cv2.Canny(minha, 50, 150), cv2.Canny(sv, 50, 150), cv2.Canny(sv, 100, 200)],
        ["Sem suavizar", "Suavizada (50,150)", "Suavizada (100,200)"])