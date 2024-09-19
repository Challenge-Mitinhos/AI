import tensorflow_hub as hub
import tensorflow_text 
import numpy as np
import lista_frases

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual-large/2")

lista = lista_frases.frases


listaf = []
for i in range(len(lista)):
    x = embed(lista[i][1])
    listaf.append([lista[i][0], lista[i][1], x])

def analisar_frase(frase):
    x = embed(frase)
    teste = []
    for i in listaf:
        resi = np.inner(x, i[2])
        teste.append([i[0], resi, i[1]])

    sorted_list = sorted(teste, key=lambda x: x[1], reverse=True)
    return sorted_list

def obter_resultado(final_result):
    faixa_maior = final_result[0][1][0][0]
    faixa_menor = faixa_maior * 0.95

    resultado = []
    for i in final_result:
        if faixa_menor > i[1][0][0]:
            break
        resultado.append([i[0], i[1][0][0]])

    return resultado

def conversar():
    frase_usuario = input("Diga o problema do seu veículo: ")
    resultado_inicial = obter_resultado(analisar_frase(frase_usuario))

    print(resultado_inicial)
    if resultado_inicial[0][0] == 'saudação':
        print("Saudações!")
    else: 
        if len(resultado_inicial) == 0 or resultado_inicial[0][1] < 0.75:
            frase_usuario2 = input("Dê mais detalhes ou digite o seu problema de outra forma: ")
            frase_completa = frase_usuario + " " + frase_usuario2
            resultado_final = analisar_frase(frase_completa)
            resultado_final = obter_resultado(resultado_final)

            if len(resultado_final) == 0 or resultado_final[0][1] < 0.7:
                print("Você parece ter um problema de", resultado_inicial[0], ". Te encaminharei para um Parceiro para análise precisa.")
            else:
                print("Com base nas informações adicionais, o problema parece ser", resultado_final[0])
        else:
            print("Com base no seu input, o problema parece ser", resultado_inicial[0])

while True:
    conversar()
