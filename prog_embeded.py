import tensorflow_hub as hub
import tensorflow_text 
import numpy as np
import lista_frases

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual-large/2")

lista = lista_frases.frases

listaf = []
for i in range(len(lista)):
    if len(lista[i]) == 3:
        x = embed(lista[i][2])
        listaf.append([lista[i][0], lista[i][1], lista[i][2], x])
    else:
        x = embed(lista[i][1])
        listaf.append([lista[i][0], lista[i][1], x])


def analisar_frase(frase):
    x = embed(frase)
    teste = []
    for i in listaf:
        if len(i) == 4:
            resi = np.inner(x, i[3])
            teste.append([i[0], resi, i[1], i[2]])
        else:
            resi = np.inner(x, i[2])
            teste.append([i[0], resi, i[1]])
    lista_ordenada = sorted(teste, key=lambda x: x[1], reverse=True)
    return lista_ordenada


def analisar_problema(frase_descricao):
    x = embed(frase_descricao)
    teste = []
    for i in listaf:
        if len(i) == 4:
            resi = np.inner(x, i[3])
            teste.append([i[1], resi, i[2]])
    lista_ordenada = sorted(teste, key=lambda x: x[1], reverse=True)
    return lista_ordenada


def obter_resultado(lista_resultados_embeds):
    faixa_maior = lista_resultados_embeds[0][1][0][0]
    faixa_menor = faixa_maior * 0.95

    resultado = []
    for i in lista_resultados_embeds:
        if faixa_menor > i[1][0][0]:
            break
        resultado.append([i[0], i[1][0][0]])
    return resultado

# frase = "A bateria desliga toda vez que uso o limpador de parabrisa"
# print(obter_resultado(analisar_frase(frase)))
# if obter_resultado(analisar_frase(frase))[0][0] == 'descrição de problema':
#     print(obter_resultado(analisar_problema(frase)))

def verificar_mensagem_gerar_resposta(mensagem_usuario):
    resultado = obter_resultado(analisar_frase(mensagem_usuario))[0][0]

    if resultado == 'descrição de problema':
        categoria_de_problema = obter_resultado(analisar_problema(mensagem_usuario))[0][0]
        return f"Você parece ter um problema de {categoria_de_problema}"
    elif resultado == 'saudação':
        return "Olá, eu sou o AutoCare Bot! Como posso te ajudar?"
    elif resultado == 'positivo' or 'negativo':
        return "Ok."

print(verificar_mensagem_gerar_resposta("Olá meu querido"))
print(verificar_mensagem_gerar_resposta("A bateria desliga toda vez que uso o limpador de parabrisa"))
print(verificar_mensagem_gerar_resposta("Sim"))
