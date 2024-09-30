import tensorflow_hub as hub
import tensorflow_text 
import numpy as np
import lista_frases

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual-large/2")

lista = lista_frases.frases

problema_bateria = lista_frases.problemas_bateria
problema_freios = lista_frases.problemas_freios
problema_motor = lista_frases.problemas_motor
problema_suspensao = lista_frases.problemas_suspensao

listaf = []
for i in range(len(lista)):
    if len(lista[i]) == 3:
        x = embed(lista[i][2])
        listaf.append([lista[i][0], lista[i][1], lista[i][2], x])
    else:
        x = embed(lista[i][1])
        listaf.append([lista[i][0], lista[i][1], x])

embedded_bateria = []
embedded_freios = []
embedded_motor = []
embedded_suspensao = []

def vetorizar_frases(lista_origem, lista_destino):
    for i in range(len(lista_origem)):
        x = embed(lista_origem[i][1])
        lista_destino.append([lista_origem[i][0], lista_origem[i][1], lista_origem[i][2], x])


vetorizar_frases(problema_bateria, embedded_bateria)
vetorizar_frases(problema_freios, embedded_freios)
vetorizar_frases(problema_motor, embedded_motor)
vetorizar_frases(problema_suspensao, embedded_suspensao)

def analisar_frase(frase, lista_comparar):
    x = embed(frase)
    teste = []
    for i in lista_comparar:
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

def obter_resultado_detalhado(lista_resultados_embeds):
    faixa_maior = lista_resultados_embeds[0][1][0][0]
    faixa_menor = faixa_maior * 0.95

    resultado = []
    for i in lista_resultados_embeds:
        if faixa_menor > i[1][0][0]:
            break
        resultado.append([i[0], i[1][0][0], i[2], i[3]])
    return resultado
# frase = "A bateria desliga toda vez que uso o limpador de parabrisa"
# print(obter_resultado(analisar_frase(frase)))
# if obter_resultado(analisar_frase(frase))[0][0] == 'descrição de problema':
#     print(obter_resultado(analisar_problema(frase)))
categoria_to_embedded = {
    'bateria': embedded_bateria,
    'freios': embedded_freios,
    'motor': embedded_motor,
    'suspensão': embedded_suspensao
}

mensagem_anterior = []

def verificar_mensagem_gerar_resposta(mensagem_usuario):
    global mensagem_anterior
    resposta = None
    
    result = obter_resultado(analisar_frase(mensagem_usuario, listaf))
    resultado = result[0][0]

    if result[0][1] > 0.38:
        if mensagem_anterior and mensagem_anterior[1] == 'descrição de problema':
            categ_mensagem_anterior = mensagem_anterior[0]
            lista_embedded = categoria_to_embedded.get(categ_mensagem_anterior)
            if lista_embedded:
                resultado_problema = obter_resultado_detalhado(analisar_frase(mensagem_usuario, lista_embedded))
                resposta = f"Identifiquei que o problema parece estar relacionado a: {resultado_problema[0][0]}. O procedimento padrão para essa situação costuma ser: {resultado_problema[0][2]} e o valor médio estimado é de: R${resultado_problema[0][3]},00. Gostaria de ser encaminhado para um Centro Automotivo Parceiro para assistência?"  
                mensagem_anterior = []
        elif resultado == 'descrição de problema':
            categoria_de_problema = obter_resultado(analisar_problema(mensagem_usuario))[0][0]
            mensagem_anterior = [categoria_de_problema, resultado]
            resposta = f"Você parece ter um problema de {categoria_de_problema}, você pode me dar mais detalhes para um diagnóstico mais preciso?"
        elif resultado == 'saudação':
            resposta = "Olá, eu sou o AutoCare Bot! Como posso te ajudar?"
        elif resultado == 'positivo' or 'negativo':
            resposta = "Ok."
    else:
        resposta = "Não entendi muito bem, você pode repetir por favor?"
    
    return resposta

    

if __name__ == "__main__":
    while True:
        mensagem = input("Você: ")
        print(verificar_mensagem_gerar_resposta(mensagem))
