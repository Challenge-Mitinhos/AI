from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow_hub as hub
import tensorflow_text 
import numpy as np

app = Flask(__name__)
CORS(app)

embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual-large/2")

lista = [
    ['bateria', 'Acho que o carro está descarregado, não consigo ligá-lo.'],
    ['bateria', 'As luzes do painel nem acendem quando tento dar partida.'],
    ['bateria', 'O carro ficou parado muito tempo e agora não liga mais.'],
    ['bateria', 'O carro não liga de manhã, parece que está sem força.'],
    ['bateria', 'O carro não quer pegar de jeito nenhum.'],
    ['bateria', 'O motor não dá nem sinal de vida.'],
    ['bateria', 'Parece que o carro está sem energia.'],
    ['bateria', 'Preciso dar uma carga no carro para ele ligar.'],
    ['bateria', 'Preciso de um cabo para dar partida no carro.'],
    ['bateria', 'Quando giro a chave, o motor só faz um clique e nada acontece.'],
    ['bateria', 'O carro não liga depois que deixei os faróis acesos por muito tempo.'],
    ['bateria', 'O rádio funciona, mas o motor não responde ao tentar ligar.'],
    ['bateria', 'A bateria foi trocada recentemente, mas o carro não liga.'],
    ['bateria', 'O carro não liga quando faz frio, parece que a bateria está fraca.'],
    ['bateria', 'O painel pisca, mas o carro não dá partida.'],
    ['bateria', 'Tentei várias vezes, mas o motor não gira, acho que é a bateria.'],
    ['bateria', 'Parece que a bateria morreu, o carro não responde.'],
    ['bateria', 'A luz da bateria está acesa no painel e o carro está estranho.'],
    ['bateria', 'O carro só liga com a ajuda de um carregador portátil.'],
    ['bateria', 'O carro demora muito para pegar, mesmo quando tento várias vezes.'],
    ['bateria', 'Quando giro a chave, as luzes piscam, mas o motor não liga.'],
    ['bateria', 'A bateria parece nova, mas o carro não responde.'],
    ['bateria', 'O sistema elétrico parece funcionar, mas o motor não liga.'],
    ['bateria', 'Preciso sempre de um empurrão para o carro pegar.'],
    ['bateria', 'O carro perde força quando ligo o ar-condicionado.'],
    ['bateria', 'A luz da bateria no painel está piscando intermitentemente.'],
    ['bateria', 'O motor demora para responder, principalmente pela manhã.'],
    ['bateria', 'O carro não liga após várias tentativas, mesmo com carga suficiente.'],
    ['bateria', 'A bateria descarrega rápido quando uso dispositivos no carro.'],
    ['bateria', 'As portas travam sozinhas e o carro não dá partida.'],
    ['bateria', 'O carro morre no trânsito e não consigo ligá-lo de novo.'],
    ['bateria', 'Tentei carregar a bateria, mas não segurou a carga.'],
    ['bateria', 'As luzes internas do carro piscam quando tento ligá-lo.'],
    ['bateria', 'O painel fica completamente apagado quando tento ligar o carro.'],
    ['bateria', 'A bateria parece estar fraca, mas os componentes eletrônicos funcionam.'],
    ['bateria', 'O rádio reinicia sozinho quando tento ligar o carro.'],
    ['bateria', 'O carro não responde mesmo com os faróis funcionando.'],
    ['bateria', 'Precisei trocar a bateria, mas o problema persiste.'],
    ['bateria', 'O carro desliga sozinho no meio do caminho.'],
    ['bateria', 'O sistema de ignição parece falhar repetidamente.'],
    ['bateria', 'Mesmo com a bateria nova, o carro não liga corretamente.'],
    ['bateria', 'A bateria esquenta muito quando tento dar partida.'],
    ['bateria', 'A luz de emergência acendeu e o carro não responde.'],
    ['bateria', 'O carro só pega se deixo a chave na ignição por um tempo.'],
    ['bateria', 'A bateria descarrega quando deixo o rádio ligado por pouco tempo.'],
    ['bateria', 'A chave não responde, o carro parece sem energia.'],
    ['bateria', 'O motor dá sinais de vida, mas logo morre novamente.'],
    ['bateria', 'O carro vibra quando tento ligá-lo, mas não dá partida.'],
    
    ['motor', 'A luz de alerta acendeu no painel, e o carro está rodando estranho.'],
    ['motor', 'Ele está demorando muito para responder quando piso no acelerador.'],
    ['motor', 'O carro está perdendo potência nas subidas.'],
    ['motor', 'O carro morre do nada enquanto estou dirigindo.'],
    ['motor', 'Ouço um barulho estranho vindo da frente do carro.'],
    ['motor', 'Parece que o carro está engasgando ao rodar.'],
    ['motor', 'Sinto trepidações ao acelerar.'],
    ['motor', 'Sinto um cheiro forte de algo queimando quando dirijo.'],
    ['motor', 'O carro faz barulhos estranhos quando acelero rapidamente.'],
    ['motor', 'O motor parece estar "engasgando" ao tentar ganhar velocidade.'],
    ['motor', 'Ouço ruídos metálicos vindos do motor ao subir ladeiras.'],
    ['motor', 'O carro está falhando quando acelero.'],
    ['motor', 'O carro parece estar perdendo força ao atingir altas velocidades.'],
    ['motor', 'O motor está esquentando demais, parece que algo está errado.'],
    ['motor', 'O carro perde força ao trocar de marcha.'],
    ['motor', 'Ouço um som de batida vindo do motor quando acelero.'],
    ['motor', 'O carro está vibrando mais do que o normal, especialmente no motor.'],
    ['motor', 'O motor parece estar trabalhando pesado, mesmo quando não acelero muito.'],
    ['motor', 'O motor faz um som alto ao atingir altas velocidades.'],
    ['motor', 'O motor parece estar engasgando em baixas rotações.'],
    ['motor', 'O carro está perdendo força ao tentar acelerar em subidas.'],
    ['motor', 'Sinto um cheiro de combustível quando o motor está ligado.'],
    ['motor', 'O motor treme quando ligo o carro pela primeira vez no dia.'],
    ['motor', 'O carro acelera sozinho às vezes.'],
    ['motor', 'Sinto como se o carro estivesse engasgando quando estou parado.'],
    ['motor', 'O carro desliga sozinho quando está em ponto morto.'],
    ['motor', 'O motor não responde bem ao alternar entre marchas.'],
    ['motor', 'O motor falha repetidamente quando tento manter uma velocidade constante.'],
    ['motor', 'O carro faz um barulho de explosão ao acelerar.'],
    ['motor', 'Sinto uma vibração intensa quando o carro atinge altas rotações.'],
    ['motor', 'O motor tem dificuldade em manter a marcha lenta.'],
    ['motor', 'O carro dá solavancos ao tentar acelerar em ladeiras.'],
    ['motor', 'Há um forte cheiro de óleo queimado após dirigir por muito tempo.'],
    ['motor', 'O motor parece trabalhar mais do que o necessário em subidas leves.'],
    ['motor', 'O carro perde velocidade mesmo com o acelerador totalmente pressionado.'],
    ['motor', 'O motor parece gritar quando estou a 60 km/h.'],
    ['motor', 'O carro consome muito combustível mesmo em pequenas viagens.'],
    ['motor', 'O motor demora a responder quando faço mudanças rápidas de marcha.'],
    ['motor', 'O carro não consegue manter uma rotação estável.'],
    ['motor', 'Ouço um estalo metálico ao passar por buracos.'],
    ['motor', 'O carro desacelera sozinho em estradas planas.'],
    ['motor', 'O motor demora a desligar completamente quando giro a chave.'],
    ['motor', 'Há um atraso considerável entre o pisar no acelerador e a resposta do carro.'],
    ['motor', 'O motor parece "grilar" quando estou a baixas velocidades.'],
    ['motor', 'O carro está sem força para acelerar em estradas longas.'],
    ['motor', 'O motor tem um funcionamento irregular, às vezes falha.'],
    ['motor', 'O carro esquenta rapidamente em viagens curtas.'],
    ['motor', 'O motor está emitindo um ruído agudo constante ao rodar.'],

    ['suspensao', 'A frente do carro está mais baixa que o normal.'],
    ['suspensao', 'Está difícil controlar o carro em alta velocidade.'],
    ['suspensao', 'O carro bate seco quando passa por lombadas.'],
    ['suspensao', 'O carro está fazendo um barulho estranho quando passa por buracos.'],
    ['suspensao', 'O carro está inclinando muito para um lado nas curvas.'],
    ['suspensao', 'O carro pula muito quando passo por ruas irregulares.'],
    ['suspensao', 'Ouço um rangido vindo das rodas quando freio.'],
    ['suspensao', 'Parece que o carro está balançando demais ao dirigir.'],
    ['suspensao', 'Sinto o carro meio solto nas curvas.'],
    ['suspensao', 'A traseira do carro está fazendo barulhos ao passar por buracos.'],
    ['suspensao', 'O carro está instável ao rodar em altas velocidades.'],
    ['suspensao', 'Ouço um barulho metálico quando passo por ruas irregulares.'],
    ['suspensao', 'Sinto o carro mais baixo de um lado do que do outro.'],
    ['suspensao', 'O carro está com dificuldade de passar por lombadas sem bater.'],
    ['suspensao', 'Há um rangido constante ao dirigir em estradas esburacadas.'],
    ['suspensao', 'Sinto o volante "puxar" para um dos lados ao dirigir.'],
    ['suspensao', 'O carro está fazendo um barulho de estalo quando acelero.'],
    ['suspensao', 'O carro está com uma inclinação estranha nas curvas.'],
    ['suspensao', 'Ouço um estalo metálico ao fazer curvas.'],
    ['suspensao', 'O carro está difícil de controlar em pisos irregulares.'],
    ['suspensao', 'A frente do carro parece muito baixa quando freio.'],
    ['suspensao', 'Há um barulho de batida vindo de baixo quando passo por buracos.'],
    ['suspensao', 'O carro parece estar dançando ao dirigir em alta velocidade.'],
    ['suspensao', 'As rodas da frente estão fazendo um barulho metálico quando viro o volante.'],
    ['suspensao', 'Sinto um barulho de metal batendo quando passo por lombadas.'],
    ['suspensao', 'Ouço um rangido ao girar o volante completamente.'],
    ['suspensao', 'O carro parece estar "saltando" ao passar por buracos.'],
    ['suspensao', 'O carro está muito inclinado quando passo por curvas.'],
    ['suspensao', 'Há uma vibração nas rodas ao passar por terrenos irregulares.'],
    ['suspensao', 'O carro está inclinado, com a traseira mais alta.'],

    ['freios', 'O carro demora mais do que o normal para parar.'],
    ['freios', 'Ouço um som de raspagem quando piso no freio.'],
    ['freios', 'O pedal do freio está muito mole, parece afundar.'],
    ['freios', 'Sinto o pedal do freio vibrar quando pressiono.'],
    ['freios', 'O carro puxa para um dos lados quando freio.'],
    ['freios', 'Ouço um chiado alto ao acionar os freios.'],
    ['freios', 'O carro treme quando freio em altas velocidades.'],
    ['freios', 'Os freios fazem barulho metálico ao serem acionados.'],
    ['freios', 'O pedal do freio está muito duro, quase não consigo pressionar.'],
    ['freios', 'Há um cheiro de queimado quando uso os freios repetidamente.'],
    ['freios', 'Os freios fazem um barulho de raspagem quando acionados.'],
    ['freios', 'O freio de mão parece não estar segurando bem.'],
    ['freios', 'O carro treme muito ao frear em baixa velocidade.'],
    ['freios', 'Sinto que o carro está deslizando ao frear bruscamente.'],
    ['freios', 'O pedal do freio vai até o fundo sem resistência.'],
    ['freios', 'Ouço um estalo forte quando aciono o freio bruscamente.'],
    ['freios', 'O freio parece solto, precisa de muita força para parar o carro.'],
    ['freios', 'Há um som de chiado quando piso levemente no freio.'],
    ['freios', 'O carro dá solavancos ao frear.'],
    ['freios', 'O carro continua a se mover mesmo com o pedal totalmente pressionado.'],
    ['freios', 'O carro freia de forma irregular, parece "saltar" ao parar.'],
    ['freios', 'Ouço um estalo metálico quando piso no freio com força.'],
    ['freios', 'Sinto o pedal do freio tremer quando freio em descidas.'],
    ['freios', 'Os freios estão mais fracos do que o normal, parece que não seguram bem.'],
    ['freios', 'O freio não responde imediatamente quando pressiono o pedal.'],
    ['freios', 'Ouço um barulho metálico forte quando freio em alta velocidade.'],
    ['freios', 'O pedal do freio desce muito rápido, parece que está sem pressão.'],
    ['freios', 'Os freios demoram para responder ao pressionar o pedal.'],
    ['freios', 'Ouço um som de arranhado vindo das rodas ao frear.'],
    ['freios', 'O pedal do freio vibra em altas velocidades, especialmente em curvas.']
]


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

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    frase_usuario = data.get('userInput', '')

    resultado_inicial = obter_resultado(analisar_frase(frase_usuario))
    
    if len(resultado_inicial) == 0 or resultado_inicial[0][1] < 0.8:
        frase_usuario2 = data.get('additionalInput', '')
        frase_completa = frase_usuario + " " + frase_usuario2
        resultado_final = analisar_frase(frase_completa)
        resultado_final = obter_resultado(resultado_final)

        if len(resultado_final) == 0 or resultado_final[0][1] < 0.7:
            return jsonify({"response": "Você parece ter um problema de " + resultado_inicial[0][0] + ". Te encaminharei para um Parceiro para análise precisa."})
        else:
            return jsonify({"response": "Com base nas informações adicionais, o problema parece ser " + resultado_final[0][0]})
    else:
        return jsonify({"response": "Com base no seu primeiro input, o problema parece ser " + resultado_inicial[0][0]})

if __name__ == '__main__':
    app.run(port=5000)