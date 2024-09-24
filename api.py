import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from prog_embeded import verificar_mensagem_gerar_resposta

app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.json
    mensagem_usuario = data.get("mensagem")

    if not mensagem_usuario:
        return jsonify({"erro": "Nenhuma mensagem fornecida"}), 400
    
    resposta = verificar_mensagem_gerar_resposta(mensagem_usuario)
    return jsonify({"resposta": resposta})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9001))
    app.run(debug=True, port=port)